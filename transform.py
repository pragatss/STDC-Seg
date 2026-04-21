#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np
import cv2


def _st_resize(st, new_h, new_w):
    """Resize a (C, H, W) float numpy array with bilinear interpolation."""
    if st is None:
        return None
    # cv2.resize operates on (H, W) or (H, W, C); transpose for multi-channel.
    st_hwc = st.transpose(1, 2, 0).astype(np.float32)
    resized = cv2.resize(st_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:          # single-channel edge-case
        resized = resized[:, :, np.newaxis]
    return resized.transpose(2, 0, 1).astype(st.dtype)


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        st = im_lb.get('st', None)
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h):
            return dict(im=im, lb=lb, st=st)

        if w < W or h < H:
            pad_scale = float(W) / w if w < h else float(H) / h
            w, h = int(pad_scale * w + 1), int(pad_scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
            if st is not None:
                new_st_h = max(1, int(st.shape[1] * pad_scale + 1))
                new_st_w = max(1, int(st.shape[2] * pad_scale + 1))
                st = _st_resize(st, new_st_h, new_st_w)

        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H

        cropped_st = None
        if st is not None:
            # st is kept at 1/8 of the current image dimensions by RandomScale.
            # Map image-space crop coords to st-space (divide by 8).
            scale_y = st.shape[1] / h
            scale_x = st.shape[2] / w
            st_y1 = int(int(sh) * scale_y)
            st_x1 = int(int(sw) * scale_x)
            st_crop_h = max(1, int(H * scale_y))
            st_crop_w = max(1, int(W * scale_x))
            st_y2 = min(st_y1 + st_crop_h, st.shape[1])
            st_x2 = min(st_x1 + st_crop_w, st.shape[2])
            cropped = st[:, st_y1:st_y2, st_x1:st_x2]
            # If the crop is too small (very aggressive down-scale), keep full st
            # and let F.interpolate handle the resize inside the loss.
            cropped_st = cropped if (cropped.shape[1] >= 4 and cropped.shape[2] >= 4) else st

        return dict(
            im=im.crop(crop),
            lb=lb.crop(crop),
            st=cropped_st,
        )


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        im = im_lb['im']
        lb = im_lb['lb']
        st = im_lb.get('st', None)
        flipped_st = st[:, :, ::-1].copy() if st is not None else None
        return dict(
            im=im.transpose(Image.FLIP_LEFT_RIGHT),
            lb=lb.transpose(Image.FLIP_LEFT_RIGHT),
            st=flipped_st,
        )


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        st = im_lb.get('st', None)
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        scaled_st = None
        if st is not None:
            new_st_h = max(1, int(st.shape[1] * scale))
            new_st_w = max(1, int(st.shape[2] * scale))
            scaled_st = _st_resize(st, new_st_h, new_st_w)
        return dict(
            im=im.resize((w, h), Image.BILINEAR),
            lb=lb.resize((w, h), Image.NEAREST),
            st=scaled_st,
        )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        # ColorJitter is spatial-invariant — pass soft target through unchanged.
        return dict(im=im, lb=lb, st=im_lb.get('st', None))


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb




if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
