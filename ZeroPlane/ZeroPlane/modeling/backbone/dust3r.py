import torch
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
import sys
from pathlib import Path
_backbone_dir = Path(__file__).resolve().parent          # .../ZeroPlane/ZeroPlane/modeling/backbone/
_zeroplane_root = _backbone_dir.parent.parent.parent     # .../ZeroPlane/  (the submodule root)
_dust3r_root = _zeroplane_root / 'third_party' / 'dust3r'
_croco_root = _dust3r_root / 'croco'
for _p in [str(_dust3r_root), str(_croco_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# STDC's top-level `models/` package (model_stages, etc.) conflicts with
# croco's `models/` package (dpt_block, etc.) because it is already cached
# in sys.modules when this backbone module is imported.  Temporarily evict
# all `models.*` entries so croco can find its own `models/dpt_block`.
# We restore STDC's entries immediately after.
_stdc_models_cache = {k: v for k, v in sys.modules.items()
                      if k == 'models' or k.startswith('models.')}
for _k in _stdc_models_cache:
    del sys.modules[_k]
try:
    from dust3r.model import AsymmetricCroCo3DStereo
finally:
    sys.modules.update(_stdc_models_cache)

@BACKBONE_REGISTRY.register()
class Dust3REncoderDecoder(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.croco3d_encoder_decoder = AsymmetricCroCo3DStereo.from_pretrained(cfg.MODEL.DUST3R.MODEL_NAME).to('cuda')
        self._out_features = cfg.MODEL.DUST3R.OUT_FEATURES

        out_channels = cfg.MODEL.DUST3R.OUT_CHANNELS

        # not taking effect
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32
        }

        # not taking effect
        self._out_feature_channels = {
            "res2": out_channels[0],
            "res3": out_channels[1],
            "res4": out_channels[2],
            "res5": out_channels[3],
        }

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        pairs = (x.to('cuda', non_blocking=True), x.to('cuda', non_blocking=True))

        pred1, _ = self.croco3d_encoder_decoder(pairs[0], pairs[1], feature_only=True)

        features = []
        for feat in pred1['feat']:
            features.append(feat)

        res = {}
        for idx, feat in enumerate(features):
            res['res' + str(idx + 2)] = feat

        return res
