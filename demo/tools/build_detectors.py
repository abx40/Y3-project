#!/usr/bin/env python3
"""Build the deepfake-detector models the demo loads from demo/models/detectors/.

Downloads DeepfakeBench's public checkpoints (trained on FaceForensics++) and the
matching DeepfakeBench model code, converts each detector to ONNX with a single
output, P(fake), and stores the large ones with half-size weights.

Usage:
    pip install torch efficientnet_pytorch onnx onnxruntime numpy pyyaml
    python3 demo/tools/build_detectors.py

Add the project's own fine-tuned F3Net (appears in the page as "Project F3Net"):
    python3 demo/tools/build_detectors.py --f3net-checkpoint path/to/f3net_finetuned_best.pth

Checkpoints are loaded with torch.load(weights_only=True), so no pickled code runs.
"""
import argparse, importlib.util, pathlib, sys, types, urllib.request
import numpy as np, onnx, torch, torch.nn as nn, torch.nn.functional as F, yaml

TOOLS = pathlib.Path(__file__).resolve().parent
OUT = TOOLS.parent / 'models' / 'detectors'
CACHE = TOOLS / '.download' / 'deepfakebench'
SRC_URL = 'https://raw.githubusercontent.com/SCLBD/DeepfakeBench/main/training/'
WEIGHTS_URL = 'https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/'
SOURCES = ['networks/xception.py', 'networks/mesonet.py', 'networks/efficientnetb4.py', 'detectors/f3net_detector.py'] + \
          [f'config/detector/{n}.yaml' for n in ['f3net', 'xception', 'meso4', 'meso4Inception', 'efficientnetb4']]
MODELS = {  # name: (checkpoint, store weights at half size)
    'meso4': ('meso4_best.pth', False), 'meso4Inception': ('meso4Incep_best.pth', False),
    'efficientnetb4': ('effnb4_best.pth', True), 'xception': ('xception_best.pth', True), 'f3net': ('f3net_best.pth', True),
}
sys.path.insert(0, str(TOOLS))
from build_models import half_weights  # noqa: E402


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print('downloading', url)
        urllib.request.urlretrieve(url, path)
    return path


def load_code():
    for f in SOURCES:
        fetch(SRC_URL + f, CACHE / 'src' / f)
    # DeepfakeBench registers classes in a registry; a stand-in is all the networks need.
    registry = types.ModuleType('metrics.registry')
    class Registry(dict):
        def register_module(self, module_name=None):
            def deco(cls):
                self[module_name or cls.__name__] = cls
                return cls
            return deco
    registry.BACKBONE = Registry()
    sys.modules['metrics'] = types.ModuleType('metrics'); sys.modules['metrics.registry'] = registry

    def module(name, rel):
        spec = importlib.util.spec_from_file_location(name, CACHE / 'src' / rel)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
    nets = {'xception': module('dfb_xception', 'networks/xception.py'),
            'meso': module('dfb_mesonet', 'networks/mesonet.py'),
            'effb4': module('dfb_effb4', 'networks/efficientnetb4.py')}
    # F3Net's frequency head, taken verbatim from DeepfakeBench's f3net_detector.py.
    src = (CACHE / 'src' / 'detectors/f3net_detector.py').read_text()
    f3 = types.ModuleType('dfb_f3'); f3.__dict__.update(torch=torch, nn=nn, np=np, F=F)
    exec(src[src.index('# Filter Module'):], f3.__dict__)
    nets['f3'] = f3
    return nets


class Detector(nn.Module):
    """Face crop in [-1, 1] -> P(fake)."""
    def __init__(self, backbone, fad_head=None):
        super().__init__()
        self.backbone = backbone
        if fad_head is not None:
            self.FAD_head = fad_head

    def forward(self, image):
        x = self.FAD_head(image) if hasattr(self, 'FAD_head') else image
        return torch.softmax(self.backbone.classifier(self.backbone.features(x)), dim=1)[:, 1:2]


def build(name, nets):
    cfg = lambda n: yaml.safe_load((CACHE / 'src' / f'config/detector/{n}.yaml').read_text())['backbone_config']
    if name == 'xception':
        return Detector(nets['xception'].Xception(cfg('xception')))
    if name in ('f3net', 'f3net_custom'):
        bb = nets['xception'].Xception(cfg('f3net'))
        bb.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        return Detector(bb, nets['f3'].FAD_Head(256))
    if name == 'meso4':
        return Detector(nets['meso'].Meso4(cfg('meso4')))
    if name == 'meso4Inception':
        return Detector(nets['meso'].MesoInception4(cfg('meso4Inception')))
    if name == 'efficientnetb4':
        c = dict(cfg('efficientnetb4')); c['pretrained'] = False
        bb = nets['effb4'].EfficientNetB4(c)
        bb.efficientnet.set_swish(memory_efficient=False)
        return Detector(bb)


def convert(name, checkpoint, half, nets):
    model = build(name, nets)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state) if isinstance(state, dict) else state
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith('num_batches_tracked')]
    if missing or unexpected:
        raise SystemExit(f'{name}: checkpoint does not match the model (missing {missing[:3]}, unexpected {unexpected[:3]})')
    model.eval()
    x = torch.rand(2, 3, 256, 256) * 2 - 1
    tmp = OUT / f'{name}.tmp.onnx'
    torch.onnx.export(model, (x,), str(tmp), input_names=['image'], output_names=['prob'],
                      dynamic_axes={'image': {0: 'n'}, 'prob': {0: 'n'}}, opset_version=17, dynamo=False)
    dest = OUT / (f'{name}_w16.onnx' if half else f'{name}.onnx')
    onnx.save(half_weights(onnx.load(str(tmp))) if half else onnx.load(str(tmp)), str(dest))
    tmp.unlink()
    import onnxruntime as ort
    with torch.no_grad():
        ref = model(x).numpy().ravel()
    got = ort.InferenceSession(str(dest), providers=['CPUExecutionProvider']).run(None, {'image': x.numpy()})[0].ravel()
    print(f'{dest.name}: max difference from PyTorch {float(abs(ref - got).max()):.5f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f3net-checkpoint', help="the project's fine-tuned F3Net .pth (DeepfakeBench format)")
    ap.add_argument('--only-custom', action='store_true', help='build only the custom F3Net')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nets = load_code()
    if not args.only_custom:
        for name, (ckpt, half) in MODELS.items():
            convert(name, fetch(WEIGHTS_URL + ckpt, CACHE / 'weights' / ckpt), half, nets)
    if args.f3net_checkpoint:
        convert('f3net_custom', pathlib.Path(args.f3net_checkpoint), True, nets)


if __name__ == '__main__':
    main()
