#!/usr/bin/env python3
"""Build the AI face-swap model files the demo loads from demo/models/.

Downloads InsightFace's inswapper_128 and ArcFace w600k_r50 (as published in
FaceFusion's model release), then for GitHub Pages:
  - stores their weights as float16 (half the size; the maths stays float32),
  - splits the swap model into parts under GitHub's 100 MB file limit,
  - saves the swap model's identity-mapping matrix as emap.bin,
  - precomputes the identity code of each preset face (faces/latents.json).

Usage:  pip install onnx onnxruntime numpy pillow
        python3 demo/tools/build_models.py
Licence: inswapper_128 is released by InsightFace for non-commercial research only.
"""
import json, pathlib, urllib.request
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from PIL import Image

DEMO = pathlib.Path(__file__).resolve().parent.parent
OUT, CACHE = DEMO / 'models', DEMO / 'tools' / '.download'
RELEASE = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/'
PART_SIZE = 95_000_000
ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])


def download(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print('downloading', name)
        urllib.request.urlretrieve(RELEASE + name, path)
    return path


def half_weights(model):
    """Store large float32 weights as float16, cast back to float32 when loaded."""
    g = model.graph
    inits, casts = [], []
    for init in g.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size > 1024:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + '_fp16')
            inits.append(h)
            casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT))
        else:
            inits.append(init)
    del g.initializer[:]; g.initializer.extend(inits)
    nodes = list(g.node); del g.node[:]; g.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    return model


def five_points(lm, w, h):
    p = np.array(lm).reshape(-1, 3)[:, :2] * [w, h]
    return np.array([(p[33] + p[133]) / 2, (p[362] + p[263]) / 2, p[1], p[61], p[291]])


def similarity(src, dst):
    ms, md = src.mean(0), dst.mean(0); s, d = src - ms, dst - md
    n = (s ** 2).sum()
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum() / n
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / n
    return np.array([[a, -b, md[0] - (a * ms[0] - b * ms[1])], [b, a, md[1] - (b * ms[0] + a * ms[1])]])


def main():
    OUT.mkdir(exist_ok=True)
    swap = onnx.load(download('inswapper_128.onnx'))
    emap = numpy_helper.to_array([i for i in swap.graph.initializer if i.name == 'initializer'][0]).astype(np.float32)
    emap.tofile(OUT / 'emap.bin')
    keep = [i for i in swap.graph.initializer if i.name != 'initializer']
    del swap.graph.initializer[:]; swap.graph.initializer.extend(keep)
    data = half_weights(swap).SerializeToString()
    for old in OUT.glob('inswapper_128_w16.onnx.part*'):
        old.unlink()
    for k in range(0, len(data), PART_SIZE):
        (OUT / f'inswapper_128_w16.onnx.part{k // PART_SIZE}').write_bytes(data[k:k + PART_SIZE])

    arc_path = OUT / 'arcface_w600k_r50_fp16.onnx'
    onnx.save(half_weights(onnx.load(download('arcface_w600k_r50.onnx'))), arc_path)

    arc = ort.InferenceSession(str(arc_path), providers=['CPUExecutionProvider'])
    points = json.loads((DEMO / 'faces' / 'faces.json').read_text())
    latents = {}
    for name, lm in points.items():
        img = Image.open(DEMO / 'faces' / f'{name}.jpg').convert('RGB')
        m = np.vstack([similarity(five_points(lm, *img.size), ARC), [0, 0, 1]])
        crop = img.transform((112, 112), Image.AFFINE, tuple(np.linalg.inv(m)[:2].flatten()), resample=Image.BILINEAR)
        x = ((np.asarray(crop, np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = arc.run(None, {'input': x})[0][0]
        lat = (e / np.linalg.norm(e)) @ emap
        latents[name] = [round(float(v), 6) for v in lat / np.linalg.norm(lat)]
    (DEMO / 'faces' / 'latents.json').write_text(json.dumps(latents))
    print('done:', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
