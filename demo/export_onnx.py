#!/usr/bin/env python3
"""
Export the frame-level project backends to ONNX for in-browser inference.

Covers the three backends that are small enough for a browser and carry the
strongest results: f3net, xception_df40 and efficientnet_b4. Effort CLIP-L14
(~300M params) and the temporal models (i3d, videomae) are deliberately left out.

Each model is exported once in fp32 and then converted down a precision ladder:

    fp32  reference, what the evaluation used
    fp16  half the size; intended for WebGPU (WASM has few fp16 kernels)
    int8  static QDQ quantisation calibrated on real face crops; for WASM/CPU

q4f16 is not produced: 4-bit weight quantisation (MatMulNBits) only applies to
MatMul weights, and these are convolutional networks with a single small Linear
head, so it would save almost nothing.

Every exported variant is checked against the PyTorch model on held-out frames
(int8 calibrates on the even frames, parity is measured on the odd ones) and the agreement is written to manifest.json, so a lossy variant is
visible rather than silent. The script refuses to export a checkpoint that
does not load (>= 99% of tensors), because the server loaders use strict=False
and a partial load would otherwise produce a confidently wrong model.

Usage (from the repo root, using the frame server's venv):

    zoom-frame-server/venv/bin/python -m pip install onnx onnxruntime onnxconverter-common efficientnet_pytorch
    zoom-frame-server/venv/bin/python demo/export_onnx.py --calib-video "../test vids/1.mov" --out onnx_export

Pipeline check without weights (random init, never publish the output):

    zoom-frame-server/venv/bin/python demo/export_onnx.py --random-init --out /tmp/onnx_untrained
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "zoom-frame-server"
sys.path.insert(0, str(SERVER))

from models.efficientnet_dfb import (  # noqa: E402
    EFFNB4_IMG_SIZE, EFFNB4_MEAN, EFFNB4_STD,
    DeepfakeBenchEfficientNetB4, _clean_state_dict as _clean_effnb4,
)
from models.f3net_dfb import (  # noqa: E402
    F3NET_IMG_SIZE, F3NET_MEAN, F3NET_STD,
    F3NetDFB, _clean_state_dict as _clean_f3net,
)
from models.xception_ffpp import Xception as CadeneXception  # noqa: E402

# Values mirror load_model() in zoom-frame-server/frame_server.py.
# fake_index: every DF40 head is [real, fake] and the server reads index 1.
BACKENDS = {
    "f3net": {
        "label": "F3Net (DF40)",
        "checkpoint": "models/train_on_df40/f3net_best.pth",
        "size": F3NET_IMG_SIZE, "mean": F3NET_MEAN, "std": F3NET_STD, "fake_index": 1,
    },
    "xception_df40": {
        "label": "Xception (DF40)",
        "checkpoint": "models/train_on_df40/xception.pth",
        "size": 256, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "fake_index": 1,
    },
    "efficientnet_b4": {
        "label": "EfficientNet-B4 (DF40)",
        "checkpoint": "models/train_on_df40/efficientnet_b4.pth",
        "size": EFFNB4_IMG_SIZE, "mean": EFFNB4_MEAN, "std": EFFNB4_STD, "fake_index": 1,
    },
}

MIN_LOADED_RATIO = 0.99


def _load_xception(path: Path | None):
    model = CadeneXception(num_classes=2, in_chans=3)
    if path is None:
        return model, 1.0
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module.backbone."):
            k = k[len("module.backbone."):]
        elif k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)  # the server loads this one strictly too
    return model, 1.0


def _load_partial(model, cleaned):
    missing, _unexpected = model.load_state_dict(cleaned, strict=False)
    total = max(1, len(model.state_dict()))
    return (total - len(missing)) / total


def _load_f3net(path: Path | None):
    model = F3NetDFB(img_size=F3NET_IMG_SIZE)
    if path is None:
        return model, 1.0
    return model, _load_partial(model, _clean_f3net(torch.load(path, map_location="cpu")))


def _load_effnb4(path: Path | None):
    model = DeepfakeBenchEfficientNetB4(num_classes=2, in_chans=3)
    # The memory-efficient Swish is a custom autograd Function that ONNX cannot trace.
    model.efficientnet.set_swish(memory_efficient=False)
    if path is None:
        return model, 1.0
    return model, _load_partial(model, _clean_effnb4(torch.load(path, map_location="cpu")))


LOADERS = {"f3net": _load_f3net, "xception_df40": _load_xception, "efficientnet_b4": _load_effnb4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_images(video: str | None, image_dir: str | None, count: int) -> list[Path]:
    """Face-crop-sized RGB frames for int8 calibration and parity checks."""
    if image_dir:
        files = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        return files[:count]
    if video:
        tmp = Path(tempfile.mkdtemp(prefix="df40_calib_"))
        # Spread frames across the clip rather than taking the first N.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", video, "-vf", "fps=2", "-frames:v", str(count),
             str(tmp / "f_%04d.png")],
            check=True,
        )
        return sorted(tmp.glob("*.png"))
    return []


def to_input(image_path: Path | None, size: int, mean, std, rng) -> np.ndarray:
    """Match the server: centre-square crop -> Resize(size) -> ToTensor -> Normalize."""
    if image_path is None:
        arr = rng.uniform(0, 1, (size, size, 3)).astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        side = min(w, h)
        img = img.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
        arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def probe_input(size: int) -> np.ndarray:
    """Deterministic input the browser can rebuild exactly: x[i] = 0.8 * sin(0.001 * i)."""
    n = 3 * size * size
    return (0.8 * np.sin(0.001 * np.arange(n, dtype=np.float64))).astype(np.float32).reshape(1, 3, size, size)


def softmax_fake(logits: np.ndarray, fake_index: int) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, fake_index] / e.sum(axis=1)


def parity(onnx_path: Path, inputs: list[np.ndarray], ref_logits: np.ndarray, fake_index: int) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    logits = np.concatenate([sess.run(None, {name: x})[0] for x in inputs]).astype(np.float64)
    size = inputs[0].shape[-1]
    probe = sess.run(None, {name: probe_input(size)})[0][0].astype(float)
    probs, reference = softmax_fake(logits, fake_index), softmax_fake(ref_logits, fake_index)
    diff = np.abs(probs - reference)
    agree = float(np.mean((probs >= 0.5) == (reference >= 0.5)))
    return {"max_abs_prob_diff": round(float(diff.max()), 5), "mean_abs_prob_diff": round(float(diff.mean()), 5),
            "max_abs_logit_diff": round(float(np.abs(logits - ref_logits).max()), 5),
            "decision_agreement": round(agree, 4), "n": len(inputs),
            "probe_logits_onnxruntime_cpu": [round(v, 5) for v in probe]}


def export_one(key: str, cfg: dict, args, images: list[Path], out_root: Path) -> dict:
    ckpt = None if args.random_init else SERVER / cfg["checkpoint"]
    if ckpt is not None and not ckpt.is_file():
        print(f"[skip] {key}: checkpoint not found at {ckpt}")
        return {}
    model, ratio = LOADERS[key](ckpt)
    if ratio < MIN_LOADED_RATIO:
        raise SystemExit(f"[fail] {key}: only {ratio:.1%} of tensors loaded from {ckpt}; refusing to export")

    size, mean, std, fake_index = cfg["size"], cfg["mean"], cfg["std"], cfg["fake_index"]
    rng = np.random.default_rng(0)
    sources = images or [None] * 16
    inputs = [to_input(p, size, mean, std, rng) for p in sources]

    if args.random_init:
        # Untrained BatchNorm statistics collapse every output to ~0.5, which would make
        # the parity check vacuous. Re-estimate them so outputs actually vary per input.
        torch.manual_seed(0)
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
                m.momentum = None  # cumulative average
        model.train()
        with torch.no_grad():
            for x in inputs:
                model(torch.from_numpy(x).repeat(2, 1, 1, 1))
    model.eval()

    # int8 is calibrated on the even frames; every variant is scored on the odd ones.
    held_out = inputs[1::2]
    with torch.no_grad():
        ref_logits = np.concatenate([model(torch.from_numpy(x)).numpy() for x in held_out]).astype(np.float64)
        probe_logits = model(torch.from_numpy(probe_input(size))).numpy()[0].astype(float)
    ref_probs = softmax_fake(ref_logits, fake_index)
    print(f"[ref] {key}: PyTorch P(fake) on {len(held_out)} held-out frames "
          f"min={ref_probs.min():.3f} mean={ref_probs.mean():.3f} max={ref_probs.max():.3f}")

    out = out_root / key
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "model_fp32.onnx"
    torch.onnx.export(
        model, torch.from_numpy(inputs[0]), str(fp32),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

    import onnx
    from onnxconverter_common import float16
    fp16 = out / "model_fp16.onnx"
    # MatMul stays fp32: F3Net's DCT head multiplies 256-long rows, and fp16 accumulation there
    # drifted visibly on WebGPU. The backbones are Conv-only, so the size saving is unaffected.
    m16 = float16.convert_float_to_float16(
        onnx.load(str(fp32)), keep_io_types=True,
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + ["MatMul"],
    )
    onnx.save(m16, str(fp16))

    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter({"input": x} for x in inputs[::2])

        def get_next(self):
            return next(self.it, None)

    prepped = out / "_prep.onnx"
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
    int8 = out / "model_int8.onnx"
    quantize_static(
        str(prepped), str(int8), Reader(),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prepped.unlink(missing_ok=True)

    variants = {}
    for precision, path in (("fp32", fp32), ("fp16", fp16), ("int8", int8)):
        check = parity(path, held_out, ref_logits, fake_index)
        variants[precision] = {
            "file": f"{key}/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "parity_vs_pytorch": check,
        }
        print(f"[ok] {key} {precision}: {path.stat().st_size / 1e6:.1f} MB  {check}")

    return {
        "id": key,
        "label": cfg["label"],
        "input": {"name": "input", "shape": [1, 3, size, size], "mean": mean, "std": std},
        "output": {"name": "logits", "fake_index": fake_index, "activation": "softmax"},
        "checkpoint": None if ckpt is None else {"path": cfg["checkpoint"], "sha256": sha256(ckpt)},
        "random_init": bool(args.random_init),
        "calibration": {"images": len(inputs[::2]), "held_out": len(held_out),
                        "source": args.calib_video or args.calib_dir or "random noise"},
        # The page re-runs the probe after loading and compares with the same file under
        # onnxruntime CPU (variants[*].parity_vs_pytorch.probe_logits_onnxruntime_cpu), which catches
        # execution-provider bugs such as int8 QDQ being wrong on WebGPU in ORT Web 1.30.
        # The probe is out of distribution, so int8 is not expected to match PyTorch on it.
        "probe": {"formula": "0.8*sin(0.001*i)", "logits_pytorch_fp32": [round(v, 5) for v in probe_logits]},
        "variants": variants,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "onnx_export"))
    ap.add_argument("--only", nargs="*", choices=sorted(BACKENDS), help="subset of backends")
    ap.add_argument("--calib-video", help="video to sample calibration frames from (needs ffmpeg)")
    ap.add_argument("--calib-dir", help="directory of face-crop images to calibrate on")
    ap.add_argument("--calib-count", type=int, default=64)
    ap.add_argument("--random-init", action="store_true",
                    help="skip checkpoints and export untrained weights, to test the pipeline only")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    if args.random_init and (REPO / "site") in [out_root, *out_root.parents]:
        raise SystemExit("refusing to write untrained models under site/")
    if out_root.exists() and any(out_root.iterdir()):
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images = calibration_images(args.calib_video, args.calib_dir, args.calib_count)
    if not images and not args.random_init:
        print("[warn] no calibration images; int8 will be calibrated on noise. Pass --calib-video.")

    models = [export_one(k, BACKENDS[k], args, images, out_root) for k in (args.only or BACKENDS)]
    manifest = {"format": 1, "models": [m for m in models if m]}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {len(manifest['models'])} model(s) -> {out_root}/manifest.json")


if __name__ == "__main__":
    main()
