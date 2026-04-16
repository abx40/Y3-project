import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_SERVER_DIR = REPO_ROOT / "zoom-frame-server"
MODELS_DIR = REPO_ROOT / "zoom-frame-server" / "models"
if str(FRAME_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(FRAME_SERVER_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from i3d_dfb import I3D_DEFAULT_CLIP_SIZE, I3D_IMG_SIZE, I3D_MEAN, I3D_STD, load_dfb_i3d_model  # noqa: E402
from videomae_dfb import VIDEOMAE_DEFAULT_CLIP_SIZE, VIDEOMAE_DEFAULT_MODEL_ID  # noqa: E402

try:
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
except Exception:
    AutoImageProcessor = None
    AutoModelForVideoClassification = None


FACE_MARGIN = 0.25
FACE_MIN_BBOX_AREA_RATIO = 0.03
FACE_MIN_CROP_SIDE_RATIO = 0.70


@dataclass
class ClipRow:
    dataset: str
    label: str
    split: str
    clip_path: str
    local_path: str
    basename: str
    identity_tokens: str
    source_tokens: str
    duration_s: float
    fps: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune temporal deepfake backends on the clean train split.")
    parser.add_argument("--backend", required=True, choices=["i3d", "videomae"])
    parser.add_argument(
        "--manifest",
        default=r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\finetune_manifest.csv",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-id", default=VIDEOMAE_DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples-per-clip", type=int, default=1)
    parser.add_argument("--clip-size", type=int, default=0)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-face-crop", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> List[ClipRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                ClipRow(
                    dataset=row["dataset"].strip(),
                    label=row["label"].strip(),
                    split=row["split"].strip(),
                    clip_path=row["clip_path"].strip(),
                    local_path=row["local_path"].strip(),
                    basename=row["basename"].strip(),
                    identity_tokens=row["identity_tokens"].strip(),
                    source_tokens=row["source_tokens"].strip(),
                    duration_s=float(row["duration_s"]),
                    fps=float(row["fps"]),
                    width=int(float(row["width"])),
                    height=int(float(row["height"])),
                )
            )
        return rows


_FACE_CASCADE = None


def get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load cascade at {cascade_path}")
        _FACE_CASCADE = cascade
    return _FACE_CASCADE


def crop_frame(frame: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1]


def select_crop_box(gray_frame: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = gray_frame.shape[:2]
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        bbox_area_ratio = (fw * fh) / max(1, w * h)
        if bbox_area_ratio >= FACE_MIN_BBOX_AREA_RATIO:
            cx = int(x + fw // 2)
            cy = int(y + fh // 2)
            side_from_bbox = int(max(fw, fh) * (1.0 + 2.0 * FACE_MARGIN))
            min_side = int(min(w, h) * FACE_MIN_CROP_SIDE_RATIO)
            side = max(side_from_bbox, min_side)
            side = min(side, min(w, h))
            x0 = max(0, cx - side // 2)
            y0 = max(0, cy - side // 2)
            x1 = x0 + side
            y1 = y0 + side
            if x1 > w:
                x1 = w
                x0 = max(0, x1 - side)
            if y1 > h:
                y1 = h
                y0 = max(0, y1 - side)
            return (x0, y0, x1, y1)
    side = min(h, w)
    start_y = (h - side) // 2
    start_x = (w - side) // 2
    return (start_x, start_y, start_x + side, start_y + side)


def videomae_stats(model_id: str) -> Tuple[int, List[float], List[float]]:
    size = 224
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if AutoImageProcessor is None:
        return size, mean, std
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
        proc_size = getattr(proc, "size", None)
        if isinstance(proc_size, dict):
            if "height" in proc_size and "width" in proc_size:
                size = int(min(proc_size["height"], proc_size["width"]))
            elif "shortest_edge" in proc_size:
                size = int(proc_size["shortest_edge"])
        elif isinstance(proc_size, int):
            size = int(proc_size)
        proc_mean = getattr(proc, "image_mean", None)
        proc_std = getattr(proc, "image_std", None)
        if isinstance(proc_mean, (list, tuple)) and len(proc_mean) >= 3:
            mean = [float(proc_mean[0]), float(proc_mean[1]), float(proc_mean[2])]
        if isinstance(proc_std, (list, tuple)) and len(proc_std) >= 3:
            std = [float(proc_std[0]), float(proc_std[1]), float(proc_std[2])]
    except Exception:
        pass
    return size, mean, std


def backend_spec(args: argparse.Namespace) -> Dict[str, object]:
    if args.backend == "i3d":
        clip_size = int(args.clip_size) if int(args.clip_size) > 0 else I3D_DEFAULT_CLIP_SIZE
        return {
            "clip_size": clip_size,
            "img_size": I3D_IMG_SIZE,
            "mean": I3D_MEAN,
            "std": I3D_STD,
            "loss": "bce_prob",
        }
    if args.backend == "videomae":
        clip_size = int(args.clip_size) if int(args.clip_size) > 0 else VIDEOMAE_DEFAULT_CLIP_SIZE
        img_size, mean, std = videomae_stats(str(args.model_id))
        return {
            "clip_size": clip_size,
            "img_size": img_size,
            "mean": mean,
            "std": std,
            "loss": "ce",
        }
    raise ValueError(args.backend)


def sample_frame_indices(frame_count: int, clip_size: int, fraction: float) -> List[int]:
    start = max(0, int(frame_count * 0.1))
    end = max(start + clip_size, int(frame_count * 0.9))
    max_start = max(start, end - clip_size)
    target_start = min(max_start, max(start, int(round(start + fraction * max(0, max_start - start)))))
    if clip_size <= 1:
        return [target_start]
    return [min(frame_count - 1, int(round(target_start + idx * max(1, (end - target_start - 1) / max(1, clip_size - 1))))) for idx in range(clip_size)]


def extract_clip(path: Path, fraction: float, clip_size: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count for {path}")
    frame_indices = sample_frame_indices(frame_count, clip_size, fraction)
    frames: List[np.ndarray] = []
    for target in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(target) - 1))
            ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read frame {target} from {path}")
        frames.append(frame)
    cap.release()
    return frames


def preprocess_clip(frames_bgr: List[np.ndarray], use_face_crop: bool, spec: Dict[str, object]) -> torch.Tensor:
    processed: List[torch.Tensor] = []
    img_size = int(spec["img_size"])
    mean = torch.tensor(spec["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec["std"], dtype=torch.float32).view(3, 1, 1)
    for frame_bgr in frames_bgr:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if use_face_crop:
            crop_box = select_crop_box(gray)
        else:
            side = min(frame_bgr.shape[:2])
            crop_box = (
                (frame_bgr.shape[1] - side) // 2,
                (frame_bgr.shape[0] - side) // 2,
                (frame_bgr.shape[1] + side) // 2,
                (frame_bgr.shape[0] + side) // 2,
            )
        cropped = crop_frame(frame_bgr, crop_box)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        processed.append((tensor - mean) / std)
    return torch.stack(processed, dim=0)


class TemporalClipDataset(Dataset):
    def __init__(self, rows: List[ClipRow], samples_per_clip: int, use_face_crop: bool, spec: Dict[str, object]):
        self.samples: List[Tuple[ClipRow, float]] = []
        self.use_face_crop = use_face_crop
        self.spec = spec
        fractions = [(idx + 1) / (samples_per_clip + 1) for idx in range(samples_per_clip)]
        for row in rows:
            for fraction in fractions:
                self.samples.append((row, fraction))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row, fraction = self.samples[index]
        clip = extract_clip(Path(row.local_path), fraction, int(self.spec["clip_size"]))
        tensor = preprocess_clip(clip, self.use_face_crop, self.spec)
        target = 1 if row.label == "fake" else 0
        return tensor, target


def build_loader(dataset: TemporalClipDataset, batch_size: int, workers: int, weighted: bool, seed: int) -> DataLoader:
    if weighted:
        labels = [1 if row.label == "fake" else 0 for row, _ in dataset.samples]
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available(), generator=generator)


def load_backend_model(args: argparse.Namespace, device: torch.device):
    if args.backend == "i3d":
        if not args.checkpoint:
            raise RuntimeError("--checkpoint is required for i3d")
        model, report = load_dfb_i3d_model(str(Path(args.checkpoint).resolve()), device, clip_size=int(backend_spec(args)["clip_size"]))
        return model, report.__dict__
    if args.backend == "videomae":
        if AutoModelForVideoClassification is None:
            raise RuntimeError("transformers AutoModelForVideoClassification is required for videomae fine-tune")
        model = AutoModelForVideoClassification.from_pretrained(str(args.model_id)).to(device)
        return model, {"mode": "hf_classifier", "source": str(args.model_id)}
    raise ValueError(args.backend)


def compute_loss(args: argparse.Namespace, outputs, targets: torch.Tensor, spec: Dict[str, object], criterion) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if spec["loss"] == "bce":
        logits_flat = logits.view(-1)
        target_float = targets.float()
        loss = criterion(logits_flat, target_float)
        pred = (torch.sigmoid(logits_flat) >= 0.5).long()
        return loss, pred
    if spec["loss"] == "bce_prob":
        probs_flat = logits.view(-1)
        target_float = targets.float()
        logits_safe = torch.logit(probs_flat.clamp(1e-6, 1.0 - 1e-6))
        loss = criterion(logits_safe, target_float)
        pred = (probs_flat >= 0.5).long()
        return loss, pred
    loss = criterion(logits, targets)
    pred = logits.argmax(dim=1)
    return loss, pred


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, spec: Dict[str, object]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for clips, targets in loader:
            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(clips)
                loss, pred = compute_loss(None, outputs, targets, spec, criterion)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((pred == targets).sum().item())
            total_count += int(targets.size(0))
    return {"loss": total_loss / max(1, total_count), "acc": total_correct / max(1, total_count), "count": total_count}


def save_best_checkpoint(args: argparse.Namespace, model: nn.Module, output_dir: Path, meta: Dict[str, object]) -> str:
    if args.backend == "i3d":
        checkpoint_path = output_dir / "i3d_finetuned_best.pth"
        torch.save({"state_dict": model.state_dict(), "meta": meta}, checkpoint_path)
        return str(checkpoint_path)
    model_dir = output_dir / "videomae_model"
    model.save_pretrained(model_dir)
    if AutoImageProcessor is not None:
        try:
            AutoImageProcessor.from_pretrained(str(args.model_id)).save_pretrained(model_dir)
        except Exception:
            pass
    return str(model_dir)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = backend_spec(args)

    rows = read_manifest(manifest_path)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "train_val"]
    if int(args.max_train_clips) > 0:
        train_rows = train_rows[: int(args.max_train_clips)]
    if int(args.max_val_clips) > 0:
        val_rows = val_rows[: int(args.max_val_clips)]
    if not train_rows or not val_rows:
        raise RuntimeError("Manifest must contain train and train_val rows.")

    use_face_crop = not bool(args.disable_face_crop)
    train_dataset = TemporalClipDataset(train_rows, int(args.samples_per_clip), use_face_crop, spec)
    val_dataset = TemporalClipDataset(val_rows, 1, use_face_crop, spec)
    train_loader = build_loader(train_dataset, int(args.batch_size), int(args.workers), True, int(args.seed))
    val_loader = build_loader(val_dataset, int(args.batch_size), int(args.workers), False, int(args.seed))

    device = torch.device(args.device)
    model, load_report = load_backend_model(args, device)
    if spec["loss"] == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif spec["loss"] == "bce_prob":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: List[dict] = []
    best_val_acc = float("-inf")
    best_checkpoint = ""

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        start_epoch = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_epoch = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start_epoch is not None:
            start_epoch.record()
        for step_idx, (clips, targets) in enumerate(train_loader, start=1):
            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(clips)
                loss, pred = compute_loss(args, outputs, targets, spec, criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((pred == targets).sum().item())
            total_count += int(targets.size(0))
            if int(args.log_interval) > 0 and (step_idx == 1 or step_idx % int(args.log_interval) == 0):
                print(
                    f"[train] backend={args.backend} epoch={epoch} step={step_idx}/{len(train_loader)} "
                    f"loss={total_loss / max(1, total_count):.4f} acc={total_correct / max(1, total_count):.4f}",
                    flush=True,
                )

        train_metrics = {"loss": total_loss / max(1, total_count), "acc": total_correct / max(1, total_count), "count": total_count}
        val_metrics = evaluate(model, val_loader, criterion, device, spec)
        epoch_seconds = None
        if end_epoch is not None and start_epoch is not None:
            end_epoch.record()
            torch.cuda.synchronize()
            epoch_seconds = start_epoch.elapsed_time(end_epoch) / 1000.0
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        summary = (
            f"[epoch] backend={args.backend} epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        if epoch_seconds is not None:
            summary += f" epoch_s={epoch_seconds:.1f}"
        print(summary, flush=True)

        if val_metrics["acc"] >= best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_checkpoint = save_best_checkpoint(
                args,
                model,
                output_dir,
                {
                    "backend": args.backend,
                    "epoch": epoch,
                    "manifest": str(manifest_path),
                    "clip_size": int(spec["clip_size"]),
                    "samples_per_clip": int(args.samples_per_clip),
                    "face_crop": use_face_crop,
                },
            )

    metrics = {
        "backend": args.backend,
        "device": str(device),
        "manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint),
        "model_id": str(args.model_id),
        "train_clip_count": len(train_rows),
        "val_clip_count": len(val_rows),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "face_crop": use_face_crop,
        "samples_per_clip": int(args.samples_per_clip),
        "clip_size": int(spec["clip_size"]),
        "spec": spec,
        "load_report": load_report,
        "history": history,
        "best_checkpoint": best_checkpoint,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
