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
MODELS_DIR = REPO_ROOT / "zoom-frame-server" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from f3net_dfb import F3NET_IMG_SIZE, F3NET_MEAN, F3NET_STD, load_dfb_f3net_model  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Fine-tune F3Net on a clean unseen-data manifest.")
    parser.add_argument(
        "--manifest",
        default=r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15\finetune_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "zoom-frame-server" / "models" / "train_on_df40" / "f3net_best.pth"),
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15\artifacts\finetune_run_01",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-samples-per-clip", type=int, default=4)
    parser.add_argument("--val-samples-per-clip", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260415)
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
        x = int(round(x))
        y = int(round(y))
        fw = max(1, int(round(fw)))
        fh = max(1, int(round(fh)))
        bbox_area = fw * fh
        frame_area = max(1, w * h)
        bbox_area_ratio = bbox_area / frame_area
        if bbox_area_ratio >= FACE_MIN_BBOX_AREA_RATIO:
            cx = x + fw // 2
            cy = y + fh // 2
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


def extract_frame(path: Path, fraction: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count for {path}")
    start = max(0, int(frame_count * 0.1))
    end = max(start + 1, int(frame_count * 0.9))
    target = min(frame_count - 1, max(start, int(round(start + fraction * (end - start)))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - 1))
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read target frame from {path}")
    return frame


def preprocess_frame(frame_bgr: np.ndarray, use_face_crop: bool) -> torch.Tensor:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crop_box = select_crop_box(gray) if use_face_crop else (
        (frame_bgr.shape[1] - min(frame_bgr.shape[:2])) // 2,
        (frame_bgr.shape[0] - min(frame_bgr.shape[:2])) // 2,
        (frame_bgr.shape[1] + min(frame_bgr.shape[:2])) // 2,
        (frame_bgr.shape[0] + min(frame_bgr.shape[:2])) // 2,
    )
    cropped = crop_frame(frame_bgr, crop_box)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (F3NET_IMG_SIZE, F3NET_IMG_SIZE), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(F3NET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(F3NET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


class VideoFrameDataset(Dataset):
    def __init__(self, rows: List[ClipRow], samples_per_clip: int, use_face_crop: bool):
        self.rows = rows
        self.use_face_crop = use_face_crop
        self.samples: List[Tuple[ClipRow, float]] = []
        fractions = [(idx + 1) / (samples_per_clip + 1) for idx in range(samples_per_clip)]
        for row in rows:
            for fraction in fractions:
                self.samples.append((row, fraction))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row, fraction = self.samples[index]
        tensor = preprocess_frame(extract_frame(Path(row.local_path), fraction), self.use_face_crop)
        target = 1 if row.label == "fake" else 0
        return tensor, target


def build_loader(
    dataset: VideoFrameDataset,
    batch_size: int,
    workers: int,
    weighted: bool,
    seed: int,
) -> DataLoader:
    if weighted:
        labels = [1 if row.label == "fake" else 0 for row, _ in dataset.samples]
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_count += int(targets.size(0))
    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
        "count": total_count,
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "train_val"]
    if not train_rows or not val_rows:
        raise RuntimeError("Manifest must contain both train and train_val rows.")

    use_face_crop = not bool(args.disable_face_crop)
    train_dataset = VideoFrameDataset(train_rows, int(args.train_samples_per_clip), use_face_crop=use_face_crop)
    val_dataset = VideoFrameDataset(val_rows, int(args.val_samples_per_clip), use_face_crop=use_face_crop)
    train_loader = build_loader(
        train_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        weighted=True,
        seed=int(args.seed),
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        weighted=False,
        seed=int(args.seed),
    )

    device = torch.device(args.device)
    model, load_report = load_dfb_f3net_model(str(Path(args.checkpoint).resolve()), device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: List[dict] = []
    best_val_acc = float("-inf")
    best_checkpoint_path = output_dir / "f3net_finetuned_best.pth"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_count += int(targets.size(0))

        train_metrics = {
            "loss": total_loss / max(1, total_count),
            "acc": total_correct / max(1, total_count),
            "count": total_count,
        }
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["acc"] >= best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "epoch": epoch,
                        "manifest": str(manifest_path),
                        "train_samples_per_clip": int(args.train_samples_per_clip),
                        "val_samples_per_clip": int(args.val_samples_per_clip),
                        "face_crop": use_face_crop,
                    },
                },
                best_checkpoint_path,
            )

    metrics = {
        "device": str(device),
        "manifest": str(manifest_path),
        "train_clip_count": len(train_rows),
        "val_clip_count": len(val_rows),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "face_crop": use_face_crop,
        "train_samples_per_clip": int(args.train_samples_per_clip),
        "val_samples_per_clip": int(args.val_samples_per_clip),
        "load_report": {
            "loaded_ratio": load_report.loaded_ratio,
            "missing_count": load_report.missing_count,
            "unexpected_count": load_report.unexpected_count,
        },
        "history": history,
        "best_checkpoint": str(best_checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
