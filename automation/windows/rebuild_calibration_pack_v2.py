import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30.0
SESSION_DURATION_S = 600.0
SEED = 20260414

REAL_DATASETS = ["ffpp_original", "celebdf_real", "celebdf_youtube_real"]
FAKE_DATASETS = [
    "ffpp_deepfakedetection",
    "ffpp_deepfakes",
    "ffpp_face2face",
    "ffpp_faceshifter",
    "ffpp_faceswap",
    "ffpp_neuraltextures",
    "celebdf_synthesis",
]

SESSION_SPECS = [
    ("calibration", "mixed", 1),
    ("calibration", "mixed", 2),
    ("calibration", "mixed", 3),
    ("calibration", "mixed", 4),
    ("calibration", "real_only", 1),
    ("calibration", "real_only", 2),
    ("calibration", "real_only", 3),
    ("calibration", "fake_only", 1),
    ("calibration", "fake_only", 2),
]

FFPP_DIR_MAP = {
    "ffpp_original": Path("FaceForensics++_C23") / "original",
    "ffpp_deepfakedetection": Path("FaceForensics++_C23") / "DeepFakeDetection",
    "ffpp_deepfakes": Path("FaceForensics++_C23") / "Deepfakes",
    "ffpp_face2face": Path("FaceForensics++_C23") / "Face2Face",
    "ffpp_faceshifter": Path("FaceForensics++_C23") / "FaceShifter",
    "ffpp_faceswap": Path("FaceForensics++_C23") / "FaceSwap",
    "ffpp_neuraltextures": Path("FaceForensics++_C23") / "NeuralTextures",
}

CELEB_DIR_MAP = {
    "celebdf_real": Path("Celeb-real"),
    "celebdf_synthesis": Path("Celeb-synthesis"),
    "celebdf_youtube_real": Path("YouTube-real"),
}


@dataclass
class ClipRow:
    clip_path: str
    dataset: str
    label: str
    basename: str
    identity_tokens: str
    source_tokens: str
    duration_s: float
    width: int
    height: int
    fps: float
    assigned_split: str
    drop_reason: str
    local_path: Path


@dataclass
class SegmentSpec:
    clip: ClipRow
    label: str
    source_start_s: float
    source_end_s: float
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leak-free calibration v2 pack from calibration-assigned source clips.")
    parser.add_argument("--source-root", default=r"C:\deepfake_eval\leakfree_eval")
    parser.add_argument("--raw-root", default=r"C:\deepfake_eval\finetune_f3net\raw")
    parser.add_argument("--output-root", default=r"C:\deepfake_eval\leakfree_eval_calibration_v2_2026-04-14")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_local_clip_path(row: dict, raw_root: Path) -> Path:
    dataset = row["dataset"].strip()
    basename = row["basename"].strip()
    if dataset.startswith("ffpp_"):
        rel = FFPP_DIR_MAP[dataset] / f"{basename}.mp4"
        local = raw_root / "ffpp_c23" / rel
    elif dataset.startswith("celebdf_"):
        rel = CELEB_DIR_MAP[dataset] / f"{basename}.mp4"
        local = raw_root / "celeb_df" / rel
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not local.exists():
        raise FileNotFoundError(f"Missing local source clip for {row['clip_path']}: {local}")
    return local


def load_manifest(manifest_path: Path, raw_root: Path) -> List[ClipRow]:
    clips: List[ClipRow] = []
    for row in read_csv_rows(manifest_path):
        clips.append(
            ClipRow(
                clip_path=row["clip_path"].strip(),
                dataset=row["dataset"].strip(),
                label=row["label"].strip(),
                basename=row["basename"].strip(),
                identity_tokens=row["identity_tokens"].strip(),
                source_tokens=row["source_tokens"].strip(),
                duration_s=float(row["duration_s"]),
                width=int(float(row["width"])),
                height=int(float(row["height"])),
                fps=float(row["fps"]),
                assigned_split=row["assigned_split"].strip(),
                drop_reason=row.get("drop_reason", "").strip(),
                local_path=resolve_local_clip_path(row, raw_root),
            )
        )
    return clips


def shuffle_pools(clips: Iterable[ClipRow], seed: int) -> Dict[str, List[ClipRow]]:
    pools: Dict[str, List[ClipRow]] = defaultdict(list)
    for clip in clips:
        pools[clip.dataset].append(clip)
    for dataset, items in pools.items():
        rng = random.Random(seed + sum(ord(ch) for ch in dataset))
        rng.shuffle(items)
    return pools


def probe_video(path: Path) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}, size={width}x{height}")
    return fps, frame_count, width, height


def fit_frame(frame):
    src_h, src_w = frame.shape[:2]
    scale = min(TARGET_WIDTH / src_w, TARGET_HEIGHT / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized,
        top=(TARGET_HEIGHT - new_h) // 2,
        bottom=TARGET_HEIGHT - new_h - ((TARGET_HEIGHT - new_h) // 2),
        left=(TARGET_WIDTH - new_w) // 2,
        right=TARGET_WIDTH - new_w - ((TARGET_WIDTH - new_w) // 2),
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return canvas


def choose_segment_duration(kind: str, rng: random.Random, clip_duration_s: float, remaining_budget_s: float) -> float:
    if kind == "mixed":
        target = rng.uniform(10.0, 20.0)
    else:
        target = rng.uniform(12.0, 28.0)
    return max(5.0, min(clip_duration_s, remaining_budget_s, target))


def choose_segment_start(rng: random.Random, clip_duration_s: float, segment_duration_s: float) -> float:
    slack = max(0.0, clip_duration_s - segment_duration_s)
    if slack <= 0.5:
        return 0.0
    return round(rng.uniform(0.0, slack), 3)


def next_available_clip(
    pools: Dict[str, List[ClipRow]],
    dataset_cycle: List[str],
    start_idx: int,
    used_clips: set[str],
) -> Tuple[ClipRow, int]:
    for offset in range(len(dataset_cycle)):
        dataset = dataset_cycle[(start_idx + offset) % len(dataset_cycle)]
        for clip in pools[dataset]:
            if clip.clip_path not in used_clips:
                return clip, (start_idx + offset + 1) % len(dataset_cycle)
    raise RuntimeError(f"No unused clips left for datasets: {dataset_cycle}")


def build_session_segments(
    kind: str,
    ordinal: int,
    pools: Dict[str, List[ClipRow]],
    used_clips: set[str],
    rng: random.Random,
) -> List[SegmentSpec]:
    segments: List[SegmentSpec] = []
    if kind == "mixed":
        budgets = {"real": 300.0, "fake": 300.0}
        dataset_cycles = {"real": REAL_DATASETS[:], "fake": FAKE_DATASETS[:]}
        indices = {"real": ordinal % len(REAL_DATASETS), "fake": ordinal % len(FAKE_DATASETS)}
        current_label = "real" if ordinal % 2 == 1 else "fake"
        while budgets["real"] > 0.01 or budgets["fake"] > 0.01:
            label = current_label if budgets[current_label] > 0.01 else ("fake" if current_label == "real" else "real")
            clip, next_idx = next_available_clip(pools, dataset_cycles[label], indices[label], used_clips)
            indices[label] = next_idx
            seg_duration = choose_segment_duration(kind, rng, clip.duration_s, budgets[label])
            source_start = choose_segment_start(rng, clip.duration_s, seg_duration)
            source_end = round(source_start + seg_duration, 3)
            segments.append(SegmentSpec(clip=clip, label=label, source_start_s=source_start, source_end_s=source_end, duration_s=seg_duration))
            budgets[label] -= seg_duration
            used_clips.add(clip.clip_path)
            current_label = "fake" if label == "real" else "real"
    else:
        label = "real" if kind == "real_only" else "fake"
        remaining = SESSION_DURATION_S
        dataset_cycle = REAL_DATASETS[:] if label == "real" else FAKE_DATASETS[:]
        dataset_idx = ordinal % len(dataset_cycle)
        while remaining > 0.01:
            clip, dataset_idx = next_available_clip(pools, dataset_cycle, dataset_idx, used_clips)
            seg_duration = choose_segment_duration(kind, rng, clip.duration_s, remaining)
            source_start = choose_segment_start(rng, clip.duration_s, seg_duration)
            source_end = round(source_start + seg_duration, 3)
            segments.append(SegmentSpec(clip=clip, label=label, source_start_s=source_start, source_end_s=source_end, duration_s=seg_duration))
            remaining -= seg_duration
            used_clips.add(clip.clip_path)
    return segments


def write_segment_to_video(writer: cv2.VideoWriter, segment: SegmentSpec, probe_cache: Dict[str, dict]) -> int:
    cache_key = str(segment.clip.local_path)
    if cache_key not in probe_cache:
        fps, frame_count, width, height = probe_video(segment.clip.local_path)
        probe_cache[cache_key] = {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
    meta = probe_cache[cache_key]
    src_fps = float(meta["fps"])
    start_frame = max(0, int(math.floor(segment.source_start_s * src_fps)))
    end_frame = min(int(meta["frame_count"]), int(math.ceil(segment.source_end_s * src_fps)))
    cap = cv2.VideoCapture(str(segment.clip.local_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for stitching: {segment.clip.local_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    next_output_time = 0.0
    frame_idx = start_frame
    segment_duration = max(0.0, segment.source_end_s - segment.source_start_s)
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        current_time = (frame_idx - start_frame) / src_fps
        next_source_time = min(segment_duration, (frame_idx + 1 - start_frame) / src_fps)
        fitted = fit_frame(frame)
        while next_output_time < next_source_time - 1e-9 and next_output_time < segment_duration - 1e-9:
            writer.write(fitted)
            written += 1
            next_output_time += 1.0 / TARGET_FPS
        frame_idx += 1
    if written == 0:
        raise RuntimeError(f"No frames written for segment from {segment.clip.local_path}")
    cap.release()
    return written


def session_dataset_seconds(segments: List[SegmentSpec]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for segment in segments:
        totals[segment.clip.dataset] += segment.duration_s
    return {key: round(value, 3) for key, value in sorted(totals.items())}


def build_session(
    session_name: str,
    kind: str,
    segments: List[SegmentSpec],
    session_dir: Path,
    recipe_rows: List[dict],
    probe_cache: Dict[str, dict],
) -> dict:
    video_path = session_dir / f"{session_name}.mp4"
    labels_path = session_dir / f"{session_name}_labels.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {video_path}")

    label_rows: List[dict] = []
    current_start = 0.0
    total_frames = 0
    for idx, segment in enumerate(segments, start=1):
        written_frames = write_segment_to_video(writer, segment, probe_cache)
        total_frames += written_frames
        actual_duration = written_frames / TARGET_FPS
        start_s = current_start
        end_s = start_s + actual_duration
        current_start = end_s
        label_rows.append(
            {
                "start_s": f"{start_s:.3f}",
                "end_s": f"{end_s:.3f}",
                "label": segment.label,
                "source": segment.clip.source_tokens,
                "filename": f"{segment.clip.basename}.mp4",
                "dataset": segment.clip.dataset,
                "clip_path": segment.clip.clip_path,
                "identity_tokens": segment.clip.identity_tokens,
                "source_tokens": segment.clip.source_tokens,
                "source_start_s": f"{segment.source_start_s:.3f}",
                "source_end_s": f"{segment.source_end_s:.3f}",
                "local_clip_path": str(segment.clip.local_path),
            }
        )
        recipe_rows.append(
            {
                "session_name": session_name,
                "kind": kind,
                "segment_index": idx,
                "label": segment.label,
                "dataset": segment.clip.dataset,
                "clip_path": segment.clip.clip_path,
                "local_clip_path": str(segment.clip.local_path),
                "identity_tokens": segment.clip.identity_tokens,
                "source_tokens": segment.clip.source_tokens,
                "source_start_s": round(segment.source_start_s, 3),
                "source_end_s": round(segment.source_end_s, 3),
                "segment_duration_s": round(actual_duration, 3),
            }
        )
    writer.release()

    write_csv(
        labels_path,
        label_rows,
        [
            "start_s",
            "end_s",
            "label",
            "source",
            "filename",
            "dataset",
            "clip_path",
            "identity_tokens",
            "source_tokens",
            "source_start_s",
            "source_end_s",
            "local_clip_path",
        ],
    )

    dataset_seconds = session_dataset_seconds(segments)
    duration_s = total_frames / TARGET_FPS
    real_s = sum(float(row["end_s"]) - float(row["start_s"]) for row in label_rows if row["label"] == "real")
    fake_s = sum(float(row["end_s"]) - float(row["start_s"]) for row in label_rows if row["label"] == "fake")
    return {
        "video_path": video_path,
        "labels_path": labels_path,
        "duration_s": round(duration_s, 3),
        "real_s": round(real_s, 3),
        "fake_s": round(fake_s, 3),
        "segments": len(label_rows),
        "unique_sources": len({row["source_tokens"] for row in label_rows}),
        "dataset_seconds": dataset_seconds,
    }


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_existing_split(source_root: Path, output_root: Path, split: str) -> None:
    src_dir = source_root / "sessions" / split
    dst_dir = output_root / "sessions" / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.is_file():
            hardlink_or_copy(path, dst_dir / path.name)


def build_split_metadata(
    all_clips: List[ClipRow],
    selected_calibration_segments: List[dict],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    calibration_counter = Counter()
    calibration_duration = Counter()
    for row in selected_calibration_segments:
        label = row["label"]
        calibration_counter[label] += 1
        calibration_duration[label] += float(row["segment_duration_s"])
    summary["calibration"] = {
        "real": {"segment_count": calibration_counter["real"], "duration_s": round(calibration_duration["real"], 3)},
        "fake": {"segment_count": calibration_counter["fake"], "duration_s": round(calibration_duration["fake"], 3)},
    }

    for split in ("validation", "test"):
        counter = Counter()
        duration = Counter()
        for clip in all_clips:
            if clip.assigned_split != split:
                continue
            counter[clip.label] += 1
            duration[clip.label] += clip.duration_s
        summary[split] = {
            "real": {"clip_count": counter["real"], "duration_s": round(duration["real"], 3)},
            "fake": {"clip_count": counter["fake"], "duration_s": round(duration["fake"], 3)},
        }
    return summary


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()

    ensure_clean_dir(output_root)
    (output_root / "sessions" / "calibration").mkdir(parents=True, exist_ok=True)

    manifest_path = source_root / "manifest_with_splits.csv"
    clips = load_manifest(manifest_path, raw_root)
    calibration_clips = [clip for clip in clips if clip.assigned_split == "calibration" and not clip.drop_reason]
    validation_clips = [clip for clip in clips if clip.assigned_split == "validation" and not clip.drop_reason]
    test_clips = [clip for clip in clips if clip.assigned_split == "test" and not clip.drop_reason]

    pools = shuffle_pools(calibration_clips, SEED)
    rng = random.Random(SEED)
    used_clips: set[str] = set()
    probe_cache: Dict[str, dict] = {}
    recipe_rows: List[dict] = []
    sessions_index_rows: List[dict] = []

    for split, kind, ordinal in SESSION_SPECS:
        session_name = f"{split}_{kind}_{ordinal:02d}"
        segments = build_session_segments(kind, ordinal, pools, used_clips, rng)
        session_meta = build_session(
            session_name=session_name,
            kind=kind,
            segments=segments,
            session_dir=output_root / "sessions" / split,
            recipe_rows=recipe_rows,
            probe_cache=probe_cache,
        )
        sessions_index_rows.append(
            {
                "split": split,
                "kind": kind,
                "ordinal": ordinal,
                "video_path": str((output_root / "sessions" / split / f"{session_name}.mp4").resolve()),
                "labels_path": str((output_root / "sessions" / split / f"{session_name}_labels.csv").resolve()),
                "duration_s": session_meta["duration_s"],
                "real_s": session_meta["real_s"],
                "fake_s": session_meta["fake_s"],
                "segments": session_meta["segments"],
                "unique_sources": session_meta["unique_sources"],
                "dataset_seconds": json.dumps(session_meta["dataset_seconds"], sort_keys=True),
            }
        )

    for split in ("validation", "test"):
        link_existing_split(source_root, output_root, split)
    for row in read_csv_rows(source_root / "sessions_index.csv"):
        split = row["split"].strip()
        if split == "calibration":
            continue
        sessions_index_rows.append(row)

    sessions_index_rows = sorted(sessions_index_rows, key=lambda row: (row["split"], row["kind"], int(row["ordinal"])))

    write_csv(
        output_root / "sessions_index.csv",
        sessions_index_rows,
        ["split", "kind", "ordinal", "video_path", "labels_path", "duration_s", "real_s", "fake_s", "segments", "unique_sources", "dataset_seconds"],
    )
    write_csv(
        output_root / "calibration_session_segments.csv",
        recipe_rows,
        [
            "session_name",
            "kind",
            "segment_index",
            "label",
            "dataset",
            "clip_path",
            "local_clip_path",
            "identity_tokens",
            "source_tokens",
            "source_start_s",
            "source_end_s",
            "segment_duration_s",
        ],
    )

    selected_identity_tokens = {row["identity_tokens"] for row in recipe_rows}
    selected_source_tokens = {row["source_tokens"] for row in recipe_rows}
    val_identity_tokens = {clip.identity_tokens for clip in validation_clips}
    test_identity_tokens = {clip.identity_tokens for clip in test_clips}
    val_source_tokens = {clip.source_tokens for clip in validation_clips}
    test_source_tokens = {clip.source_tokens for clip in test_clips}

    leakage_report = {
        "selected_calibration_segments": len(recipe_rows),
        "selected_unique_clips": len({row["clip_path"] for row in recipe_rows}),
        "selected_unique_identity_tokens": len(selected_identity_tokens),
        "selected_unique_source_tokens": len(selected_source_tokens),
        "validation_identity_overlap": sorted(selected_identity_tokens & val_identity_tokens),
        "test_identity_overlap": sorted(selected_identity_tokens & test_identity_tokens),
        "validation_source_overlap": sorted(selected_source_tokens & val_source_tokens),
        "test_source_overlap": sorted(selected_source_tokens & test_source_tokens),
        "session_specs": [{"split": split, "kind": kind, "ordinal": ordinal} for split, kind, ordinal in SESSION_SPECS],
        "target_video": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT, "fps": TARGET_FPS},
        "seed": SEED,
    }
    (output_root / "leakage_report.json").write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")
    (output_root / "probe_cache.json").write_text(json.dumps(probe_cache, indent=2), encoding="utf-8")

    for filename in ["manifest.csv", "manifest_with_splits.csv", "split_assignments.csv", "split_stats.json"]:
        shutil.copy2(source_root / filename, output_root / filename)

    split_summary = build_split_metadata(clips, recipe_rows)
    lines = [
        "# Leak-Free Calibration V2 Build",
        "",
        "## Calibration Sessions",
    ]
    for row in sessions_index_rows:
        if row["split"] == "calibration":
            lines.append(
                f"- {row['split']} {row['kind']} #{row['ordinal']}: {float(row['duration_s']):.3f}s, real={float(row['real_s']):.3f}s, fake={float(row['fake_s']):.3f}s, segments={row['segments']}"
            )
    lines.extend(
        [
            "",
            "## Leakage Checks",
            f"- calibration identity overlap vs validation: {len(leakage_report['validation_identity_overlap'])}",
            f"- calibration identity overlap vs test: {len(leakage_report['test_identity_overlap'])}",
            f"- calibration source overlap vs validation: {len(leakage_report['validation_source_overlap'])}",
            f"- calibration source overlap vs test: {len(leakage_report['test_source_overlap'])}",
            "",
            "## Summary",
            f"- calibration real segment seconds: {split_summary['calibration']['real']['duration_s']:.3f}",
            f"- calibration fake segment seconds: {split_summary['calibration']['fake']['duration_s']:.3f}",
            f"- validation clips reused unchanged: real={split_summary['validation']['real']['clip_count']}, fake={split_summary['validation']['fake']['clip_count']}",
            f"- test clips reused unchanged: real={split_summary['test']['real']['clip_count']}, fake={split_summary['test']['fake']['clip_count']}",
            "",
            "## Notes",
            "- Validation and test sessions were hard-linked from the existing leak-free pack.",
            "- New calibration sessions were drawn only from clips assigned to the calibration split in `manifest_with_splits.csv`.",
            "- No source clip was reused across the new calibration sessions.",
            "- Sessions were normalized to 1280x720 at 30 FPS and written as video-only MP4 files.",
            "",
        ]
    )
    (output_root / "build_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
