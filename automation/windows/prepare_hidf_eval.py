import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare HiDF supplementary evaluation files.")
    parser.add_argument("--root", required=True, help="Target root, e.g. C:\\deepfake_eval\\hidf_eval")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_metadata(metadata_path: Path):
    if not metadata_path.is_file():
        return {}
    rows = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        id_field = None
        for candidate in ("Image ID", "image_id", "id", "ID"):
            if candidate in fieldnames:
                id_field = candidate
                break
        if id_field is None:
            return {}
        for row in reader:
            key = (row.get(id_field) or "").strip()
            if key:
                rows[key] = row
    return rows


def label_for_path(path: Path) -> str:
    lowered = str(path).lower()
    if "real-vid" in lowered or "\\real\\" in lowered:
        return "real"
    if "fake-vid" in lowered or "\\fake\\" in lowered:
        return "fake"
    raise ValueError(f"Unable to infer label from path: {path}")


def subject_or_id_for_path(path: Path, label: str, metadata_rows: dict) -> str:
    stem = path.stem
    if label == "fake" and "_" in stem:
        base_id, target_id = stem.split("_", 1)
        if target_id in metadata_rows:
            meta = metadata_rows[target_id]
            race = (meta.get("Race") or meta.get("race") or "").strip()
            gender = (meta.get("Gender") or meta.get("gender") or "").strip()
            age = (meta.get("Age") or meta.get("age") or "").strip()
            enriched = ",".join(part for part in (f"base={base_id}", f"target={target_id}", race, gender, age) if part)
            return enriched or stem
        return f"base={base_id},target={target_id}"
    if stem in metadata_rows:
        meta = metadata_rows[stem]
        race = (meta.get("Race") or meta.get("race") or "").strip()
        gender = (meta.get("Gender") or meta.get("gender") or "").strip()
        age = (meta.get("Age") or meta.get("age") or "").strip()
        enriched = ",".join(part for part in (stem, race, gender, age) if part)
        return enriched or stem
    return stem


def probe_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = 0.0
    if fps > 0.0 and frame_count > 0.0:
        duration_s = frame_count / fps
    return duration_s, fps, width, height


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def duration_distribution(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "min_s": round(min(values), 3),
        "p25_s": round(percentile(values, 0.25), 3),
        "median_s": round(percentile(values, 0.50), 3),
        "p75_s": round(percentile(values, 0.75), 3),
        "max_s": round(max(values), 3),
        "mean_s": round(statistics.fmean(values), 3),
    }


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    fraction = idx - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def copy_selected(rows, subset_root: Path):
    selected = []
    for label in ("real", "fake"):
        label_rows = [row for row in rows if row["label"] == label]
        label_rows.sort(key=lambda row: (-float(row["duration_s"]), row["clip_path"]))
        chosen = label_rows[:3]
        out_dir = ensure_dir(subset_root / label)
        for row in chosen:
            src = Path(row["clip_path"])
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            selected.append(
                {
                    "clip_path": str(dst),
                    "label": label,
                    "duration_s": row["duration_s"],
                    "fps": row["fps"],
                    "width": row["width"],
                    "height": row["height"],
                    "subject_or_id": row["subject_or_id"],
                }
            )
    return selected


def build_notes(root: Path, summary: dict, selected_rows):
    lines = [
        "# HiDF Supplementary Evaluation Notes",
        "",
        "## Official Sources",
        "",
        "- GitHub: https://github.com/DSAIL-SKKU/HiDF",
        "- Zenodo dataset: https://zenodo.org/records/16140829",
        "- DOI: https://doi.org/10.1145/3711896.3737399",
        "",
        "## Citation",
        "",
        "- HiDF: A Human-Indistinguishable Deepfake Dataset. KDD 2025. DOI: 10.1145/3711896.3737399",
        "",
        "## Access / Licensing",
        "",
        "- Full dataset access is provided publicly through Zenodo from the official GitHub repository.",
        "- License: CC BY-NC 4.0, per the Zenodo record and GitHub repository.",
        "",
        "## Supplementary Use",
        "",
        "- This dataset is prepared as a supplementary evaluation set only.",
        "- It is not merged into the main leakfree_eval benchmark or its calibration / validation / test splits.",
        "- HiDF is useful here because it provides quickly accessible long-form public video clips for an additional robustness check.",
        "",
        "## Ethics / Metadata Note",
        "",
        "- The repository states that HiDF includes race, gender, and age metadata for synthesized individuals in `HiDF_metadata.csv`.",
        "- The dataset is described as using commercial deepfake tools and manually curated high-quality content.",
        "",
        "## Summary",
        "",
        f"- Real clips: {summary['real_count']}",
        f"- Fake clips: {summary['fake_count']}",
        f"- Total duration (hours): {summary['total_duration_hours']}",
        f"- Duration distribution: {json.dumps(summary['duration_distribution'], ensure_ascii=True)}",
        "",
        "## Selected Supplementary Clips",
        "",
    ]
    for row in selected_rows:
        lines.append(f"- `{Path(row['clip_path']).name}` | {row['label']} | {row['duration_s']}s | {row['subject_or_id']}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    root = Path(args.root)
    raw_root = root / "raw"
    subset_root = root / "supplementary"
    ensure_dir(raw_root)
    ensure_dir(subset_root)

    metadata_path = raw_root / "metadata.csv"
    metadata_rows = read_metadata(metadata_path)

    video_files = []
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and "supplementary" not in path.parts and "downloads" not in path.parts:
            video_files.append(path)
    if not video_files:
        raise RuntimeError(f"No video files found under {raw_root}")

    manifest_rows = []
    skipped = []
    for path in sorted(video_files):
        try:
            label = label_for_path(path)
            duration_s, fps, width, height = probe_video(path)
            manifest_rows.append(
                {
                    "clip_path": str(path),
                    "label": label,
                    "duration_s": round(duration_s, 3),
                    "fps": round(fps, 3),
                    "width": width,
                    "height": height,
                    "subject_or_id": subject_or_id_for_path(path, label, metadata_rows),
                }
            )
        except Exception as exc:
            skipped.append({"clip_path": str(path), "error": str(exc)})

    if not manifest_rows:
        raise RuntimeError("No readable video files were probed successfully.")

    manifest_path = root / "manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        ["clip_path", "label", "duration_s", "fps", "width", "height", "subject_or_id"],
    )

    selected_rows = copy_selected(manifest_rows, subset_root)
    labels_path = subset_root / "labels.csv"
    write_csv(
        labels_path,
        selected_rows,
        ["clip_path", "label", "duration_s", "fps", "width", "height", "subject_or_id"],
    )

    durations = [float(row["duration_s"]) for row in manifest_rows]
    longest = sorted(manifest_rows, key=lambda row: (-float(row["duration_s"]), row["clip_path"]))[:10]
    summary = {
        "real_count": sum(1 for row in manifest_rows if row["label"] == "real"),
        "fake_count": sum(1 for row in manifest_rows if row["label"] == "fake"),
        "total_duration_s": round(sum(durations), 3),
        "total_duration_hours": round(sum(durations) / 3600.0, 3),
        "duration_distribution": duration_distribution(durations),
        "longest_clips": longest,
        "skipped_count": len(skipped),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if skipped:
        (root / "skipped_videos.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    notes = build_notes(root, summary, selected_rows)
    (root / "dataset_notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
