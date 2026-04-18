import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / "zoom-frame-server" / "venv" / "Scripts" / "python.exe"
OFFLINE_SCORER = REPO_ROOT / "automation" / "windows" / "offline_mp4_sanity.py"
DEFAULT_TRAINING_SUMMARY = REPO_ROOT / "results_finetune_clean_progress_2026-04-17" / "training_summary.csv"
DEFAULT_SESSIONS_INDEX = Path(r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\sessions_index.csv")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results_offline_ft_all_full_2026-04-17"
DEFAULT_EXISTING_F3NET = REPO_ROOT / "results_offline_ft_f3net_full_2026-04-17_v2"

CHECKPOINT_ENV = {
    "xception_df40": "XCEPTION_CKPT",
    "effort_clip_l14": "EFFORT_CKPT",
    "efficientnet_b4": "EFFICIENTNET_CKPT",
    "f3net": "F3NET_CKPT",
    "i3d": "I3D_CKPT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full offline fine-tuned session-pack pass for selected backends.")
    parser.add_argument(
        "--training-summary",
        default=str(DEFAULT_TRAINING_SUMMARY),
        help="CSV with fine-tuned checkpoint paths.",
    )
    parser.add_argument(
        "--sessions-index",
        default=str(DEFAULT_SESSIONS_INDEX),
        help="Clean experiment sessions_index.csv.",
    )
    parser.add_argument(
        "--models",
        default="xception_df40,effort_clip_l14,efficientnet_b4,f3net,i3d,videomae",
        help="Comma-separated backend list.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory that will contain per-model offline result folders plus a consolidated summary.",
    )
    parser.add_argument(
        "--existing-f3net-dir",
        default=str(DEFAULT_EXISTING_F3NET),
        help="Existing F3Net offline bundle to reuse instead of rerunning.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if offline_scores.csv already exists for a model.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_eval_path(path_str: str) -> str:
    if path_str.startswith("/Users/abx/deepfake_eval/"):
        suffix = path_str[len("/Users/abx/deepfake_eval/"):].replace("/", "\\")
        return str(Path(r"C:\deepfake_eval") / suffix)
    return path_str


def manual_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def load_session_meta(sessions_index: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(sessions_index)
    meta: Dict[str, Dict[str, str]] = {}
    for row in rows:
        meta[normalize_eval_path(row["video_path"])] = {
            "split": row["split"],
            "kind": row["kind"],
            "labels_path": normalize_eval_path(row["labels_path"]),
        }
    return meta


def label_lookup_builder(session_meta: Dict[str, Dict[str, str]]):
    cache: Dict[str, List[Dict[str, str]]] = {}

    def label_for(video_path: str, second: int) -> Optional[str]:
        entry = session_meta.get(video_path)
        if entry is None:
            return None
        labels_path = entry["labels_path"]
        if labels_path not in cache:
            cache[labels_path] = read_csv_rows(Path(labels_path))
        t = float(second)
        for seg in cache[labels_path]:
            start = float(seg["start_s"])
            end = float(seg["end_s"])
            if start <= t < end:
                return seg["label"]
        last = cache[labels_path][-1]
        if t <= float(last["end_s"]) + 1e-6:
            return last["label"]
        return None

    return label_for


def build_auc_summary(scores_csv: Path, sessions_index: Path) -> Dict[str, object]:
    session_meta = load_session_meta(sessions_index)
    label_for = label_lookup_builder(session_meta)
    rows = read_csv_rows(scores_csv)

    result: Dict[str, object] = {}
    all_labels: List[int] = []
    all_scores: List[float] = []

    for split in ("calibration", "validation", "test"):
        labels: List[int] = []
        scores: List[float] = []
        by_kind: Dict[str, Dict[str, List[float]]] = {}

        for row in rows:
            video_path = row["video_path"]
            session = session_meta.get(video_path)
            if session is None or session["split"] != split:
                continue
            score_raw = row.get("score_raw", "")
            if score_raw in ("", None):
                continue
            label = label_for(video_path, int(row["second"]))
            if label not in {"real", "fake"}:
                continue
            y = 1 if label == "fake" else 0
            s = float(score_raw)
            labels.append(y)
            scores.append(s)
            all_labels.append(y)
            all_scores.append(s)
            kind = session["kind"]
            by_kind.setdefault(kind, {"labels": [], "scores": []})
            by_kind[kind]["labels"].append(y)
            by_kind[kind]["scores"].append(s)

        result[split] = {
            "rows": len(labels),
            "auc_raw": manual_auc(labels, scores),
            "fake_rate": (sum(labels) / len(labels)) if labels else None,
            "mean_score_raw": (sum(scores) / len(scores)) if scores else None,
            "by_kind": {
                kind: {
                    "rows": len(payload["labels"]),
                    "auc_raw": manual_auc(payload["labels"], payload["scores"]),
                    "fake_rate": (sum(payload["labels"]) / len(payload["labels"])) if payload["labels"] else None,
                    "mean_score_raw": (sum(payload["scores"]) / len(payload["scores"])) if payload["scores"] else None,
                }
                for kind, payload in by_kind.items()
            },
        }

    result["all"] = {
        "rows": len(all_labels),
        "auc_raw": manual_auc(all_labels, all_scores),
        "fake_rate": (sum(all_labels) / len(all_labels)) if all_labels else None,
        "mean_score_raw": (sum(all_scores) / len(all_scores)) if all_scores else None,
    }
    return result


def load_checkpoint_map(training_summary: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in read_csv_rows(training_summary):
        backend = row["backend"].strip()
        checkpoint = row.get("checkpoint_or_model_dir", "").strip()
        if backend and checkpoint:
            mapping[backend] = checkpoint
    return mapping


def build_video_list(sessions_index: Path) -> List[str]:
    rows = read_csv_rows(sessions_index)
    return [normalize_eval_path(row["video_path"]) for row in rows]


def configure_env(base_env: Dict[str, str], backend: str, checkpoint_path: str) -> Dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    if backend == "videomae":
        env["VIDEOMAE_CKPT"] = ""
        env["VIDEOMAE_MODEL_ID"] = checkpoint_path
    else:
        env_name = CHECKPOINT_ENV[backend]
        env[env_name] = checkpoint_path
    return env


def run_model(
    backend: str,
    checkpoint_path: str,
    videos: List[str],
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "offline_run.log"
    env = configure_env(os.environ, backend, checkpoint_path)
    cmd = [
        str(PYTHON),
        str(OFFLINE_SCORER),
        "--models",
        backend,
        "--videos",
        ",".join(videos),
        "--output-dir",
        str(output_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"backend={backend}\n")
        log_handle.write(f"video_count={len(videos)}\n")
        log_handle.flush()
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        log_handle.write(f"\nreturncode {result.returncode}\n")
        log_handle.flush()
    return result.returncode


def write_summary(output_root: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "backend",
        "status",
        "output_dir",
        "rows",
        "valid_score_rows",
        "mean_score_raw",
        "auc_calibration",
        "auc_validation",
        "auc_test",
        "auc_all",
    ]
    csv_path = output_root / "offline_results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# Fine-Tuned Offline Session-Pack Summary",
        "",
        "| backend | status | rows | mean raw score | calib AUROC | val AUROC | test AUROC | overall AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {backend} | {status} | {rows} | {mean_score_raw} | {auc_calibration} | {auc_validation} | {auc_test} | {auc_all} |".format(
                backend=row["backend"],
                status=row["status"],
                rows=row["rows"],
                mean_score_raw=row["mean_score_raw"] or "",
                auc_calibration=row["auc_calibration"] or "",
                auc_validation=row["auc_validation"] or "",
                auc_test=row["auc_test"] or "",
                auc_all=row["auc_all"] or "",
            )
        )
    (output_root / "offline_results_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    training_summary = Path(args.training_summary).resolve()
    sessions_index = Path(args.sessions_index).resolve()
    output_root = Path(args.output_root).resolve()
    existing_f3net_dir = Path(args.existing_f3net_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_map = load_checkpoint_map(training_summary)
    videos = build_video_list(sessions_index)
    missing_videos = [video for video in videos if not Path(video).exists()]
    if missing_videos:
        raise FileNotFoundError(f"Missing session videos: {missing_videos[:5]}")

    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    summary_rows: List[Dict[str, object]] = []

    for backend in requested_models:
        if backend == "f3net" and existing_f3net_dir.exists() and (existing_f3net_dir / "offline_scores.csv").exists() and not args.force:
            output_dir = existing_f3net_dir
            status = "reused_existing"
        else:
            output_dir = output_root / backend
            if (output_dir / "offline_scores.csv").exists() and not args.force:
                status = "reused_existing"
            else:
                checkpoint_path = checkpoint_map.get(backend)
                if not checkpoint_path:
                    raise KeyError(f"No checkpoint path for backend {backend} in {training_summary}")
                returncode = run_model(backend, checkpoint_path, videos, output_dir)
                if returncode != 0:
                    raise RuntimeError(f"Offline run failed for {backend} with return code {returncode}")
                status = "completed"

        summary_path = output_dir / "offline_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing offline_summary.json for {backend} at {output_dir}")
        scores_csv = output_dir / "offline_scores.csv"
        auc_summary = build_auc_summary(scores_csv, sessions_index)
        (output_dir / "offline_auc_summary.json").write_text(json.dumps(auc_summary, indent=2), encoding="utf-8")

        base_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        backend_summary = base_summary[backend]
        summary_rows.append(
            {
                "backend": backend,
                "status": status,
                "output_dir": str(output_dir),
                "rows": backend_summary.get("rows"),
                "valid_score_rows": backend_summary.get("valid_score_rows"),
                "mean_score_raw": format_metric(backend_summary.get("mean_score_raw")),
                "auc_calibration": format_metric(auc_summary["calibration"]["auc_raw"]),
                "auc_validation": format_metric(auc_summary["validation"]["auc_raw"]),
                "auc_test": format_metric(auc_summary["test"]["auc_raw"]),
                "auc_all": format_metric(auc_summary["all"]["auc_raw"]),
            }
        )

    write_summary(output_root, summary_rows)


if __name__ == "__main__":
    main()
