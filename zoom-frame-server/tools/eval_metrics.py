import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def parse_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_int(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_gt_label(raw_value):
    if raw_value is None:
        return None
    val = str(raw_value).strip().lower()
    if val in {"fake", "1", "true"}:
        return 1
    if val in {"real", "0", "false"}:
        return 0
    return None


def load_predictions_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = dict(row)
            item["t_rel"] = parse_float(row.get("t_rel"))
            item["t_recv"] = parse_float(row.get("t_recv"))
            item["t_infer_start"] = parse_float(row.get("t_infer_start"))
            item["t_infer_end"] = parse_float(row.get("t_infer_end"))
            item["t_decision"] = parse_float(row.get("t_decision"))
            item["score_raw"] = parse_float(row.get("score_raw"))
            item["score_smoothed"] = parse_float(row.get("score_smoothed"))
            item["pred_label"] = parse_int(row.get("pred_label"))
            item["gt_label"] = parse_int(row.get("gt_label"))
            item["score_for_metrics"] = (
                item["score_smoothed"]
                if item["score_smoothed"] is not None
                else item["score_raw"]
            )
            rows.append(item)
    return rows


def load_label_segments(labels_csv):
    segments = []
    if not labels_csv or not os.path.isfile(labels_csv):
        return segments

    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"start", "end"}.issubset(set(reader.fieldnames)):
            for row in reader:
                start = parse_float(row.get("start"))
                end = parse_float(row.get("end"))
                gt = parse_gt_label(row.get("real|fake") or row.get("label"))
                if start is None or end is None:
                    continue
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "gt_label": gt,
                        "src": row.get("src"),
                        "filename": row.get("filename"),
                    }
                )
        else:
            f.seek(0)
            plain_reader = csv.reader(f)
            for row in plain_reader:
                if len(row) < 3:
                    continue
                start = parse_float(row[0])
                end = parse_float(row[1])
                gt = parse_gt_label(row[2])
                if start is None or end is None:
                    continue
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "gt_label": gt,
                        "src": row[3] if len(row) > 3 else None,
                        "filename": row[4] if len(row) > 4 else None,
                    }
                )
    segments.sort(key=lambda x: x["start"])
    return segments


def gt_for_time(t_rel, segments):
    for seg in segments:
        if seg["start"] <= t_rel < seg["end"]:
            return seg["gt_label"]
    return None


def safe_div(a, b):
    if b == 0:
        return None
    return a / b


def percentile_stats(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"p50": None, "p95": None, "p99": None, "count": 0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "count": int(arr.size),
    }


def roc_auc_manual(y_true, scores):
    y = np.array(y_true, dtype=int)
    s = np.array(scores, dtype=float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]

    tp = 0
    fp = 0
    points = [(0.0, 0.0)]
    i = 0
    n = len(y_sorted)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / pos
        fpr = fp / neg
        points.append((fpr, tpr))
        i = j

    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))

    auc = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        auc += (x2 - x1) * (y1 + y2) * 0.5
    return float(auc)


def pr_auc_manual(y_true, scores):
    y = np.array(y_true, dtype=int)
    s = np.array(scores, dtype=float)
    pos = int((y == 1).sum())
    if pos == 0:
        return None

    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]

    tp = 0
    fp = 0
    points = [(0.0, 1.0)]  # recall, precision
    i = 0
    n = len(y_sorted)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = safe_div(tp, pos) or 0.0
        precision = safe_div(tp, tp + fp)
        precision = precision if precision is not None else 1.0
        points.append((recall, precision))
        i = j

    auc = 0.0
    for i in range(1, len(points)):
        r1, p1 = points[i - 1]
        r2, p2 = points[i]
        auc += (r2 - r1) * (p1 + p2) * 0.5
    return float(auc)


def derive_fake_segments_from_rows(rows):
    fake_segments = []
    current_start = None
    current_end = None
    for row in rows:
        gt = row.get("gt_label")
        t_rel = row.get("t_rel")
        if gt == 1 and t_rel is not None:
            if current_start is None:
                current_start = t_rel
                current_end = t_rel
            else:
                if t_rel - current_end > 1.5:
                    fake_segments.append({"start": current_start, "end": current_end})
                    current_start = t_rel
                current_end = t_rel
        else:
            if current_start is not None:
                fake_segments.append({"start": current_start, "end": current_end})
                current_start = None
                current_end = None
    if current_start is not None:
        fake_segments.append({"start": current_start, "end": current_end})
    return fake_segments


def compute_time_to_first_detection(rows, fake_segments):
    out = []
    for seg in fake_segments:
        start = seg["start"]
        end = seg["end"]
        det_t = None
        for row in rows:
            t_rel = row.get("t_rel")
            pred = row.get("pred_label")
            if t_rel is None:
                continue
            if start <= t_rel < end and pred == 1:
                det_t = t_rel
                break
        out.append(
            {
                "segment_start": start,
                "segment_end": end,
                "detected": det_t is not None,
                "time_to_first_detection_s": (det_t - start) if det_t is not None else None,
            }
        )
    return out


def maybe_write_plot(rows, fake_segments, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    ts = [r["t_rel"] for r in rows if r.get("t_rel") is not None]
    scores = [r["score_for_metrics"] for r in rows if r.get("t_rel") is not None]
    if len(ts) == 0:
        return False

    plt.figure(figsize=(12, 4))
    plt.plot(ts, scores, linewidth=1.2, label="score")
    for seg in fake_segments:
        plt.axvspan(seg["start"], seg["end"], alpha=0.15, color="red")
    plt.ylim(0.0, 1.0)
    plt.xlabel("t_rel (s)")
    plt.ylabel("fake score")
    plt.title("Score timeline vs GT fake segments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from backend predictions log.")
    parser.add_argument("--run_id", type=str, default=None, help="Run id under log_dir.")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv.")
    parser.add_argument("--labels_csv", type=str, default=None, help="Optional labels CSV.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base log dir.")
    parser.add_argument("--output", type=str, default=None, help="Output metrics.json path.")
    args = parser.parse_args()

    if args.predictions:
        pred_path = Path(args.predictions)
        run_dir = pred_path.parent
        run_id = args.run_id or run_dir.name
    else:
        if not args.run_id:
            raise SystemExit("Provide either --predictions or --run_id.")
        run_id = args.run_id
        run_dir = Path(args.log_dir) / run_id
        pred_path = run_dir / "predictions.csv"

    if not pred_path.exists():
        raise SystemExit(f"Predictions log not found: {pred_path}")

    rows = load_predictions_csv(pred_path)
    if not rows:
        raise SystemExit("Predictions log is empty.")

    labels_csv = args.labels_csv
    run_meta_path = run_dir / "run_meta.json"
    run_meta = {}
    if run_meta_path.exists():
        with open(run_meta_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)
        if not labels_csv:
            labels_csv = run_meta.get("labels_csv")

    segments = load_label_segments(labels_csv)
    if segments:
        for r in rows:
            if r.get("gt_label") is None and r.get("t_rel") is not None:
                r["gt_label"] = gt_for_time(r["t_rel"], segments)

    eval_rows = [r for r in rows if r.get("gt_label") in (0, 1) and r.get("pred_label") in (0, 1)]
    y_true = [r["gt_label"] for r in eval_rows]
    y_pred = [r["pred_label"] for r in eval_rows]
    # Use raw scores for ROC/PR curves (calibration ranking), while confusion
    # metrics stay tied to pred_label at the deployed operating point.
    y_score_raw = [r["score_raw"] for r in eval_rows if r.get("score_raw") is not None]
    y_score_raw_true = [r["gt_label"] for r in eval_rows if r.get("score_raw") is not None]
    y_score_smoothed = [r["score_smoothed"] for r in eval_rows if r.get("score_smoothed") is not None]
    y_score_smoothed_true = [r["gt_label"] for r in eval_rows if r.get("score_smoothed") is not None]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = safe_div(tp + tn, len(y_true))
    recall = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    balanced_accuracy = None
    if recall is not None and tnr is not None:
        balanced_accuracy = 0.5 * (recall + tnr)
    precision = safe_div(tp, tp + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    t_rels = [r["t_rel"] for r in rows if r.get("t_rel") is not None]
    duration_min = None
    if t_rels:
        duration = max(t_rels) - min(t_rels)
        duration_min = duration / 60.0 if duration > 0 else None
    false_alarms_per_minute = None
    if duration_min and duration_min > 0:
        false_alarms_per_minute = fp / duration_min

    roc_auc_raw = None
    pr_auc_raw = None
    if y_score_raw and len(set(y_score_raw_true)) > 1:
        roc_auc_raw = roc_auc_manual(y_score_raw_true, y_score_raw)
        pr_auc_raw = pr_auc_manual(y_score_raw_true, y_score_raw)

    roc_auc_smoothed = None
    pr_auc_smoothed = None
    if y_score_smoothed and len(set(y_score_smoothed_true)) > 1:
        roc_auc_smoothed = roc_auc_manual(y_score_smoothed_true, y_score_smoothed)
        pr_auc_smoothed = pr_auc_manual(y_score_smoothed_true, y_score_smoothed)

    infer_lat_ms = []
    e2e_lat_ms = []
    for r in rows:
        t0 = r.get("t_infer_start")
        t1 = r.get("t_infer_end")
        tr = r.get("t_recv")
        td = r.get("t_decision")
        if t0 is not None and t1 is not None and t1 >= t0:
            infer_lat_ms.append((t1 - t0) * 1000.0)
        if tr is not None and td is not None and td >= tr:
            e2e_lat_ms.append((td - tr) * 1000.0)

    fake_segments = [s for s in segments if s.get("gt_label") == 1]
    if not fake_segments:
        fake_segments = derive_fake_segments_from_rows(rows)
    ttf = compute_time_to_first_detection(rows, fake_segments)
    detected_delays = [x["time_to_first_detection_s"] for x in ttf if x["time_to_first_detection_s"] is not None]

    metrics = {
        "run_id": run_id,
        "predictions_path": str(pred_path),
        "labels_csv": labels_csv,
        "num_rows_total": len(rows),
        "num_rows_eval": len(eval_rows),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "curve_score_source": "score_raw",
        "roc_auc": roc_auc_raw,
        "pr_auc": pr_auc_raw,
        "roc_auc_smoothed": roc_auc_smoothed,
        "pr_auc_smoothed": pr_auc_smoothed,
        "operating_point_pred_source": "pred_label",
        "fpr": fpr,
        "fnr": fnr,
        "false_alarms_per_minute": false_alarms_per_minute,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "time_to_first_detection": {
            "segments": ttf,
            "num_fake_segments": len(fake_segments),
            "num_detected_segments": sum(1 for x in ttf if x["detected"]),
            "mean_s": float(np.mean(detected_delays)) if detected_delays else None,
            "median_s": float(np.median(detected_delays)) if detected_delays else None,
        },
        "latency": {
            "infer_latency_ms": percentile_stats(infer_lat_ms),
            "end_to_end_latency_ms": percentile_stats(e2e_lat_ms),
        },
    }

    output_path = Path(args.output) if args.output else (run_dir / "metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_path = run_dir / "score_timeline.png"
    plot_generated = maybe_write_plot(rows, fake_segments, plot_path)
    print(f"metrics_json={output_path}")
    if plot_generated:
        print(f"timeline_plot={plot_path}")
    else:
        print("timeline_plot=not_generated")


if __name__ == "__main__":
    main()
