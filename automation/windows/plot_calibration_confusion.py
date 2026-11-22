#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BACKEND_ORDER = [
    'f3net',
    'xception_df40',
    'efficientnet_b4',
    'effort_clip_l14',
    'i3d',
    'videomae',
]

BACKEND_LABELS = {
    'f3net': 'F3Net',
    'xception_df40': 'Xception DF40',
    'efficientnet_b4': 'EfficientNet-B4',
    'effort_clip_l14': 'Effort CLIP-L14',
    'i3d': 'I3D',
    'videomae': 'VideoMAE',
}

LIVE_ROOTS = {
    'f3net': Path('/Users/abx/Y3 Project/repo/results_f3net_ft_clean_live_2026-04-16'),
    'xception_df40': Path('/Users/abx/Y3 Project/repo/results_ft_clean_xception_efficientnet_live_retry_2026-04-16'),
    'efficientnet_b4': Path('/Users/abx/Y3 Project/repo/results_ft_clean_xception_efficientnet_live_retry_2026-04-16'),
    'effort_clip_l14': Path('/Users/abx/Y3 Project/repo/results_ft_clean_effort_live_2026-04-17'),
    'i3d': Path('/Users/abx/Y3 Project/repo/results_ft_clean_i3d_live_2026-04-17'),
    'videomae': Path('/Users/abx/Y3 Project/repo/results_ft_clean_videomae_live_2026-04-17'),
}


@dataclass(frozen=True)
class Policy:
    backend: str
    polarity: str
    t_on: float
    t_off: float
    persistence: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate the calibration confusion-matrix figure from a calibration bundle.'
    )
    parser.add_argument(
        '--calibration-dir',
        required=True,
        help='Path to a results_finetune_clean_fpr_calibration_* directory containing calibration_metrics.csv',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to the output PNG figure.',
    )
    parser.add_argument(
        '--counts-output',
        default='',
        help='Optional path to write per-backend TP/FN/FP/TN counts as CSV.',
    )
    parser.add_argument(
        '--title',
        default='',
        help='Optional figure title. Defaults to a title derived from the bundle metadata.',
    )
    return parser.parse_args()


def load_window_seconds(calibration_dir: Path) -> int:
    thresholds_path = calibration_dir / 'thresholds.json'
    if thresholds_path.exists():
        with thresholds_path.open() as f:
            data = json.load(f)
        meta = data.get('_meta', {})
        win = meta.get('window_seconds')
        if isinstance(win, int):
            return win
    raise FileNotFoundError(f'Could not determine window_seconds from {thresholds_path}')


def load_policies(calibration_dir: Path) -> Dict[str, Policy]:
    metrics_path = calibration_dir / 'calibration_metrics.csv'
    policies: Dict[str, Policy] = {}
    with metrics_path.open(newline='') as f:
        for row in csv.DictReader(f):
            backend = row['backend']
            policies[backend] = Policy(
                backend=backend,
                polarity=row['polarity'],
                t_on=float(row['t_on']),
                t_off=float(row['t_off']),
                persistence=int(row['persistence_windows']),
            )
    missing = [bk for bk in BACKEND_ORDER if bk not in policies]
    if missing:
        raise ValueError(f'Missing backend policies in {metrics_path}: {missing}')
    return policies


def orient_score(score: float, polarity: str) -> float:
    if polarity == 'low':
        return 1.0 - score
    return score


def iter_prediction_files(backend: str) -> Iterable[Path]:
    root = LIVE_ROOTS[backend]
    pattern = f'calibration_*__{backend}/predictions.csv'
    for path in sorted(root.glob(pattern)):
        yield path


def aggregate_windows(predictions_csv: Path, policy: Policy, window_seconds: int) -> Tuple[List[float], List[int]]:
    buckets: Dict[int, List[float]] = {}
    labels: Dict[int, List[int]] = {}

    with predictions_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            score_raw = row.get('score_raw', '')
            if score_raw == '':
                continue
            t_rel = row.get('t_rel', '')
            gt_label = row.get('gt_label', '')
            if t_rel == '':
                continue
            win_idx = int(float(t_rel)) // window_seconds
            buckets.setdefault(win_idx, []).append(float(score_raw))
            if gt_label not in ('', 'unknown'):
                labels.setdefault(win_idx, []).append(int(gt_label))

    win_scores: List[float] = []
    win_labels: List[int] = []
    for win_idx in sorted(buckets):
        lbls = labels.get(win_idx, [])
        if not lbls:
            continue
        score = float(np.mean(buckets[win_idx]))
        score = orient_score(score, policy.polarity)
        majority = 1 if (sum(lbls) / len(lbls)) >= 0.5 else 0
        win_scores.append(score)
        win_labels.append(majority)
    return win_scores, win_labels


def apply_hysteresis(scores: Iterable[float], policy: Policy) -> List[int]:
    state = False
    on_run = 0
    off_run = 0
    output: List[int] = []
    for score in scores:
        if not state:
            if score >= policy.t_on:
                on_run += 1
                if on_run >= policy.persistence:
                    state = True
                    off_run = 0
            else:
                on_run = 0
        else:
            if score <= policy.t_off:
                off_run += 1
                if off_run >= policy.persistence:
                    state = False
                    on_run = 0
            else:
                off_run = 0
        output.append(1 if state else 0)
    return output


def confusion_counts(states: Iterable[int], labels: Iterable[int]) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for state, label in zip(states, labels):
        if label == 1 and state == 1:
            tp += 1
        elif label == 1 and state == 0:
            fn += 1
        elif label == 0 and state == 1:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def build_counts(calibration_dir: Path) -> Tuple[int, Dict[str, Tuple[int, int, int, int]]]:
    window_seconds = load_window_seconds(calibration_dir)
    policies = load_policies(calibration_dir)
    counts: Dict[str, Tuple[int, int, int, int]] = {}
    for backend in BACKEND_ORDER:
        all_scores: List[float] = []
        all_labels: List[int] = []
        for predictions_csv in iter_prediction_files(backend):
            scores, labels = aggregate_windows(predictions_csv, policies[backend], window_seconds)
            all_scores.extend(scores)
            all_labels.extend(labels)
        states = apply_hysteresis(all_scores, policies[backend])
        counts[backend] = confusion_counts(states, all_labels)
    return window_seconds, counts


def render_figure(output_path: Path, window_seconds: int, counts: Dict[str, Tuple[int, int, int, int]], title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.5))
    axes = axes.flatten()
    cmap = plt.cm.Blues
    vmax = max(max(vals) for vals in counts.values())

    for ax, backend in zip(axes, BACKEND_ORDER):
        tp, fn, fp, tn = counts[backend]
        matrix = np.array([[tp, fn], [fp, tn]])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(BACKEND_LABELS[backend], fontsize=11, pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pred Fake', 'Pred Real'], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Actual\nFake', 'Actual\nReal'], fontsize=9)

        labels = [['TP', 'FN'], ['FP', 'TN']]
        for i in range(2):
            for j in range(2):
                val = int(matrix[i, j])
                text_color = 'white' if val > (0.55 * vmax) else 'black'
                ax.text(j, i - 0.10, labels[i][j], ha='center', va='center', fontsize=10, color=text_color)
                ax.text(j, i + 0.14, str(val), ha='center', va='center', fontsize=18, fontweight='bold', color=text_color)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_counts_csv(path: Path, counts: Dict[str, Tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['backend', 'tp', 'fn', 'fp', 'tn'])
        for backend in BACKEND_ORDER:
            tp, fn, fp, tn = counts[backend]
            writer.writerow([backend, tp, fn, fp, tn])


def main() -> None:
    args = parse_args()
    calibration_dir = Path(args.calibration_dir)
    output_path = Path(args.output)
    counts_output = Path(args.counts_output) if args.counts_output else None

    window_seconds, counts = build_counts(calibration_dir)
    title = args.title or f'Confusion at the calibrated {window_seconds}-s operating point (calibration windows)'
    render_figure(output_path, window_seconds, counts, title)

    if counts_output is not None:
        write_counts_csv(counts_output, counts)

    for backend in BACKEND_ORDER:
        tp, fn, fp, tn = counts[backend]
        print(f'{backend}: TP={tp} FN={fn} FP={fp} TN={tn}')
    print(f'Saved figure to {output_path}')
    if counts_output is not None:
        print(f'Saved counts to {counts_output}')


if __name__ == '__main__':
    main()
