#!/usr/bin/env python3
# plot_partial_trajectories.py

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def compress_consecutive_points(points):
    out = []
    prev = None
    for p in points:
        if prev is None or p != prev:
            out.append(p)
        prev = p
    return out


def extract_gt_waypoints(df: pd.DataFrame):
    required = {"waypoint_idx", "target_x", "target_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Не хватает колонок для GT: {required - set(df.columns)}")

    gt = (
        df.sort_values(["waypoint_idx", "step"])
          .groupby("waypoint_idx", as_index=False)
          .first()[["target_x", "target_y"]]
          .dropna()
    )

    points = list(map(tuple, gt.to_numpy()))
    return compress_consecutive_points(points)


def should_save_trajectory(df: pd.DataFrame, min_waypoints: int = 5) -> bool:
    if "waypoint_idx" not in df.columns:
        return False
    n_unique = df["waypoint_idx"].dropna().nunique()
    return n_unique >= min_waypoints


def plot_trajectory(df: pd.DataFrame, csv_path: Path, out_path: Path):
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("Не хватает колонок x/y")

    actual = df[["x", "y"]].dropna().to_numpy()
    gt_points = extract_gt_waypoints(df)

    if len(actual) < 2:
        raise ValueError("Слишком мало точек реальной траектории")
    if len(gt_points) == 0:
        raise ValueError("GT-вейпоинты не найдены")

    plt.figure(figsize=(10, 8))
    plt.plot(actual[:, 0], actual[:, 1], linewidth=2, label="Real path")
    plt.scatter(actual[0, 0], actual[0, 1], s=60, marker="o", label="Start")
    plt.scatter(actual[-1, 0], actual[-1, 1], s=60, marker="x", label="End")

    gt_x = [p[0] for p in gt_points]
    gt_y = [p[1] for p in gt_points]
    plt.plot(gt_x, gt_y, linestyle="--", linewidth=2, marker="o", label="GT waypoints")

    plt.title(csv_path.stem)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def process_folder(input_dir: Path, output_dir: Path, min_waypoints: int = 5):
    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"CSV-файлы не найдены в {input_dir}")
        return

    processed = 0
    saved = 0
    skipped = 0

    for csv_path in csv_files:
        processed += 1
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] {csv_path.name}: не удалось прочитать CSV ({e})")
            skipped += 1
            continue

        try:
            if not should_save_trajectory(df, min_waypoints=min_waypoints):
                skipped += 1
                continue

            rel = csv_path.relative_to(input_dir)
            out_path = output_dir / rel.with_suffix(".png")

            plot_trajectory(df, csv_path, out_path)
            print(f"[OK] {csv_path.name} -> {out_path}")
            saved += 1

        except Exception as e:
            print(f"[SKIP] {csv_path.name}: {e}")
            skipped += 1

    print()
    print(f"Обработано: {processed}")
    print(f"Сохранено:  {saved}")
    print(f"Пропущено:  {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Папка с CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Куда сохранять PNG")
    parser.add_argument("--min-waypoints", type=int, default=5, help="Минимум уникальных waypoint_idx")
    args = parser.parse_args()

    process_folder(Path(args.input_dir), Path(args.output_dir), min_waypoints=args.min_waypoints)


if __name__ == "__main__":
    main()