# plot_training_metrics_linear.py
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PRIMARY_METRICS = [
    "completion_ratio",
    "episode_return",
    "success",
    "rmse_goal_distance",
    "rmse_path_deviation",
    "mean_heading_error",
    "mean_abs_roll",
    "mean_abs_pitch",
    "path_length",
    "steps",
]


def make_distinct_color_map(labels: list[str]) -> dict[str, tuple]:
    labels = list(labels)
    n = max(len(labels), 1)

    color_map: dict[str, tuple] = {}
    for i, label in enumerate(labels):
        hue = i / n
        saturation = 0.75
        value = 0.90
        color_map[label] = colorsys.hsv_to_rgb(hue, saturation, value)

    return color_map


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return s.rolling(window=window, min_periods=1).mean()


def find_existing_metric(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Не найдена ни одна из колонок: {candidates}")


def plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    x_col: str,
    out_path: Path,
    color_map: dict[str, tuple] | None = None,
    window: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    if "path_type" in df.columns:
        path_types = sorted(df["path_type"].fillna("unknown").astype(str).unique().tolist())
    else:
        path_types = ["all"]

    if color_map is None:
        color_map = make_distinct_color_map(path_types)

    for path_type in path_types:
        if "path_type" in df.columns:
            g = df[df["path_type"].fillna("unknown").astype(str) == path_type].copy()
        else:
            g = df.copy()

        if g.empty:
            continue

        x = pd.to_numeric(g[x_col], errors="coerce")
        y = pd.to_numeric(g[metric_col], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask].astype(int)
        y = y[mask].astype(float)

        if len(x) == 0:
            continue

        color = color_map.get(path_type, (0.2, 0.2, 0.2))

        ax.scatter(x, y, s=10, alpha=0.22, color=color, edgecolors="none")
        ax.plot(
            x,
            smooth_series(y, window),
            linewidth=2,
            color=color,
            label=path_type,
        )

    ax.set_title(metric_col)
    ax.set_xlabel("global_episode")
    ax.set_ylabel(metric_col)
    ax.grid(True, alpha=0.25)
    if "path_type" in df.columns:
        ax.legend(title="path_type", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Путь к training_metrics.csv")
    parser.add_argument(
        "--out",
        type=str,
        default="training_plots_linear",
        help="Папка для сохранения графиков",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Окно сглаживания",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV пустой.")

    # Главное отличие от предыдущего скрипта:
    # номер эпизода определяется просто порядком строк в файле.
    df = df.reset_index(drop=True)
    df["global_episode"] = df.index + 1

    if "path_type" in df.columns:
        df["path_type"] = df["path_type"].fillna("unknown").astype(str)
        unique_path_types = sorted(df["path_type"].unique().tolist())
        color_map = make_distinct_color_map(unique_path_types)
    else:
        color_map = {"all": (0.2, 0.2, 0.2)}

    pd.DataFrame(
        [
            {"path_type": k, "r": v[0], "g": v[1], "b": v[2]}
            for k, v in color_map.items()
        ]
    ).to_csv(out_dir / "path_type_color_map.csv", index=False)

    # Сводка по всему файлу как по одной длинной сессии
    summary_rows = []
    for metric in ["completion_ratio", "episode_return", "success", "rmse_goal_distance"]:
        if metric in df.columns:
            summary_rows.append(
                {
                    "metric": metric,
                    "mean": pd.to_numeric(df[metric], errors="coerce").mean(),
                    "median": pd.to_numeric(df[metric], errors="coerce").median(),
                    "min": pd.to_numeric(df[metric], errors="coerce").min(),
                    "max": pd.to_numeric(df[metric], errors="coerce").max(),
                }
            )
    pd.DataFrame(summary_rows).to_csv(out_dir / "overall_summary.csv", index=False)

    # Графики по важным метрикам
    # steps тоже входит в список, поэтому отдельный график steps_by_path_type.png
    # будет сгенерирован автоматически тем же циклом ниже.
    available_metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    for metric in available_metrics:
        plot_metric(
            df=df,
            metric_col=metric,
            x_col="global_episode",
            out_path=out_dir / f"{metric}_by_path_type.png",
            color_map=color_map,
            window=args.window,
        )

    # Отдельно полезный график completion_ratio, если колонка есть
    if "completion_ratio" in df.columns:
        plot_metric(
            df=df,
            metric_col="completion_ratio",
            x_col="global_episode",
            out_path=out_dir / "completion_ratio.png",
            color_map=color_map,
            window=args.window,
        )

    print(f"Готово. Результаты сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()