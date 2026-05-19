# plot_training_metrics.py
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
]


def split_into_sessions(df: pd.DataFrame, episode_col: str = "episode_index") -> pd.DataFrame:
    """
    Делит лог на сессии обучения.
    Новая сессия начинается, когда episode_index уменьшается.
    """
    if episode_col not in df.columns:
        raise KeyError(f"Нет колонки {episode_col}")

    if df.empty:
        raise ValueError("CSV пустой.")

    session_ids = []
    session_id = 1
    prev_ep = None

    for ep in df[episode_col].astype(int).tolist():
        if prev_ep is not None and ep < prev_ep:
            session_id += 1
        session_ids.append(session_id)
        prev_ep = ep

    out = df.copy()
    out["session_id"] = session_ids
    return out


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Возвращает первую существующую колонку из списка candidates.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Не найдена ни одна из колонок: {candidates}")


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    """
    Простое скользящее среднее.
    """
    window = max(int(window), 1)
    return s.rolling(window=window, min_periods=1).mean()


def make_distinct_color_map(labels: list[str]) -> dict[str, tuple]:
    """
    Делает максимально разные цвета для любого числа категорий.
    Используем равномерно распределённые оттенки HSV.
    """
    labels = list(labels)
    n = max(len(labels), 1)

    color_map: dict[str, tuple] = {}
    for i, label in enumerate(labels):
        hue = i / n
        saturation = 0.75
        value = 0.90
        color_map[label] = colorsys.hsv_to_rgb(hue, saturation, value)

    return color_map


def save_color_map(color_map: dict[str, tuple], out_path: Path) -> None:
    rows = []
    for label, rgb in color_map.items():
        rows.append(
            {
                "path_type": label,
                "r": rgb[0],
                "g": rgb[1],
                "b": rgb[2],
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def prepare_session_df(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует типы данных в сессии.
    """
    df = session_df.copy()

    if "path_type" not in df.columns:
        df["path_type"] = "unknown"
    else:
        df["path_type"] = df["path_type"].fillna("unknown").astype(str)

    return df


def plot_metric_by_path_type(
    session_df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    color_map: dict[str, tuple],
    window: int = 20,
) -> None:
    """
    Строит один график метрики для одной сессии, разбивая линии по path_type.
    """
    df = session_df.copy()
    df = df.sort_values("episode_index")

    fig, ax = plt.subplots(figsize=(14, 6))

    path_types = list(df["path_type"].dropna().astype(str).unique())
    for path_type in path_types:
        g = df[df["path_type"].astype(str) == path_type].sort_values("episode_index")
        if g.empty:
            continue

        x = pd.to_numeric(g["episode_index"], errors="coerce")
        y = pd.to_numeric(g[metric_col], errors="coerce")

        mask = x.notna() & y.notna()
        x = x[mask].astype(int)
        y = y[mask].astype(float)

        if len(x) == 0:
            continue

        color = color_map.get(path_type, (0.2, 0.2, 0.2))

        # raw points
        ax.scatter(
            x,
            y,
            s=10,
            alpha=0.22,
            color=color,
            edgecolors="none",
        )

        # smoothed line
        ax.plot(
            x,
            smooth_series(y, window),
            linewidth=2,
            color=color,
            label=path_type,
        )

    ax.set_title(metric_col)
    ax.set_xlabel("episode_index")
    ax.set_ylabel(metric_col)
    ax.grid(True, alpha=0.25)
    ax.legend(title="path_type", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_session_dashboard(
    session_df: pd.DataFrame,
    session_id: int,
    out_dir: Path,
    color_map: dict[str, tuple],
    window: int = 20,
) -> None:
    """
    Для одной сессии строит набор графиков по важным метрикам.
    """
    available_metrics = [m for m in PRIMARY_METRICS if m in session_df.columns]
    if not available_metrics:
        return

    session_out = out_dir / f"session_{session_id:03d}"
    session_out.mkdir(parents=True, exist_ok=True)

    for metric in available_metrics:
        plot_metric_by_path_type(
            session_df=session_df,
            metric_col=metric,
            out_path=session_out / f"{metric}_by_path_type.png",
            color_map=color_map,
            window=window,
        )


def make_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Сохраняет сводные таблицы по сессиям и типам траекторий.
    """
    # Сводка по сессиям
    sessions_summary = (
        df.groupby("session_id")
        .agg(
            rows=("episode_index", "size"),
            start_episode=("episode_index", "min"),
            end_episode=("episode_index", "max"),
            unique_path_types=("path_type", pd.Series.nunique),
        )
        .reset_index()
    )
    sessions_summary.to_csv(out_dir / "sessions_summary.csv", index=False)

    # Сводка по сессиям и path_type
    agg_dict = {
        "episodes": ("episode_index", "count"),
        "start_episode": ("episode_index", "min"),
        "end_episode": ("episode_index", "max"),
    }

    for metric in [
        "completion_ratio",
        "episode_return",
        "success",
        "rmse_goal_distance",
        "rmse_path_deviation",
        "mean_heading_error",
        "mean_abs_roll",
        "mean_abs_pitch",
        "path_length",
        "waypoints_reached",
    ]:
        if metric in df.columns:
            agg_dict[f"{metric}_mean"] = (metric, "mean")
            agg_dict[f"{metric}_median"] = (metric, "median")

    by_type_summary = (
        df.groupby(["session_id", "path_type"])
        .agg(**agg_dict)
        .reset_index()
    )
    by_type_summary.to_csv(out_dir / "summary_by_session_and_path_type.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Путь к training_metrics.csv")
    parser.add_argument(
        "--out",
        type=str,
        default="training_plots",
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

    if "episode_index" not in df.columns:
        raise KeyError("В CSV нет колонки episode_index.")

    # Не сортируем по episode_index, иначе потеряем факт рестарта.
    # Сохраняем исходный порядок строк.
    df = split_into_sessions(df, episode_col="episode_index")
    df["path_type"] = df.get("path_type", "unknown").fillna("unknown").astype(str)

    # Цвета для всех типов траекторий — единый стабильный маппинг.
    unique_path_types = sorted(df["path_type"].dropna().astype(str).unique().tolist())
    color_map = make_distinct_color_map(unique_path_types)
    save_color_map(color_map, out_dir / "path_type_color_map.csv")

    # Сводки
    make_summaries(df, out_dir)

    # Графики по каждой сессии
    for session_id, session_df in df.groupby("session_id", sort=True):
        session_df = prepare_session_df(session_df)
        plot_session_dashboard(
            session_df=session_df,
            session_id=int(session_id),
            out_dir=out_dir,
            color_map=color_map,
            window=args.window,
        )

    print(f"Готово. Результаты сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()