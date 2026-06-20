from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
HNM_OUTPUT_DIR = OUTPUT_DIR / "hard_negative_mining"
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / "05_hard_negative_mining"

HNM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_PATH = HNM_OUTPUT_DIR / "image_loss_components.csv"

TOP_N = 250

COMPONENTS = [
    "loc_loss",
    "conf_loss_obj",
    "conf_loss_noobj",
    "class_loss",
]

COMPONENT_LABELS = {
    "loc_loss": "Localization",
    "conf_loss_obj": "Objectness",
    "conf_loss_noobj": "No-object",
    "class_loss": "Classification",
}

CONFIGS = [
    {
        "config_name": "balanced_default",
        "display_name": "Balanced",
        "lambda_coord": 0.5,
        "lambda_obj": 0.5,
        "lambda_noobj": 0.5,
        "lambda_cls": 0.5,
        "family": "main",
    },
    {
        "config_name": "localization_heavy",
        "display_name": "Localization-heavy",
        "lambda_coord": 2.0,
        "lambda_obj": 0.5,
        "lambda_noobj": 0.5,
        "lambda_cls": 0.5,
        "family": "main",
    },
    {
        "config_name": "objectness_heavy",
        "display_name": "Objectness-heavy",
        "lambda_coord": 0.5,
        "lambda_obj": 2.0,
        "lambda_noobj": 0.5,
        "lambda_cls": 0.5,
        "family": "main",
    },
    {
        "config_name": "no_object_heavy",
        "display_name": "No-object-heavy",
        "lambda_coord": 0.5,
        "lambda_obj": 0.5,
        "lambda_noobj": 2.0,
        "lambda_cls": 0.5,
        "family": "main",
    },
    {
        "config_name": "classification_heavy",
        "display_name": "Classification-heavy",
        "lambda_coord": 0.5,
        "lambda_obj": 0.5,
        "lambda_noobj": 0.5,
        "lambda_cls": 2.0,
        "family": "main",
    },
    {
        "config_name": "localization_only",
        "display_name": "Localization-only",
        "lambda_coord": 1.0,
        "lambda_obj": 0.0,
        "lambda_noobj": 0.0,
        "lambda_cls": 0.0,
        "family": "component_only",
    },
    {
        "config_name": "objectness_only",
        "display_name": "Objectness-only",
        "lambda_coord": 0.0,
        "lambda_obj": 1.0,
        "lambda_noobj": 0.0,
        "lambda_cls": 0.0,
        "family": "component_only",
    },
    {
        "config_name": "no_object_only",
        "display_name": "No-object-only",
        "lambda_coord": 0.0,
        "lambda_obj": 0.0,
        "lambda_noobj": 1.0,
        "lambda_cls": 0.0,
        "family": "component_only",
    },
    {
        "config_name": "classification_only",
        "display_name": "Classification-only",
        "lambda_coord": 0.0,
        "lambda_obj": 0.0,
        "lambda_noobj": 0.0,
        "lambda_cls": 1.0,
        "family": "component_only",
    },
]


def parse_class_names(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [value]


def weighted_score(df, cfg, mode, scales):
    if mode == "raw":
        loc = df["loc_loss"]
        obj = df["conf_loss_obj"]
        noobj = df["conf_loss_noobj"]
        cls = df["class_loss"]
    elif mode == "normalized":
        loc = df["loc_loss"] / scales["loc_loss"]
        obj = df["conf_loss_obj"] / scales["conf_loss_obj"]
        noobj = df["conf_loss_noobj"] / scales["conf_loss_noobj"]
        cls = df["class_loss"] / scales["class_loss"]
    else:
        raise ValueError(mode)

    return (
        cfg["lambda_coord"] * loc
        + cfg["lambda_obj"] * obj
        + cfg["lambda_noobj"] * noobj
        + cfg["lambda_cls"] * cls
    )


def contribution_columns(df, cfg, mode, scales):
    if mode == "raw":
        base = {
            "loc_loss": df["loc_loss"],
            "conf_loss_obj": df["conf_loss_obj"],
            "conf_loss_noobj": df["conf_loss_noobj"],
            "class_loss": df["class_loss"],
        }
    else:
        base = {
            "loc_loss": df["loc_loss"] / scales["loc_loss"],
            "conf_loss_obj": df["conf_loss_obj"] / scales["conf_loss_obj"],
            "conf_loss_noobj": df["conf_loss_noobj"] / scales["conf_loss_noobj"],
            "class_loss": df["class_loss"] / scales["class_loss"],
        }

    contrib = pd.DataFrame({
        "contrib_loc": cfg["lambda_coord"] * base["loc_loss"],
        "contrib_obj": cfg["lambda_obj"] * base["conf_loss_obj"],
        "contrib_noobj": cfg["lambda_noobj"] * base["conf_loss_noobj"],
        "contrib_cls": cfg["lambda_cls"] * base["class_loss"],
    })

    labels = {
        "contrib_loc": "Localization",
        "contrib_obj": "Objectness",
        "contrib_noobj": "No-object",
        "contrib_cls": "Classification",
    }

    dominant = contrib.idxmax(axis=1).map(labels)
    return contrib, dominant


def build_top_samples(df, scales):
    rows = []

    for mode in ["raw", "normalized"]:
        for cfg in CONFIGS:
            scored = df.copy()
            scored["score_mode"] = mode
            scored["config_name"] = cfg["config_name"]
            scored["display_name"] = cfg["display_name"]
            scored["family"] = cfg["family"]
            scored["lambda_coord"] = cfg["lambda_coord"]
            scored["lambda_obj"] = cfg["lambda_obj"]
            scored["lambda_noobj"] = cfg["lambda_noobj"]
            scored["lambda_cls"] = cfg["lambda_cls"]
            scored["hnm_score"] = weighted_score(scored, cfg, mode, scales)

            contrib, dominant = contribution_columns(scored, cfg, mode, scales)
            scored = pd.concat([scored, contrib], axis=1)
            scored["dominant_contribution"] = dominant

            scored = scored.sort_values(
                ["hnm_score", "image_file"],
                ascending=[False, True]
            ).head(TOP_N).copy()

            scored["rank"] = np.arange(1, len(scored) + 1)

            rows.append(scored)

    return pd.concat(rows, axis=0, ignore_index=True)


def summarize_top(top_df):
    summary_rows = []

    group_cols = [
        "score_mode",
        "config_name",
        "display_name",
        "family",
        "lambda_coord",
        "lambda_obj",
        "lambda_noobj",
        "lambda_cls",
    ]

    for keys, group in top_df.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))

        row = {
            **d,
            "top_n": len(group),
            "mean_hnm_score": group["hnm_score"].mean(),
            "median_hnm_score": group["hnm_score"].median(),
            "mean_num_objects": group["num_objects"].mean(),
            "median_num_objects": group["num_objects"].median(),
            "mean_raw_prediction_count": group["raw_prediction_count"].mean(),
            "mean_matched_prediction_count": group["matched_prediction_count"].mean(),
            "mean_false_positive_prediction_count": group["false_positive_prediction_count"].mean(),
            "mean_missed_gt_count": group["missed_gt_count"].mean(),
            "zero_prediction_images": int((group["raw_prediction_count"] == 0).sum()),
            "share_zero_prediction_images": float((group["raw_prediction_count"] == 0).mean()),
            "mean_loc_loss": group["loc_loss"].mean(),
            "mean_conf_loss_obj": group["conf_loss_obj"].mean(),
            "mean_conf_loss_noobj": group["conf_loss_noobj"].mean(),
            "mean_class_loss": group["class_loss"].mean(),
        }

        for component in ["Localization", "Objectness", "No-object", "Classification"]:
            row[f"dominant_{component}_share"] = float((group["dominant_contribution"] == component).mean())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_overlap(top_df):
    rows = []

    for mode, mode_group in top_df.groupby("score_mode"):
        configs = list(mode_group["config_name"].drop_duplicates())

        sets = {
            cfg: set(mode_group[mode_group["config_name"] == cfg]["image_path"])
            for cfg in configs
        }

        for cfg_a in configs:
            for cfg_b in configs:
                inter = len(sets[cfg_a] & sets[cfg_b])
                union = len(sets[cfg_a] | sets[cfg_b])
                rows.append({
                    "score_mode": mode,
                    "config_a": cfg_a,
                    "config_b": cfg_b,
                    "intersection_count": inter,
                    "jaccard_overlap": inter / union if union else 0.0,
                })

    return pd.DataFrame(rows)


def build_density(top_df):
    rows = []
    order = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]

    for (mode, cfg, display), group in top_df.groupby(["score_mode", "config_name", "display_name"]):
        counts = group["density_bucket"].value_counts()
        total = len(group)

        for bucket in order:
            count = int(counts.get(bucket, 0))
            rows.append({
                "score_mode": mode,
                "config_name": cfg,
                "display_name": display,
                "density_bucket": bucket,
                "image_count": count,
                "image_share": count / total if total else 0.0,
            })

    return pd.DataFrame(rows)


def build_class_presence(top_df):
    rows = []

    for (mode, cfg, display), group in top_df.groupby(["score_mode", "config_name", "display_name"]):
        total = len(group)
        class_counts = {}

        for value in group["class_names_present"]:
            for class_name in parse_class_names(value):
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name, count in class_counts.items():
            rows.append({
                "score_mode": mode,
                "config_name": cfg,
                "display_name": display,
                "class_name": class_name,
                "image_count": int(count),
                "image_share": count / total if total else 0.0,
            })

    return pd.DataFrame(rows)


def plot_component_scale(df, scales):
    stats = pd.DataFrame({
        "component": ["loc_loss", "conf_loss_obj", "conf_loss_noobj", "class_loss"],
        "mean": [df[c].mean() for c in COMPONENTS],
        "p95": [scales[c] for c in COMPONENTS],
    })

    x = np.arange(len(stats))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, stats["mean"], width, label="Mean")
    plt.bar(x + width / 2, stats["p95"], width, label="95th percentile")
    plt.yscale("log")
    plt.xticks(x, [COMPONENT_LABELS[c] for c in stats["component"]], rotation=20, ha="right")
    plt.ylabel("Loss value, log scale")
    plt.title("Loss component scale before λ weighting")
    plt.legend()
    plt.tight_layout()
    path = FIGURE_DIR / "01_component_scale.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", path)


def plot_overlap(overlap_df, mode, configs, filename, title):
    sub = overlap_df[
        (overlap_df["score_mode"] == mode)
        & (overlap_df["config_a"].isin(configs))
        & (overlap_df["config_b"].isin(configs))
    ].copy()

    matrix = sub.pivot(index="config_a", columns="config_b", values="jaccard_overlap").loc[configs, configs]

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.values, aspect="auto", vmin=0, vmax=1)
    plt.xticks(range(len(configs)), configs, rotation=45, ha="right")
    plt.yticks(range(len(configs)), configs)
    plt.colorbar(label="Jaccard overlap")
    plt.title(title)

    for i in range(len(configs)):
        for j in range(len(configs)):
            plt.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", path)


def plot_density(density_df, mode, configs, filename):
    sub = density_df[
        (density_df["score_mode"] == mode)
        & (density_df["config_name"].isin(configs))
    ].copy()

    pivot = sub.pivot(index="display_name", columns="density_bucket", values="image_share").fillna(0.0)
    config_display_order = [
        next(c["display_name"] for c in CONFIGS if c["config_name"] == cfg)
        for cfg in configs
    ]
    bucket_order = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]
    pivot = pivot.loc[config_display_order, bucket_order]

    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_ylabel("Share of sampled images")
    ax.set_xlabel("λ configuration")
    ax.set_title(f"Density bucket distribution of top-{TOP_N} HNM samples ({mode})")
    ax.legend(title="Density bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", path)


def plot_profile(summary_df, mode, configs, filename):
    sub = summary_df[
        (summary_df["score_mode"] == mode)
        & (summary_df["config_name"].isin(configs))
    ].copy()

    sub["display_order"] = sub["config_name"].map({cfg: i for i, cfg in enumerate(configs)})
    sub = sub.sort_values("display_order")

    labels = sub["display_name"].tolist()
    metrics = [
        ("mean_num_objects", "Mean objects"),
        ("mean_raw_prediction_count", "Mean raw predictions"),
        ("mean_false_positive_prediction_count", "Mean false positives"),
        ("mean_missed_gt_count", "Mean missed GT"),
    ]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(11, 6))

    for idx, (col, label) in enumerate(metrics):
        plt.bar(x + (idx - 1.5) * width, sub[col].values, width, label=label)

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Average count per sampled image")
    plt.title(f"Image profile of top-{TOP_N} HNM samples ({mode})")
    plt.legend()
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", path)


def plot_dominant(summary_df, mode, configs, filename):
    sub = summary_df[
        (summary_df["score_mode"] == mode)
        & (summary_df["config_name"].isin(configs))
    ].copy()

    sub["display_order"] = sub["config_name"].map({cfg: i for i, cfg in enumerate(configs)})
    sub = sub.sort_values("display_order")

    pivot = sub.set_index("display_name")[[
        "dominant_Localization_share",
        "dominant_Objectness_share",
        "dominant_No-object_share",
        "dominant_Classification_share",
    ]]
    pivot.columns = ["Localization", "Objectness", "No-object", "Classification"]

    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_ylabel("Share of sampled images")
    ax.set_xlabel("λ configuration")
    ax.set_title(f"Dominant loss contribution among top-{TOP_N} HNM samples ({mode})")
    ax.legend(title="Dominant contribution", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", path)


def plot_class_presence(class_df, mode, configs, filename):
    sub = class_df[
        (class_df["score_mode"] == mode)
        & (class_df["config_name"].isin(configs))
    ].copy()

    top_classes = (
        sub.groupby("class_name")["image_count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )

    sub = sub[sub["class_name"].isin(top_classes)]

    pivot = sub.pivot_table(
        index="class_name",
        columns="display_name",
        values="image_share",
        aggfunc="mean",
        fill_value=0.0,
    )

    config_display_order = [
        next(c["display_name"] for c in CONFIGS if c["config_name"] == cfg)
        for cfg in configs
    ]
    pivot = pivot[config_display_order]

    plt.figure(figsize=(10, 7))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Share of sampled images containing class")
    plt.title(f"Class presence in top-{TOP_N} HNM samples ({mode})")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", path)


def main():
    if not COMPONENT_PATH.exists():
        raise FileNotFoundError(f"Missing component file: {COMPONENT_PATH}")

    df = pd.read_csv(COMPONENT_PATH)

    scales = {}
    for col in COMPONENTS:
        p95 = float(df[col].quantile(0.95))
        scales[col] = p95 if p95 > 0 else 1.0

    print("=" * 100)
    print("HNM λ SWEEP")
    print("=" * 100)
    print("Top-N sampled images per configuration:", TOP_N)
    print("\nRobust normalization scales, 95th percentile:")
    for col, scale in scales.items():
        print(f"{col:20s}: {scale:.6f}")

    config_df = pd.DataFrame(CONFIGS)
    config_path = HNM_OUTPUT_DIR / "lambda_configurations.csv"
    config_df.to_csv(config_path, index=False)
    print("[WRITE]", config_path)

    top_df = build_top_samples(df, scales)
    summary_df = summarize_top(top_df)
    overlap_df = build_overlap(top_df)
    density_df = build_density(top_df)
    class_df = build_class_presence(top_df)

    top_path = HNM_OUTPUT_DIR / "top_images_by_lambda.csv"
    summary_path = HNM_OUTPUT_DIR / "lambda_summary.csv"
    overlap_path = HNM_OUTPUT_DIR / "lambda_overlap.csv"
    density_path = HNM_OUTPUT_DIR / "density_by_lambda.csv"
    class_path = HNM_OUTPUT_DIR / "class_presence_by_lambda.csv"

    top_df.to_csv(top_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)
    density_df.to_csv(density_path, index=False)
    class_df.to_csv(class_path, index=False)

    print("[WRITE]", top_path)
    print("[WRITE]", summary_path)
    print("[WRITE]", overlap_path)
    print("[WRITE]", density_path)
    print("[WRITE]", class_path)

    main_configs = [
        "balanced_default",
        "localization_heavy",
        "objectness_heavy",
        "no_object_heavy",
        "classification_heavy",
    ]

    plot_component_scale(df, scales)

    plot_overlap(
        overlap_df,
        mode="raw",
        configs=main_configs,
        filename="02_raw_overlap_by_lambda.png",
        title=f"Top-{TOP_N} sample overlap across raw λ settings",
    )

    plot_overlap(
        overlap_df,
        mode="normalized",
        configs=main_configs,
        filename="03_normalized_overlap_by_lambda.png",
        title=f"Top-{TOP_N} sample overlap across scale-aware λ settings",
    )

    plot_density(
        density_df,
        mode="normalized",
        configs=main_configs,
        filename="04_density_distribution_by_lambda.png",
    )

    plot_profile(
        summary_df,
        mode="normalized",
        configs=main_configs,
        filename="05_image_profile_by_lambda.png",
    )

    plot_dominant(
        summary_df,
        mode="normalized",
        configs=main_configs,
        filename="06_dominant_component_by_lambda.png",
    )

    plot_class_presence(
        class_df,
        mode="normalized",
        configs=main_configs,
        filename="07_class_presence_by_lambda.png",
    )

    print()
    print("=" * 100)
    print("MAIN λ CONFIGURATION SUMMARY")
    print("=" * 100)

    print(
        summary_df[
            (summary_df["family"] == "main")
            & (summary_df["score_mode"].isin(["raw", "normalized"]))
        ][[
            "score_mode",
            "display_name",
            "top_n",
            "mean_num_objects",
            "mean_raw_prediction_count",
            "mean_false_positive_prediction_count",
            "mean_missed_gt_count",
            "zero_prediction_images",
            "mean_loc_loss",
            "mean_conf_loss_obj",
            "mean_conf_loss_noobj",
            "mean_class_loss",
            "dominant_Localization_share",
            "dominant_Objectness_share",
            "dominant_No-object_share",
            "dominant_Classification_share",
        ]].sort_values(["score_mode", "display_name"]).to_string(index=False)
    )

    print()
    print("=" * 100)
    print("RAW MAIN CONFIG OVERLAP")
    print("=" * 100)
    raw_main = overlap_df[
        (overlap_df["score_mode"] == "raw")
        & (overlap_df["config_a"].isin(main_configs))
        & (overlap_df["config_b"].isin(main_configs))
    ]
    print(raw_main.pivot(index="config_a", columns="config_b", values="jaccard_overlap").loc[main_configs, main_configs].round(3).to_string())

    print()
    print("=" * 100)
    print("NORMALIZED MAIN CONFIG OVERLAP")
    print("=" * 100)
    norm_main = overlap_df[
        (overlap_df["score_mode"] == "normalized")
        & (overlap_df["config_a"].isin(main_configs))
        & (overlap_df["config_b"].isin(main_configs))
    ]
    print(norm_main.pivot(index="config_a", columns="config_b", values="jaccard_overlap").loc[main_configs, main_configs].round(3).to_string())

    print()
    print("DONE.")


if __name__ == "__main__":
    main()
