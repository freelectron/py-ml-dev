# %%
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv("./static/data/data_sensors.csv")
df_labeled = df[~df["Label"].isna()]
df_unlabeled = df[df["Label"].isna()]
sensor_cols = [c for c in df.columns if c.startswith("Sensor")]

# %%
ANGLES: list[tuple[int, int]] = [(20, 0), (20, 60), (20, 120), (45, 180), (70, 240), (10, 300)]
FOCUS_SENSORS = ["Sensor 2", "Sensor 9", "Sensor 13"]


def label_colors(labels: np.ndarray) -> tuple[list[float], dict[float, np.ndarray]]:
    unique = sorted(set(labels))
    colors = cm.tab10(np.linspace(0, 0.4, len(unique)))
    return unique, {l: c for l, c in zip(unique, colors)}


def add_radius_feature(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].values
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)
    return np.concat([X, R], axis=1)


def plot_sensor_overview(df: pd.DataFrame, sensor_cols: list[str]) -> None:
    n = len(sensor_cols)
    colors = cm.tab20(np.linspace(0, 1, n))
    fig, (ax_dist, ax_time) = plt.subplots(1, 2, figsize=(16, 6))
    for i, col in enumerate(sensor_cols):
        ax_dist.hist(df[col].dropna(), bins=30, alpha=0.4, color=colors[i], label=col)
        ax_time.plot(df.index, df[col], linewidth=0.7, alpha=0.6, color=colors[i], label=col)
    ax_dist.set(title="All sensors — distribution", xlabel="value", ylabel="count")
    ax_dist.legend(fontsize=7, ncol=2)
    ax_time.set(title="All sensors — readings over time", xlabel="index (time)", ylabel="value")
    ax_time.legend(fontsize=7, ncol=2)
    plt.suptitle("Sensor overview", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_sensor_detail(df: pd.DataFrame, sensor_cols: list[str], acf_lags: int = 50) -> None:
    n = len(sensor_cols)
    colors = cm.tab20(np.linspace(0, 1, n))
    for i, col in enumerate(sensor_cols):
        fig, axes = plt.subplot_mosaic(
            [["hist", "time"], ["hist", "acf"]],
            figsize=(16, 6),
        )
        color = colors[i % n]
        axes["hist"].hist(df[col].dropna(), bins=30, alpha=0.7, color=color, edgecolor="black")
        axes["hist"].set(title=f"{col} — Distribution", xlabel="Value", ylabel="Count")
        axes["time"].plot(df.index, df[col], linewidth=1, color=color)
        axes["time"].set(title=f"{col} — Readings over Time", xlabel="Index (Time)", ylabel="Value")
        plot_acf(df[col].dropna(), lags=acf_lags, ax=axes["acf"], color=color, vlines_kwargs={"colors": color})
        axes["acf"].set_title(f"{col} — ACF (lags={acf_lags})")
        plt.suptitle(f"Analysis for {col}", fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_label_pie(df_labeled: pd.DataFrame) -> None:
    labels = df_labeled["Label"].values
    unique_labels, label_color_map = label_colors(labels)
    counts = [np.sum(labels == l) for l in unique_labels]
    colors = [label_color_map[l] for l in unique_labels]
    fig, ax = plt.subplots(figsize=(7, 7))
    _, _, autotexts = ax.pie(
        counts,
        labels=[f"Class {int(l)}" for l in unique_labels],
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * sum(counts) / 100))})",
        startangle=140,
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Label distribution (labeled data)", fontsize=13)
    plt.tight_layout()
    plt.show()


def _draw_violin_box(
    ax: Axes,
    data_groups: list[np.ndarray],
    positions: list[int],
    group_colors: list,
    x_labels: list[str],
) -> None:
    parts = ax.violinplot(data_groups, positions=positions, showextrema=False, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(group_colors[i])
        pc.set_alpha(0.45)
    bp = ax.boxplot(
        data_groups, positions=positions, widths=0.25, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(group_colors[i])
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)


def plot_violin_box_by_sensor(
    df_labeled: pd.DataFrame,
    df_unlabeled: pd.DataFrame,
    sensors: list[str],
) -> None:
    labels = df_labeled["Label"].values
    unique_labels, label_color_map = label_colors(labels)
    colors = [label_color_map[l] for l in unique_labels]
    for sensor in sensors:
        fig, (ax_lab, ax_unlab) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{sensor} — box + violin", fontsize=13)
        data_labeled = [df_labeled.loc[labels == l, sensor].values for l in unique_labels]
        _draw_violin_box(ax_lab, data_labeled, list(range(len(unique_labels))),
                         colors, [f"Class {int(l)}" for l in unique_labels])
        ax_lab.set(xlabel="Label", ylabel="Value", title="Labeled")
        data_unlabeled = [df_unlabeled[sensor].dropna().values]
        _draw_violin_box(ax_unlab, data_unlabeled, [0], ["steelblue"], ["unlabeled"])
        ax_unlab.set(xlabel="", ylabel="Value", title="Unlabeled")
        plt.tight_layout()
        plt.show()


def plot_3d_multiangle(
    X: np.ndarray,
    labels: np.ndarray,
    label_color_map: dict,
    axis_labels: tuple[str, str, str],
    title: str,
) -> None:
    unique_labels = sorted(set(labels))
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=14)
    for idx, (elev, azim) in enumerate(ANGLES):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        for l in unique_labels:
            mask = labels == l
            label = f"Class {int(l)}" if l != -1 else "Unlabeled"
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                       label=label, s=30, alpha=0.85, color=label_color_map[l])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(axis_labels[0], fontsize=7)
        ax.set_ylabel(axis_labels[1], fontsize=7)
        ax.set_zlabel(axis_labels[2], fontsize=7)
        ax.set_title(f"elev={elev}°, azim={azim}°", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


def plot_2d_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    label_color_map: dict,
    axis_labels: tuple[str, str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for l in sorted(set(labels)):
        mask = labels == l
        label = f"Class {int(l)}" if l != -1 else "Unlabeled"
        ax.scatter(X[mask, 0], X[mask, 1], label=label, s=20, alpha=0.7,
                   color=label_color_map[l])
    ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title=title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_ssl_decision_boundaries(
    X: np.ndarray,
    y: np.ndarray,
    label_color_map: dict,
    hyper_params: list[tuple[dict, dict]],
) -> None:
    unique_labels = sorted(set(y))
    region_colors = [label_color_map[l] for l in unique_labels if l != -1]
    handles = [
        mpatches.Patch(facecolor=label_color_map[i], edgecolor="black", label=i)
        for i in unique_labels
    ]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="black", label="Unlabeled"))

    for hp_ls, hp_svc in hyper_params:
        ls_clf = LabelSpreading(**hp_ls).fit(X, y)
        svc_clf = SelfTrainingClassifier(SVC(probability=True, random_state=42, **hp_svc)).fit(X, y)
        classifiers = [(ls_clf, "LabelSpreading"), (svc_clf, "SVC")]

        fig, axes = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(14, 6))
        point_colors = [label_color_map[label] for label in y]
        for ax, (clf, title) in zip(axes, classifiers):
            DecisionBoundaryDisplay.from_estimator(
                clf, X, response_method="predict_proba",
                plot_method="contourf", ax=ax, multiclass_colors=region_colors,
            )
            ax.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor="black")
            ax.set_title(title)
        fig.suptitle(f"Semi-supervised boundaries (LS: {hp_ls}, SVC: {hp_svc})", y=1)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
        plt.show()

# %% [markdown]
# ## Sensor overview
# %%
# plot_sensor_overview(df, sensor_cols)
plot_sensor_detail(df, sensor_cols)
# %% [markdown]
# ## Label distribution & per-sensor distributions
# %%
plot_label_pie(df_labeled)
plot_violin_box_by_sensor(df_labeled, df_unlabeled, FOCUS_SENSORS)
# %% [markdown]
# ## Dimensionality reduction — labeled data
# %%
X_lab = df_labeled[sensor_cols].values
labels_lab = df_labeled["Label"].values
unique_lab, color_map_lab = label_colors(labels_lab)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_lab)
explained = pca.explained_variance_ratio_ * 100
plot_3d_multiangle(
    X_pca, labels_lab, color_map_lab,
    (f"PC1 ({explained[0]:.1f}%)", f"PC2 ({explained[1]:.1f}%)", f"PC3 ({explained[2]:.1f}%)"),
    "PCA — 3D projection (labeled data)",
)

## Do not use t-SNE - too few datapoints
# from sklearn.manifold import TSNE
# X_tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(X_lab) - 1)).fit_transform(X_lab)
# plot_3d_multiangle(
#     X_tsne, labels_lab, color_map_lab,
#     ("t-SNE 1", "t-SNE 2", "t-SNE 3"),
#     "t-SNE — 3D projection (labeled data)",
# )
# %% [markdown]
# ## Dimensionality reduction — unlabeled data (PCA)
# %%
X_unlab = df_unlabeled[sensor_cols].values
pca_unlab = PCA(n_components=3)
X_pca_unlab = pca_unlab.fit_transform(X_unlab)
explained_unlab = pca_unlab.explained_variance_ratio_ * 100

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca_unlab[:, 0], X_pca_unlab[:, 1], X_pca_unlab[:, 2], s=50, alpha=0.85, color="steelblue")
ax.set(
    xlabel=f"PC1 ({explained_unlab[0]:.1f}%)",
    ylabel=f"PC2 ({explained_unlab[1]:.1f}%)",
    zlabel=f"PC3 ({explained_unlab[2]:.1f}%)",
    title="PCA — 3D projection (unlabeled data)",
)
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Feature space: Sensor 2 / 9 / 13 — 3D scatter
# %%
X_3s = df_labeled[FOCUS_SENSORS].values
unique_3s, color_map_3s = label_colors(df_labeled["Label"].values)
plot_3d_multiangle(
    X_3s, df_labeled["Label"].values, color_map_3s,
    ("Sensor 2", "Sensor 9", "Sensor 13"),
    "Sensor 2 / 9 / 13 — 3D scatter (labeled data)",
)

## Review how the data looks like for the three sensors when projected on them
from itertools import combinations

focus_sensors = ['Sensor 2', 'Sensor 9', 'Sensor 13']
pairs = list(combinations(focus_sensors, 2))

labels = df_labeled['Label'].values
unique_labels = sorted(set(labels))
colors = cm.tab10(np.linspace(0, 0.4, len(unique_labels)))
label_color_map = {l: c for l, c in zip(unique_labels, colors)}

fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
fig.suptitle("Pairwise sensor scatter — labeled (by class) + unlabeled (grey)", fontsize=13)

for ax, (sx, sy) in zip(axes, pairs):
    # unlabeled underneath
    ax.scatter(df_unlabeled[sx], df_unlabeled[sy],
               color='lightgrey', edgecolor='none', s=15, alpha=0.5, label='Unlabeled', zorder=1)
    # labeled on top, colored by class
    for l in unique_labels:
        mask = labels == l
        ax.scatter(df_labeled.loc[mask, sx], df_labeled.loc[mask, sy],
                   color=label_color_map[l], edgecolor='black', linewidths=0.3,
                   s=25, alpha=0.85, label=f'Class {int(l)}', zorder=2)
    ax.set_xlabel(sx)
    ax.set_ylabel(sy)
    ax.set_title(f"{sx} vs {sy}")

axes[0].legend(fontsize=8, markerscale=1.2)
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Feature engineering: add radius dimension
# %%
X_r = add_radius_feature(df_labeled, FOCUS_SENSORS)[:, [0, 1, 3]]  # Sensor 2, 9, R
unique_r, color_map_r = label_colors(df_labeled["Label"].values)
plot_3d_multiangle(
    X_r, df_labeled["Label"].values, color_map_r,
    ("Sensor 2", "Sensor 9", "R"),
    "Sensor 2 / 9 / R — 3D scatter (labeled data)",
)
# %% [markdown]
# ## Sensor 9 vs R — 2D scatter
# %%
X_2d_lab = add_radius_feature(df_labeled, FOCUS_SENSORS)[:, [1, 3]]  # Sensor 9, R
unique_2d, color_map_2d = label_colors(df_labeled["Label"].values)
plot_2d_scatter(
    X_2d_lab, df_labeled["Label"].values, color_map_2d,
    ("Sensor 9", "R"),
    "Sensor 9 vs R — 2D scatter (labeled data)",
)

X_2d_unlab = add_radius_feature(df_unlabeled, FOCUS_SENSORS)[:, [1, 3]]
unlabeled_color_map = {-1: "steelblue"}
plot_2d_scatter(
    X_2d_unlab, np.full(len(X_2d_unlab), -1), unlabeled_color_map,
    ("Sensor 9", "R"),
    "Sensor 9 vs R — 2D scatter (unlabeled data)",
)
# %% [markdown]
# ## Semi-supervised classification — decision boundaries
# %%
X_ssl = add_radius_feature(df, FOCUS_SENSORS)[:, [1, 3]]  # Sensor 9, R
df["label_ssl"] = df["Label"].fillna(-1)
y_ssl = df["label_ssl"].values

unique_ssl = sorted(set(y_ssl))
unique_ssl_labeled = [l for l in unique_ssl if l != -1]
colors_ssl = cm.tab10(np.linspace(0, 0.4, len(unique_ssl_labeled)))
color_map_ssl = {l: c for l, c in zip(unique_ssl_labeled, colors_ssl)}
color_map_ssl[-1] = (1, 1, 1)

hyper_params = [
    ({"gamma": 10},  {"gamma": 0.2}),
    ({"gamma": 20},  {"gamma": 0.5}),
    ({"gamma": 25},  {"gamma": 0.7}),
    ({"gamma": 70},  {"gamma": 1.2}),
]
plot_ssl_decision_boundaries(X_ssl, y_ssl, color_map_ssl, hyper_params)
df.drop(columns=["label_ssl"], inplace=True)
# %% [markdown]
# ## Final predictions
# %%
X_final = add_radius_feature(df, FOCUS_SENSORS)[:, [1, 3]]  # Sensor 9, R
df["label_ssl"] = df["Label"].fillna(-1)
y_final = df["label_ssl"].values

ssl_clf = SelfTrainingClassifier(SVC(probability=True, random_state=42, gamma=0.7)).fit(X_final, y_final)
y_proba = ssl_clf.predict_proba(X_final)

df = pd.concat([
    df,
    pd.DataFrame(y_proba, columns=[f"Probability_label_{c}" for c in ssl_clf.classes_]),
    pd.DataFrame(ssl_clf.classes_[y_proba.argmax(axis=1)], columns=["Predicted_Label"]),
], axis=1)
df.drop(columns=["label_ssl"], inplace=True)
# %%
assert len(df[(df["Label"] != df["Predicted_Label"]) & (~df["Label"].isna())]) == 0, \
    "Predicted label does not match annotated label"