#!/usr/bin/env python3
"""
uv run python 14_patient_profiling_new.py       --csv_good /home/julieletallec/test/figures_features_M1_10_20_burst_new_good_patients_10s/mean_change_metrics_per_seizure.csv     --csv_bad /home/julieletallec/test/figures_features_M1_10_20_burst_new_bad_patients_10s/mean_change_metrics_per_seizure.csv     --cohort     --aggregate sum     --bce_good /home/julieletallec/test/figures_grid_search_kwta_20_10_burst_nozerovalF1_specialFP_new0.1_BCE/best_F1_top10pct/timewise_onset_bce_from_npz_middle_seq_only.csv     --bce_bad /home/julieletallec/test/M2_on_M1_outputs/figures_m2_on_bad_pat_bce/timewise_onset_bce_from_npz.csv     --bce_plot     --bce_agg median      --f1_good /home/julieletallec/test/figures_grid_search_kwta_20_10_burst_nozerovalF1_specialFP_new0.1_BCE/best_F1_top10pct/best_model_per_patient_selected_by_F1.csv     --f1_bad /home/julieletallec/test/M2_on_M1_outputs/figures_m2_on_bad_pat_bce/patient_metrics_from_preds_ranked.csv     --f1_plot     --f1_agg max     --noshow     --feature_subscores_plot     --stability_alpha 5     --keep_top_patients_per_feature 5      --multiply_by_n_selected     --amp_alpha 1.2 --umap2_plot --umap_neighbors 10  --umap2_min_dist 0.015  --umap2_metric euclidean



Objectif
- Même script qu’avant, mais possibilité de passer 2 CSV:
  * un CSV "good outcome"
  * un CSV "bad outcome"

Figures patient:
- Ligne 1 : courbes change_* par seizure (par feature)
- Ligne 2 : scores "final_score" (log1p(amplitude) × gating(consistency))
- Ligne 3 : score "movement_consistency" = log1p(1/(std(|y|)+eps))

Figure cohorte (FINAL demandé) :
- Barplots : FINAL SCORE uniquement, pour:
  * change_global_abs
  * change_soz_amp_abs
  (patients triés par patient_score__change_soz_amp_abs en ordre croissant)
- En dessous: 2 boxplots (un par change_* ci-dessus),
  avec 2 boîtes (good vs bad), + points individuels
  + test statistique (Mann–Whitney U) + p-value.
- Couleurs barplots: vert=good, rouge=bad, gris=unknown.
- Les patients "unknown/conflict" sont affichés dans les barplots (gris) mais EXCLUS des boxplots & tests.
"""

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CHANGE_COLS = [
    "change_global_abs",
    "change_soz_abs",
    "change_nonsoz_abs",
    "change_soz_amp_abs",
]

# Couleurs outcome (barres cohorte)
OUTCOME_COLORS = {
    "good": "tab:green",
    "bad": "tab:red",
    "unknown": "tab:gray",
}


SOZ_FRONTAL = {
    "CHUM::Patient_11",        # Frontal (R)
    "CHUM::Patient_17",        # Frontal (R)
    "CHUM::Patient_22",        # Frontal (R)
    "ds004100::sub-HUP112",    # Frontal
    "ds004100::sub-HUP171",    # Frontal
    "ds004100::sub-HUP172",    # Frontal
    "ds004100::sub-HUP180",    # Frontal
    "ds004100::sub-HUP188",    # Frontal
}


SOZ_TEMPORAL = {
    "ds004100::sub-HUP074",    # Temporal
    "ds004100::sub-HUP082",    # Temporal
    "ds004100::sub-HUP089",    # Temporal
    "ds004100::sub-HUP097",    # Temporal
    "ds004100::sub-HUP107",    # Temporal
    "ds004100::sub-HUP111",    # Temporal
    "ds004100::sub-HUP144",    # Temporal
    "ds004100::sub-HUP148",    # Temporal
    "ds004100::sub-HUP173",    # Temporal
    "ds004100::sub-HUP181",    # Temporal
    "ds004100::sub-HUP080", 
}


SOZ_MEDIAL_TEMPORAL = {
    "ds004100::sub-HUP141",    # Medial-temporal
    "ds004100::sub-HUP157",    # Medial-temporal
    "ds004100::sub-HUP185",    # Medial-temporal
    "ds004100::sub-HUP114",    # Medial-temporal
    "ds004100::sub-HUP133",    # Medial-temporal
    "ds004100::sub-HUP138",    # Medial-temporal
    "ds004100::sub-HUP151",    # Medial-temporal
    "ds004100::sub-HUP162",    # Medial-temporal
    "ds004100::sub-HUP187",    # Medial-temporal
    
}


SOZ_INSULAR = {
    "ds004100::sub-HUP150",    # Insular
}

SOZ_PARIETAL = {
    "CHUM::Patient_01",        # Parietal (R)
}


SOZ_MIXED = {
    "CHUM::Patient_02",        # Fronto-parietal (R)
    "CHUM::Patient_07",        # Fronto-temporo-insular (L)
    "CHUM::Patient_09",        # Insular-Opercular (L)
    "CHUM::Patient_14",        # Fronto-parietal (L)
    "CHUM::Patient_16",        # Temporo-insular (R) + generalized
    "CHUM::Patient_21",        # Fronto-insular (L) 
}



# Colonnes cohorte demandées (seulement 2 change_*)
COHORT_CHANGE_COLS = ["change_global_abs", "change_soz_amp_abs"]


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# SOZ GROUP helpers (from your sets)
# ---------------------------

SOZ_COLORS = {
    "temporal": "tab:blue",
    "frontal": "tab:orange",
    "insular": "tab:purple",
    "parietal": "tab:brown",
    "medial_temporal": "tab:pink",
    "mixed": "gray"
}

def normalize_patient_soz_list_id(pid: str) -> str:
    """
    Convertit:
      CHUM::Patient_01        -> Patient_01
      ds004100::sub-HUP074    -> HUP074
    puis applique normalize_patient_id().
    """
    if pid is None:
        return ""
    s = str(pid).strip()
    if "::" in s:
        s = s.split("::", 1)[1]
    return normalize_patient_id(s)

def build_soz_group_map() -> dict:
    """
    Retourne dict patient_key -> soz_group
    e.g. "Patient_01" -> "temporal", "HUP074" -> "temporal", ...
    """
    m = {}
    for pid in SOZ_TEMPORAL:
        m[normalize_patient_soz_list_id(pid)] = "temporal"
    for pid in SOZ_FRONTAL:
        m[normalize_patient_soz_list_id(pid)] = "frontal"
    for pid in SOZ_INSULAR:
        m[normalize_patient_soz_list_id(pid)] = "insular"
    for pid in SOZ_MIXED:
        m[normalize_patient_soz_list_id(pid)] = "mixed"
    for pid in SOZ_MEDIAL_TEMPORAL:
        m[normalize_patient_soz_list_id(pid)] = "medial_temporal"
    for pid in SOZ_PARIETAL:
        m[normalize_patient_soz_list_id(pid)] = "parietal"

        


    return m


# ---------------------------
# Build patient x feature matrix from your final_score per feature
# ---------------------------

def build_patient_feature_matrix_from_final_scores(
    df: pd.DataFrame,
    y_col: str = "change_soz_amp_abs",
    n_features: int = 15,
    feature_select: str = "most_common",  # "most_common" or "top_global_score"
    impute: str = "zero",  # "zero" or "median"
    standardize: bool = True,
):
    """
    Returns:
      mat_df: DataFrame patient x chosen_features (values=final_score)
      meta_df: DataFrame with patient, patient_key, outcome, soz_group
      chosen_features: list[str]
      X: numpy array ready for embedding
    """
    scores_long = build_feature_scores_table_all_patients(
        df,
        y_col=y_col,
        eps=1e-12,
        zero_rel=0.05,
        zero_abs_thr=1e-8,
        stability_alpha=1.5,
        amp_alpha=0.7,
    )
    if scores_long.empty:
        return None, None, [], None

    tmp = scores_long.copy()
    tmp["final_score"] = pd.to_numeric(tmp["final_score"], errors="coerce")

    if feature_select == "most_common":
        feat_order = (
            tmp.dropna(subset=["final_score"])
               .groupby("feature")["patient"]
               .nunique()
               .sort_values(ascending=False)
               .index.tolist()
        )
    elif feature_select == "top_global_score":
        feat_order = (
            tmp.dropna(subset=["final_score"])
               .groupby("feature")["final_score"]
               .median()
               .sort_values(ascending=False)
               .index.tolist()
        )
    else:
        raise ValueError("feature_select invalide (most_common | top_global_score)")

    chosen = feat_order[:int(n_features)]
    if len(chosen) < 2:
        return None, None, chosen, None

    mat_df = (
        tmp[tmp["feature"].isin(chosen)]
        .pivot_table(index="patient", columns="feature", values="final_score", aggfunc="median")
        .reindex(columns=chosen)
    )

    # outcome per patient
    outcome_map = (
        tmp.groupby("patient")["outcome"]
        .agg(lambda x: x.iloc[0] if x.nunique() == 1 else "unknown")
        .to_dict()
    )
    patient_series = mat_df.index.to_series().astype(str)
    patient_key = patient_series.map(normalize_patient_id)
    outcome = patient_series.map(lambda p: outcome_map.get(p, "unknown"))

    # SOZ group
    soz_map = build_soz_group_map()
    soz_group = patient_key.map(lambda k: soz_map.get(k, "unknown"))

    meta_df = pd.DataFrame({
        "patient": patient_series.to_numpy(),
        "patient_key": patient_key.to_numpy(),
        "outcome": outcome.to_numpy(),
        "soz_group": soz_group.to_numpy(),
    })

    # impute -> X
    X = mat_df.to_numpy(dtype=float)

    if impute == "zero":
        X = np.nan_to_num(X, nan=0.0)
    elif impute == "median":
        col_meds = np.nanmedian(X, axis=0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_meds, inds[1])
    else:
        raise ValueError("impute invalide (zero | median)")

    if standardize:
        try:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
        except Exception:
            print("[WARN] sklearn indisponible -> pas de standardization.")

    return mat_df, meta_df, chosen, X


# ---------------------------
# Compute UMAP (fallback PCA) once
# ---------------------------

def compute_2d_embedding(
    X: np.ndarray,
    method: str = "umap",  # "umap" or "pca"
    umap_neighbors: int = 20,
    umap_min_dist: float = 0.15,
    umap_metric: str = "cosine",
    random_state: int = 0,
):
    if X is None or X.size == 0:
        return None, "none"

    if method == "pca":
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
        return emb, "pca"

    # default: try UMAP, fallback PCA
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            random_state=int(random_state),
        )
        emb = reducer.fit_transform(X)
        return emb, "umap"
    except Exception as e:
        print(f"[WARN] UMAP indisponible ({e}) -> fallback PCA.")
        try:
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
            return emb, "pca"
        except Exception as ee:
            print(f"[ERROR] PCA indisponible aussi ({ee}).")
            return None, "none"


# ---------------------------
# Plot 1 figure = 2 panels:
#   left: colored by SOZ implant group
#   right: colored by F1 value
# ---------------------------

def plot_umap_soz_and_f1_side_by_side(
    df: pd.DataFrame,
    outdir: str,
    # build matrix params
    y_col: str = "change_soz_amp_abs",
    n_features: int = 15,
    feature_select: str = "most_common",  # "most_common" or "top_global_score"
    impute: str = "zero",  # "zero" or "median"
    standardize: bool = True,
    # embedding params
    umap_neighbors: int = 20,
    umap_min_dist: float = 0.15,
    umap_metric: str = "cosine",
    # F1 inputs (re-use your functions)
    f1_good: str | None = None,
    f1_bad: str | None = None,
    f1_agg: str = "max",
    # plot params
    annotate: bool = False,
    noshow: bool = False,
    # gradient params
    grad_gridsize: int = 260,
    grad_sigma: float = 0.5,
    grad_padding: float = 0.07,
    show_support_points_on_gradient: bool = True,
):
    """
    3 panneaux (mêmes coords 2D):
      1) color = SOZ group
      2) color = F1 points
      3) color = F1 gradient (heatmap lissée) + overlay points

    Sauvegarde:
      - PNG: umap_3panel_soz_f1_scatter_f1_gradient__{y_col}.png
      - CSV: umap_3panel_soz_f1__{y_col}.csv
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- Build matrix patient x feature + meta
    mat_df, meta_df, chosen_feats, X = build_patient_feature_matrix_from_final_scores(
        df=df,
        y_col=y_col,
        n_features=n_features,
        feature_select=feature_select,
        impute=impute,
        standardize=standardize,
    )
    if mat_df is None or X is None or len(chosen_feats) < 2:
        print("[WARN] Impossible de construire une matrice patient×feature pour UMAP.")
        return None

    # ---- Compute embedding once
    emb, method_used = compute_2d_embedding(
        X,
        method="umap",
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        random_state=0,
    )
    if emb is None:
        print("[WARN] Embedding 2D impossible.")
        return None

    # ---- Load + aggregate F1 per patient_key (optional)
    if (f1_good is not None) or (f1_bad is not None):
        f1_raw = load_two_f1_csvs(f1_good, f1_bad)              # your function
        f1_pat = aggregate_f1_per_patient(f1_raw, agg=f1_agg)   # your function
        f1_df = f1_pat[["patient_key", "f1_value"]].copy()
    else:
        f1_df = pd.DataFrame(columns=["patient_key", "f1_value"])

    # ---- Merge everything for plotting
    plot_df = meta_df.copy()
    plot_df["x"] = emb[:, 0]
    plot_df["y"] = emb[:, 1]
    plot_df = plot_df.merge(f1_df, on="patient_key", how="left")
    plot_df["f1_value"] = pd.to_numeric(plot_df["f1_value"], errors="coerce")


    # >>> AJOUT: print patients unknown dans le terminal
    unknown_patients = (
        plot_df.loc[plot_df["soz_group"] == "unknown", ["patient", "patient_key"]]
        .drop_duplicates()
        .sort_values("patient")
    )
    print(f"[UMAP] unknown/conflict patients: {len(unknown_patients)}")
    if len(unknown_patients):
        print(unknown_patients.to_string(index=False))


    emb_2d = plot_df[["x","y"]].to_numpy(float)
    metrics = soz_separation_metrics_from_embedding(emb_2d, plot_df["soz_group"].to_numpy(), k=5, drop_unknown=True)
    print(f"[SOZ separation] n={metrics['n']} groups={metrics['n_groups']} "
           f"silhouette={metrics['silhouette']:.3f}  knn_purity@5={metrics['knn_purity']:.3f}")



    # ---- Figure with 3 panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "UMAP embedding of patient-level preictal-to-ictal feature consistency profiles (M1 features)\n",
        #f"y_col={y_col} | features={len(chosen_feats)} ({feature_select}) | "
        #f"impute={impute} | standardize={standardize} | "
        #f"method={method_used} metric={umap_metric} neighbors={umap_neighbors} min_dist={umap_min_dist}",
        y=0.9, fontsize = 16
    )

    # ==========================================================
    # Panel 1: SOZ group colors
    # ==========================================================
    ax0 = axes[0]
    ax0.set_title("Color = SOZ implant location")

    for g in ["temporal",  "medial_temporal", "frontal", "insular", "parietal", "mixed"]:
        sub = plot_df[plot_df["soz_group"] == g]
        if sub.empty:
            continue
        ax0.scatter(
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            s=90,
            alpha=0.85,
            color=SOZ_COLORS.get(g, "tab:gray"),
            edgecolors="none",
            label=g,
        )

    if annotate:
        for _, r in plot_df.iterrows():
            ax0.text(float(r["x"]), float(r["y"]), f" {r['patient']}", fontsize=7, alpha=0.85)

    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False)
    ax0.set_xlabel("UMAP dimension 1")
    ax0.set_ylabel("UMAP dimension 2")
    
    

    # ==========================================================
    # Panel 2: F1 scatter (continuous)
    # ==========================================================



    finite = np.isfinite(plot_df["f1_value"].to_numpy(dtype=float))
    sub_finite = plot_df[finite]
    sub_missing = plot_df[~finite]

 

    # ==========================================================
    # Panel 3: F1 gradient (smoothed field)
    # ==========================================================
    ax1 = axes[1]
    ax1.set_title("Model 2 performance (F1@top 10%, smoothed)")
    ax1.set_xlabel("UMAP dimension 1")
    ax1.set_ylabel("UMAP dimension 2")

    # Same finite selection
    if sub_finite.shape[0] >= 3:
        Xg, Yg, Z = smooth_value_field_on_grid(
            sub_finite["x"].to_numpy(float),
            sub_finite["y"].to_numpy(float),
            sub_finite["f1_value"].to_numpy(float),
            gridsize=int(grad_gridsize),
            sigma=float(grad_sigma),
            padding=float(grad_padding),
        )
        if Z is not None:
            im = ax1.imshow(
                Z,
                origin="lower",
                extent=[
                    float(np.min(Xg)), float(np.max(Xg)),
                    float(np.min(Yg)), float(np.max(Yg)),
                ],
                cmap="viridis",
                aspect="auto",
                alpha=0.95,
            )
            cb2 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cb2.set_label("F1@top_10pct")
    else:
        ax1.text(
            0.5, 0.5,
            "Not enough F1 points for gradient",
            transform=ax1.transAxes,
            ha="center", va="center",
            alpha=0.85,
        )

    # overlay support points
    if show_support_points_on_gradient:
        ax1.scatter(
            plot_df["x"].to_numpy(),
            plot_df["y"].to_numpy(),
            s=90,
            alpha=0.35,
            color="k",
            edgecolors="none",
        )



    if annotate:
        for _, r in plot_df.iterrows():
            ax1.text(float(r["x"]), float(r["y"]), f" {r['patient']}", fontsize=7, alpha=0.85)

    ax1.grid(True, alpha=0.25)

    # ---- Save
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpng = os.path.join(outdir, f"umap_3panel_soz_f1_scatter_f1_gradient__{y_col}.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure 3 panneaux sauvegardée → {outpng}")

    outcsv = os.path.join(outdir, f"umap_3panel_soz_f1__{y_col}.csv")
    plot_df.to_csv(outcsv, index=False)
    print(f"[OK] CSV embedding+meta sauvegardé → {outcsv}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return plot_df

def plot_patient_umap_from_feature_scores(
    df: pd.DataFrame,
    outdir: str,
    y_col: str = "change_soz_amp_abs",
    n_features: int = 15,
    feature_select: str = "most_common",  # "most_common" or "top_global_score"
    impute: str = "zero",  # "zero" or "median"
    standardize: bool = True,
    umap_neighbors: int = 20,
    umap_min_dist: float = 0.15,
    umap_metric: str = "cosine",
    noshow: bool = False,
    annotate: bool = False,
):
    os.makedirs(outdir, exist_ok=True)

    # 1) long scores: patient, outcome, feature, final_score
    scores_long = build_feature_scores_table_all_patients(
        df,
        y_col=y_col,
        eps=1e-12,
        zero_rel=0.05,
        zero_abs_thr=1e-8,
        stability_alpha=1.5,
        amp_alpha=0.7,
    )
    if scores_long.empty:
        print("[WARN] scores_long vide, UMAP impossible.")
        return None

    # 2) choisir un set de features global (taille n_features)
    tmp = scores_long.copy()
    tmp["final_score"] = pd.to_numeric(tmp["final_score"], errors="coerce")

    if feature_select == "most_common":
        feat_order = (
            tmp.dropna(subset=["final_score"])
               .groupby("feature")["patient"]
               .nunique()
               .sort_values(ascending=False)
               .index.tolist()
        )
    elif feature_select == "top_global_score":
        feat_order = (
            tmp.dropna(subset=["final_score"])
               .groupby("feature")["final_score"]
               .median()
               .sort_values(ascending=False)
               .index.tolist()
        )
    else:
        raise ValueError("feature_select invalide")

    chosen = feat_order[:int(n_features)]
    if len(chosen) < 2:
        print("[WARN] Pas assez de features pour UMAP.")
        return None

    # 3) pivot patient x feature
    mat = (
        tmp[tmp["feature"].isin(chosen)]
        .pivot_table(index="patient", columns="feature", values="final_score", aggfunc="median")
        .reindex(columns=chosen)
    )

    # outcome par patient
    outcome_map = (
        tmp.groupby("patient")["outcome"]
        .agg(lambda x: x.iloc[0] if x.nunique() == 1 else "unknown")
        .to_dict()
    )
    outcomes = mat.index.to_series().map(lambda p: outcome_map.get(p, "unknown"))

    # 4) imputation
    X = mat.to_numpy(dtype=float)
    if impute == "zero":
        X = np.nan_to_num(X, nan=0.0)
    elif impute == "median":
        col_meds = np.nanmedian(X, axis=0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_meds, inds[1])
    else:
        raise ValueError("impute invalide")

    # 5) standardize
    if standardize:
        try:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
        except Exception:
            print("[WARN] sklearn non dispo -> pas de standardization.")

    # 6) UMAP (ou fallback PCA si umap-learn absent)
    emb = None
    method_used = None
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            random_state=0,
        )
        emb = reducer.fit_transform(X)
        method_used = "umap"
    except Exception as e:
        print(f"[WARN] UMAP indisponible ({e}) -> fallback PCA.")
        try:
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2, random_state=0).fit_transform(X)
            method_used = "pca"
        except Exception as ee:
            print(f"[ERROR] PCA aussi indisponible ({ee}).")
            return None

    # 7) plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(
        f"Patient embedding 2D ({method_used}) — y_col={y_col}\n"
        f"features={len(chosen)} select={feature_select} impute={impute} standardize={standardize} "
        f"metric={umap_metric} neighbors={umap_neighbors} min_dist={umap_min_dist}"
    )

    for out in ["good", "bad", "unknown"]:
        m = outcomes.to_numpy() == out
        if not np.any(m):
            continue
        ax.scatter(
            emb[m, 0], emb[m, 1],
            s=70, alpha=0.85,
            color=OUTCOME_COLORS.get(out, "tab:gray"),
            label=out,
            edgecolors="none",
        )

    if annotate:
        for i, p in enumerate(mat.index.astype(str).tolist()):
            ax.text(emb[i, 0], emb[i, 1], f" {p}", fontsize=7, alpha=0.85)

    ax.grid(True, alpha=0.25)
    ax.legend()

    outpng = os.path.join(outdir, f"patient_embedding_{method_used}_{y_col}.png")
    fig.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Embedding sauvegardé → {outpng}")

    outcsv = os.path.join(outdir, f"patient_embedding_{method_used}_{y_col}.csv")
    out_df = pd.DataFrame({
        "patient": mat.index.astype(str),
        "outcome": outcomes.astype(str).to_numpy(),
        "x": emb[:, 0],
        "y": emb[:, 1],
    })
    out_df.to_csv(outcsv, index=False)
    print(f"[OK] CSV embedding → {outcsv}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return out_df





def normalize_patient_id(pid: str) -> str:
    """
    Normalise les IDs patients pour matcher BCE/F1 <-> cohort.

    Exemples:
      ds004100_CHUM_Patient_07  -> Patient_07
      CHUM_Patient_07           -> Patient_07
      CHUM__Patient_01          -> Patient_01
      sub-HUP130                -> HUP130
      HUP130                    -> HUP130
    """
    if pid is None:
        return ""

    s = str(pid).strip()
    s = s.replace("__", "_")

    for prefix in ["ds004100_", "CHUM_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]

    if s.startswith("sub-"):
        s = s[len("sub-"):]

    return s


def add_patient_key_column(df: pd.DataFrame, patient_col: str = "patient", key_col: str = "patient_key") -> pd.DataFrame:
    out = df.copy()
    out[key_col] = out[patient_col].astype(str).map(normalize_patient_id)
    return out


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["seizure"] = pd.to_numeric(df["seizure"], errors="coerce").astype("Int64")
    for c in CHANGE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["patient"] = df["patient"].astype(str)
    df["feature"] = df["feature"].astype(str)
    return df.sort_values(["patient", "seizure", "feature"]).reset_index(drop=True)


def load_two_csvs(csv_good: str | None, csv_bad: str | None, csv_single: str | None) -> pd.DataFrame:
    """Retourne un DF unique avec une colonne 'outcome' ∈ {good, bad, unknown}."""
    if csv_good or csv_bad:
        dfs = []
        if csv_good:
            dfg = load_csv(csv_good)
            dfg["outcome"] = "good"
            dfs.append(dfg)
        if csv_bad:
            dfb = load_csv(csv_bad)
            dfb["outcome"] = "bad"
            dfs.append(dfb)

        if not dfs:
            raise ValueError("Tu as demandé le mode 2 CSV mais aucun chemin n'a été fourni.")

        df = pd.concat(dfs, ignore_index=True)

        # conflit -> unknown
        outcome_per_patient = df.groupby("patient")["outcome"].nunique()
        conflicted = outcome_per_patient[outcome_per_patient > 1].index.tolist()
        if conflicted:
            df.loc[df["patient"].isin(conflicted), "outcome"] = "unknown"

        return df.sort_values(["patient", "seizure", "feature"]).reset_index(drop=True)

    if csv_single:
        df = load_csv(csv_single)
        df["outcome"] = "unknown"
        return df

    raise ValueError("Il faut fournir soit --csv, soit (--csv_good et/ou --csv_bad).")


def compute_feature_scores(
    sub: pd.DataFrame,
    y_col: str,
    eps: float = 1e-12,
    # scoring params
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.0,
    amp_alpha: float = 1.0,
    # activity/debug
    active_frac_thr: float = 0.4,
) -> pd.DataFrame:
    """
    Score par feature pour une colonne change_*.

    Colonnes renvoyées:
      - amplitude_score: median(|y|)
      - amp_ref_patient: median(amplitude_score) sur les features du patient
      - amp_norm: amplitude_score / (amp_ref_patient + eps)
      - amp_term: log1p(amp_norm) ** amp_alpha
      - stability: (1/(1+cv)) ** stability_alpha, cv = std(|y|)/mean(|y|)
      - sign_term: abs(mean(sign(y))) en ignorant les points proches de 0
      - movement_consistency: log1p(1/(std(|y|)+eps))
      - final_score: amp_term * stability * sign_term
      - zero_threshold_used, frac_nonzero, is_active, n_points
    """
    if sub.empty:
        return pd.DataFrame()

    # PASS 1: amplitude par feature -> amp_ref patient
    amp_map = {}
    npts_map = {}

    for feat, g in sub.groupby("feature"):
        g = g.sort_values("seizure")
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
        y = y[np.isfinite(y)]
        y_abs = np.abs(y)

        npts_map[feat] = int(y_abs.size)
        if y_abs.size < 3:
            amp_map[feat] = np.nan
        else:
            amp_map[feat] = float(np.median(y_abs))

    amps = np.array([v for v in amp_map.values() if np.isfinite(v)], dtype=float)
    amp_ref = float(np.median(amps)) if amps.size else 0.0

    zero_thr = max(float(zero_abs_thr), float(zero_rel) * float(amp_ref))

    # PASS 2: scores
    rows = []
    for feat, g in sub.groupby("feature"):
        g = g.sort_values("seizure")
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
        y = y[np.isfinite(y)]

        y_abs = np.abs(y)
        n_points = int(y_abs.size)
        amp = float(amp_map.get(feat, np.nan))

        frac_nonzero = float(np.mean(y_abs > zero_thr)) if n_points > 0 else 0.0
        is_active = int(frac_nonzero >= float(active_frac_thr))

        # movement_consistency même si pas assez de points (debug)
        std_abs = float(np.std(y_abs)) if n_points > 0 else float("nan")
        movement_consistency = float(np.log1p(1.0 / (std_abs + eps))) if np.isfinite(std_abs) else float("nan")

        if n_points < 3 or not np.isfinite(amp):
            rows.append(
                {
                    "feature": feat,
                    "n_points": n_points,
                    "amplitude_score": float("nan"),
                    "amp_ref_patient": float(amp_ref),
                    "amp_norm": float("nan"),
                    "amp_term": float("nan"),
                    "stability": float("nan"),
                    "sign_term": float("nan"),
                    "movement_consistency": float(movement_consistency),
                    "final_score": float("nan"),
                    "zero_threshold_used": float(zero_thr),
                    "frac_nonzero": float(frac_nonzero),
                    "is_active": int(is_active),
                }
            )
            continue

        amp_norm = float(amp / (amp_ref + eps)) if amp_ref > 0 else 0.0

        amp_term = float(np.log1p(amp_norm))
        if amp_alpha is not None and float(amp_alpha) != 1.0:
            amp_term = float(amp_term ** float(amp_alpha))

        mean_abs = float(np.mean(y_abs))
        std_abs = float(np.std(y_abs))
        cv = float(std_abs / (mean_abs + eps))
        stability = float(1.0 / (1.0 + cv))
        if stability_alpha is not None and float(stability_alpha) != 1.0:
            stability = float(stability ** float(stability_alpha))

        y_nz = y[y_abs > zero_thr]
        if y_nz.size == 0:
            sign_term = 0.0
        else:
            sign_term = float(abs(np.mean(np.sign(y_nz))))

        movement_consistency = float(np.log1p(1.0 / (std_abs + eps)))
        final = float(amp_term * stability * sign_term)

        rows.append(
            {
                "feature": feat,
                "n_points": n_points,
                "amplitude_score": float(amp),
                "amp_ref_patient": float(amp_ref),
                "amp_norm": float(amp_norm),
                "amp_term": float(amp_term),
                "stability": float(stability),
                "sign_term": float(sign_term),
                "movement_consistency": float(movement_consistency),
                "final_score": float(final),
                "zero_threshold_used": float(zero_thr),
                "frac_nonzero": float(frac_nonzero),
                "is_active": int(is_active),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)


def build_feature_scores_table_all_patients(
    df: pd.DataFrame,
    y_col: str,
    eps: float = 1e-12,
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.0,
    amp_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Table LONGUE:
      patient, outcome, feature, amp_norm, amp_term, stability, sign_term, final_score
    """
    rows = []
    for patient, subp in df.groupby("patient"):
        out = compute_feature_scores(
            subp,
            y_col=y_col,
            eps=eps,
            zero_rel=zero_rel,
            zero_abs_thr=zero_abs_thr,
            stability_alpha=stability_alpha,
            amp_alpha=amp_alpha,
        ).copy()

        if out.empty:
            continue

        out["patient"] = str(patient)
        outcome = subp["outcome"].iloc[0] if subp["outcome"].nunique() == 1 else "unknown"
        out["outcome"] = outcome

        keep_cols = [
            "patient", "outcome", "feature",
            "amp_norm", "amp_term",
            "stability",
            "sign_term",
            "final_score",
        ]

        for c in keep_cols:
            if c not in out.columns:
                out[c] = np.nan

        rows.append(out[keep_cols])

    if not rows:
        return pd.DataFrame(columns=[
            "patient","outcome","feature","amp_norm","amp_term","stability","sign_term","final_score"
        ])

    return pd.concat(rows, ignore_index=True)


def build_feature_color_map(features):
    feats = list(features)
    cmap = plt.get_cmap("tab20")
    return {f: cmap(i % cmap.N) for i, f in enumerate(feats)}


def plot_patient(
    df: pd.DataFrame,
    patient_name: str,
    outdir: str,
    noshow: bool = False,
    topn: int = 12,
    # hyperparams scoring (uniforme)
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.5,
    amp_alpha: float = 0.7,
):
    os.makedirs(outdir, exist_ok=True)

    sub = df[df["patient"] == patient_name].copy()
    if sub.empty:
        raise ValueError(f"Aucune donnée pour patient='{patient_name}'")

    sub["seizure"] = pd.to_numeric(sub["seizure"], errors="coerce")
    sub = sub.dropna(subset=["seizure"]).sort_values("seizure")

    features = sorted(sub["feature"].dropna().unique())
    color_map = build_feature_color_map(features)

    fig, axes = plt.subplots(
        3, 4, figsize=(26, 14),
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )
    fig.suptitle(f"{patient_name} — change_* par seizure + scores", y=1.02)

    xmin = int(sub["seizure"].min())
    xmax = int(sub["seizure"].max())

    # Ligne 1 : courbes
    for ax, y_col in zip(axes[0], CHANGE_COLS):
        for feat in features:
            s = sub[sub["feature"] == feat][["seizure", y_col]].dropna()
            if not s.empty:
                ax.plot(
                    s["seizure"], s[y_col],
                    marker="o", linewidth=1, markersize=3,
                    color=color_map.get(feat, None), alpha=0.95,
                )
        ax.set_title(y_col)
        ax.set_xlabel("seizure index")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, alpha=0.3)

    # Lignes 2-3 : scores
    all_scores = []
    for col_i, y_col in enumerate(CHANGE_COLS):
        scores = compute_feature_scores(
            sub,
            y_col=y_col,
            eps=1e-12,
            zero_rel=zero_rel,
            zero_abs_thr=zero_abs_thr,
            stability_alpha=stability_alpha,
            amp_alpha=amp_alpha,
        )
        ax_final = axes[1][col_i]
        ax_movc = axes[2][col_i]

        if scores.empty:
            ax_final.set_title(f"Scores — {y_col} (aucune donnée)")
            ax_final.axis("off")
            ax_movc.axis("off")
            continue

        scores_outfile = os.path.join(outdir, f"{patient_name}_scores_{y_col}.csv")
        scores.to_csv(scores_outfile, index=False)
        all_scores.append((y_col, scores_outfile))

        top = scores.head(int(topn)).copy()

        # final_score
        top_final = top.sort_values("final_score", ascending=True)
        bar_colors = [color_map.get(f, None) for f in top_final["feature"].tolist()]
        ax_final.barh(top_final["feature"], top_final["final_score"], height=0.6, alpha=0.9, color=bar_colors)
        xmax_score = float(top_final["final_score"].max())
        ax_final.set_xlim(0, xmax_score * 1.15 if xmax_score > 0 else 1.0)
        ax_final.set_title(f"Top {len(top_final)} — final_score ({y_col})")
        ax_final.set_xlabel("final_score = log1p(amp_norm)^amp_alpha × stability^alpha × sign_term")
        ax_final.grid(True, axis="x", alpha=0.3)
        for i, (_, r) in enumerate(top_final.iterrows()):
            ax_final.text(float(r["final_score"]), i, f" {r['final_score']:.4f}", va="center", ha="left", fontsize=9)

        # movement_consistency
        top_movc = top.sort_values("movement_consistency", ascending=True)
        bar_colors_movc = [color_map.get(f, None) for f in top_movc["feature"].tolist()]
        ax_movc.barh(
            top_movc["feature"], top_movc["movement_consistency"],
            height=0.6, alpha=0.9, color=bar_colors_movc,
        )
        xmax_mc = float(top_movc["movement_consistency"].max())
        ax_movc.set_xlim(0, xmax_mc * 1.15 if xmax_mc > 0 else 1.0)
        ax_movc.set_title(f"Top {len(top_movc)} — movement_consistency ({y_col})")
        ax_movc.set_xlabel("movement_consistency = log1p(1/(std(|y|)+eps))")
        ax_movc.grid(True, axis="x", alpha=0.3)
        for i, (_, r) in enumerate(top_movc.iterrows()):
            ax_movc.text(float(r["movement_consistency"]), i, f" {r['movement_consistency']:.4f}", va="center", ha="left", fontsize=9)

    fig.tight_layout()
    outfile = os.path.join(outdir, f"{patient_name}_change_curves_with_scores.png")
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure patient sauvegardée → {outfile}")

    if all_scores:
        print("[OK] Scores patient (CSV) sauvegardés :")
        for y_col, p in all_scores:
            print(f" - {y_col}: {p}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)


def cohort_patient_scores(
    df: pd.DataFrame,
    aggregate: str = "sum",
    k: int = 15,
    min_seizures: int = 3,
    # scoring params (uniforme)
    zero_abs_thr: float = 1e-8,
    zero_rel: float = 0.05,
    stability_alpha: float = 1.5,
    amp_alpha: float = 0.7,
    # filtre top-N
    keep_top_patients_per_feature: int | None = None,
    # multiplier par nb de sélections (patient-specific)
    multiply_by_n_selected: bool = False,
):
    """
    Construit un tableau patient-level:
      patient, n_seizures, outcome, patient_score__change_*, overall_score

    Si keep_top_patients_per_feature est défini:
      pour chaque feature, on garde uniquement les patients classés dans le top-N
      selon final_score pour cette feature.

    Si multiply_by_n_selected=True:
      score_patient = base_score * n_selected (spécifique au patient).
    """

    def _agg_vals(vals: np.ndarray, mode: str, kk: int) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")

        mode = (mode or "sum").lower()
        if mode == "sum":
            return float(np.sum(vals))
        if mode == "max":
            return float(np.max(vals))
        if mode == "mean":
            return float(np.mean(vals))
        if mode == "topk":
            kk = max(1, int(kk))
            top = np.sort(vals)[-min(kk, vals.size):]
            return float(np.mean(top))

        raise ValueError(f"mode agrégation inconnu: {mode} (sum/max/mean/topk)")

    tmp = df.copy()
    tmp["seizure"] = pd.to_numeric(tmp["seizure"], errors="coerce")
    tmp = tmp.dropna(subset=["seizure"])

    seizure_counts = tmp.groupby("patient")["seizure"].nunique()
    keep_patients = seizure_counts[seizure_counts >= int(min_seizures)].index.tolist()
    tmp = tmp[tmp["patient"].isin(keep_patients)].copy()

    outcome_map = tmp.groupby("patient")["outcome"].agg(
        lambda x: x.iloc[0] if x.nunique() == 1 else "unknown"
    ).to_dict()

    base = (
        pd.DataFrame({"patient": sorted(tmp["patient"].unique().tolist())})
        .assign(
            n_seizures=lambda d: d["patient"].map(lambda p: int(seizure_counts.loc[p])),
            outcome=lambda d: d["patient"].map(lambda p: outcome_map.get(p, "unknown")),
        )
    )

    for y_col in CHANGE_COLS:
        rows = []
        for patient, subp in tmp.groupby("patient"):
            fs = compute_feature_scores(
                subp,
                y_col=y_col,
                eps=1e-12,
                zero_abs_thr=zero_abs_thr,
                zero_rel=zero_rel,
                stability_alpha=stability_alpha,
                amp_alpha=amp_alpha,
            )
            if fs is None or fs.empty:
                continue
            fs2 = fs[["feature", "final_score"]].copy()
            fs2["patient"] = str(patient)
            rows.append(fs2)

        if not rows:
            base[f"patient_score__{y_col}"] = np.nan
            base[f"n_selected__{y_col}"] = 0
            continue

        long = pd.concat(rows, ignore_index=True)
        long["final_score"] = pd.to_numeric(long["final_score"], errors="coerce")
        long = long.dropna(subset=["final_score"]).copy()

        if keep_top_patients_per_feature is not None and int(keep_top_patients_per_feature) > 0:
            N = int(keep_top_patients_per_feature)
            long["_rank"] = long.groupby("feature")["final_score"].rank(method="min", ascending=False)
            long = long[long["_rank"] <= N].copy()
            long = long.drop(columns=["_rank"])

        n_selected = (
            long.groupby("patient")["feature"]
            .size()
            .rename(f"n_selected__{y_col}")
            .reset_index()
        )

        pat_base_score = (
            long.groupby("patient")["final_score"]
            .apply(lambda s: _agg_vals(s.to_numpy(dtype=float), aggregate, k))
            .rename("base_score")
            .reset_index()
        )

        pat_scores = pat_base_score.merge(n_selected, on="patient", how="left")

        if multiply_by_n_selected:
            pat_scores[f"patient_score__{y_col}"] = (
                pat_scores["base_score"] * pat_scores[f"n_selected__{y_col}"].astype(float)
            )
        else:
            pat_scores[f"patient_score__{y_col}"] = pat_scores["base_score"]

        pat_scores = pat_scores[["patient", f"patient_score__{y_col}", f"n_selected__{y_col}"]]
        base = base.merge(pat_scores, on="patient", how="left")

    score_cols = [f"patient_score__{c}" for c in CHANGE_COLS]
    base["overall_score"] = base[score_cols].max(axis=1, skipna=True)

    return base.sort_values("overall_score", ascending=False, na_position="last").reset_index(drop=True)


def _mann_whitney_u_pvalue(x: np.ndarray, y: np.ndarray):
    """
    Mann–Whitney U (two-sided) p-value.
    SciPy si dispo; sinon permutation test approx sur diff de médianes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size < 2 or y.size < 2:
        return np.nan, "insufficient_n"

    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue), "mannwhitneyu_scipy"
    except Exception:
        rng = np.random.default_rng(0)
        obs = float(np.median(x) - np.median(y))
        pooled = np.concatenate([x, y])
        n_x = x.size
        n_perm = 20000

        count = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            xx = pooled[:n_x]
            yy = pooled[n_x:]
            stat = float(np.median(xx) - np.median(yy))
            if abs(stat) >= abs(obs):
                count += 1
        p = (count + 1) / (n_perm + 1)
        return float(p), "perm_test_median_diff"


def plot_cohort(
    df: pd.DataFrame,
    outdir: str,
    aggregate: str = "topk",
    k: int = 15,
    min_seizures: int = 3,
    noshow: bool = False,
    # scoring params (uniforme)
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.5,
    amp_alpha: float = 0.7,
    # filtre top-N
    keep_top_patients_per_feature: int | None = None,
    # multiplier
    multiply_by_n_selected: bool = False,
):
    """
    Version finale:
      - 2 plots CÔTE À CÔTE
      - gauche: barplot FINAL SCORE pour change_global_abs (tous patients)
      - droite: boxplot good vs bad pour change_global_abs (unknown exclus)
    """
    os.makedirs(outdir, exist_ok=True)

    cohort = cohort_patient_scores(
        df,
        aggregate=aggregate,
        k=k,
        min_seizures=min_seizures,
        zero_rel=zero_rel,
        zero_abs_thr=zero_abs_thr,
        stability_alpha=stability_alpha,
        amp_alpha=amp_alpha,
        keep_top_patients_per_feature=keep_top_patients_per_feature,
        multiply_by_n_selected=multiply_by_n_selected,
    )

    score_name_long = "Preictal-to-Ictal Change Consistency Score (Features for M1)"
    score_name_short = "PI→I Change Consistency Score"

    if cohort.empty:
        print("[WARN] Cohorte vide (pas assez de données / patients).")
        return cohort

    cohort_csv = os.path.join(outdir, "cohort_patient_scores.csv")
    cohort.to_csv(cohort_csv, index=False)
    print(f"[OK] CSV cohorte sauvegardé → {cohort_csv}")

    col = "patient_score__change_global_abs"
    title = "change_global_abs"

    # tri des patients (comme avant)
    sort_col = "patient_score__change_soz_amp_abs"
    cohort_sorted = cohort.sort_values(sort_col, ascending=True, na_position="last").reset_index(drop=True)

    def _median_line(ax, vals: np.ndarray, color: str):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        med = float(np.median(vals))
        ax.axvline(med, linestyle="--", linewidth=2.0, color=color, alpha=0.9)

    # ==========================================================
    # Figure: 1 x 2 (côte à côte)
    # ==========================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        "Cohorte — FINAL SCORE (change_global_abs)\n"
        f"aggregate={aggregate}, k={k}, min_seizures={min_seizures}, "
        f"keep_top_patients_per_feature={keep_top_patients_per_feature}, "
        f"multiply_by_n_selected={multiply_by_n_selected}, "
        f"stability_alpha={stability_alpha}, amp_alpha={amp_alpha}",
        y=0.98,
    )

    # ==========================================================
    # LEFT: barplot (tous patients)
    # ==========================================================
    ax0 = axes[0]

    df_plot = cohort_sorted[["patient", "outcome", col]].copy()
    df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
    df_plot = df_plot.dropna(subset=[col]).sort_values(col, ascending=True)

    patients_loc = df_plot["patient"].tolist()
    vals = df_plot[col].to_numpy(dtype=float)

    colors = [
        OUTCOME_COLORS.get(o, OUTCOME_COLORS["unknown"])
        for o in df_plot["outcome"].astype(str).tolist()
    ]
    y = np.arange(len(patients_loc))

    ax0.barh(y, vals, alpha=0.9, color=colors)
    ax0.set_yticks(y)
    ax0.set_yticklabels(patients_loc)
    ax0.invert_yaxis()

    xmax = float(np.max(vals)) if vals.size else 1.0
    ax0.set_xlim(0, xmax * 1.15 if xmax > 0 else 1.0)

    ax0.set_title("Barplot — tous patients (triés)")
    ax0.set_xlabel("patient final_score")
    ax0.grid(True, axis="x", alpha=0.3)

    good_vals = df_plot.loc[df_plot["outcome"] == "good", col].to_numpy(dtype=float)
    bad_vals = df_plot.loc[df_plot["outcome"] == "bad", col].to_numpy(dtype=float)
    _median_line(ax0, good_vals, OUTCOME_COLORS["good"])
    _median_line(ax0, bad_vals, OUTCOME_COLORS["bad"])

    # ==========================================================
    # RIGHT: boxplot (good vs bad)
    # ==========================================================
    ax1 = axes[1]

    cohort_known = cohort[cohort["outcome"].isin(["good", "bad"])].copy()
    rng = np.random.default_rng(0)

    good = pd.to_numeric(
        cohort_known.loc[cohort_known["outcome"] == "good", col],
        errors="coerce",
    ).to_numpy(float)
    bad = pd.to_numeric(
        cohort_known.loc[cohort_known["outcome"] == "bad", col],
        errors="coerce",
    ).to_numpy(float)

    good = good[np.isfinite(good)]
    bad = bad[np.isfinite(bad)]

    bp = ax1.boxplot(
        [good, bad],
        labels=["good", "bad"],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
    )
    if len(bp["boxes"]) >= 2:
        bp["boxes"][0].set_facecolor(OUTCOME_COLORS["good"])
        bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][1].set_facecolor(OUTCOME_COLORS["bad"])
        bp["boxes"][1].set_alpha(0.35)

    def _scatter_points(xpos, data, color):
        if data.size == 0:
            return
        jitter = rng.normal(0, 0.04, size=data.size)
        ax1.scatter(
            np.full(data.size, xpos, dtype=float) + jitter,
            data,
            alpha=0.85,
            s=28,
            color=color,
            edgecolors="none",
        )

    _scatter_points(1, good, OUTCOME_COLORS["good"])
    _scatter_points(2, bad, OUTCOME_COLORS["bad"])

    pval, method = _mann_whitney_u_pvalue(good, bad)
    n_good, n_bad = 20, 16

    ax1.set_title(
        f"Boxplot — good vs bad surgical outcome\n"
        f"Mann–Whitney U p={pval:.3g} (n_good={n_good}, n_bad={n_bad})",
        fontsize = 14
    )
    ax1.set_ylabel("patient final_score")
    ax1.grid(True, axis="y", alpha=0.3)

    if method != "mannwhitneyu_scipy":
        ax1.text(
            0.02, 0.02,
            f"test={method}",
            transform=ax1.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            alpha=0.85,
        )

    ax0.set_xlabel(score_name_short)
    ax1.set_ylabel(score_name_short)
    fig.suptitle(f"Cohort — {score_name_long}", y=0.98, fontsize = 18)
    ax0.set_title("Barplot — all patients (sorted)", fontsize = 14)


    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=OUTCOME_COLORS["good"], label="good outcome"),
        Patch(facecolor=OUTCOME_COLORS["bad"], label="bad outcome"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    outpng = os.path.join(outdir, "cohort_patient_scores_change_global_abs.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure cohorte sauvegardée → {outpng}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return cohort



def load_bce_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "patient" not in df.columns or "bce_onset" not in df.columns:
        raise ValueError(f"CSV BCE invalide: {csv_path} (colonnes attendues: patient, bce_onset, ...)")
    df["patient"] = df["patient"].astype(str)
    if "seizure_id" in df.columns:
        df["seizure_id"] = df["seizure_id"].astype(str)
    df["bce_onset"] = pd.to_numeric(df["bce_onset"], errors="coerce")
    df["patient_key"] = df["patient"].map(normalize_patient_id)
    return df


def load_two_bce_csvs(bce_good: str | None, bce_bad: str | None) -> pd.DataFrame:
    dfs = []
    if bce_good:
        dfg = load_bce_csv(bce_good)
        dfg["outcome"] = "good"
        dfs.append(dfg)
    if bce_bad:
        dfb = load_bce_csv(bce_bad)
        dfb["outcome"] = "bad"
        dfs.append(dfb)

    if not dfs:
        raise ValueError("Il faut fournir au moins --bce_good ou --bce_bad pour le plot BCE.")

    df = pd.concat(dfs, ignore_index=True)

    n_out = df.groupby("patient")["outcome"].nunique()
    conflicted = n_out[n_out > 1].index.tolist()
    if conflicted:
        df.loc[df["patient"].isin(conflicted), "outcome"] = "unknown"

    return df.reset_index(drop=True)


def aggregate_bce_per_patient(bce_df: pd.DataFrame, agg: str = "median", drop_unknown_seizure: bool = True) -> pd.DataFrame:
    tmp = bce_df.copy()

    if drop_unknown_seizure and "seizure_id" in tmp.columns:
        tmp = tmp[tmp["seizure_id"] != "?"].copy()

    tmp = tmp.dropna(subset=["bce_onset"]).copy()

    agg = (agg or "median").lower()
    if agg not in {"median", "mean", "min", "max"}:
        raise ValueError(f"--bce_agg invalide: {agg} (choix: median, mean, min, max)")

    if agg == "median":
        f = np.median
    elif agg == "mean":
        f = np.mean
    elif agg == "min":
        f = np.min
    else:
        f = np.max

    out = (
        tmp.groupby("patient_key")
        .agg(
            patient=("patient", lambda x: x.iloc[0]),
            bce_value=("bce_onset", lambda x: float(f(x.to_numpy(dtype=float)))),
            n_rows=("bce_onset", "size"),
            outcome=("outcome", lambda x: x.iloc[0] if x.nunique() == 1 else "unknown"),
        )
        .reset_index()
    )
    return out


def load_f1_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "patient" not in df.columns:
        raise ValueError(f"CSV F1 invalide: {csv_path} (colonne 'patient' manquante)")
    if "F1@top_10pct" not in df.columns:
        raise ValueError(f"CSV F1 invalide: {csv_path} (colonne 'F1@top_10pct' manquante)")

    df["patient"] = df["patient"].astype(str)
    df["patient_key"] = df["patient"].map(normalize_patient_id)
    df["F1@top_10pct"] = pd.to_numeric(df["F1@top_10pct"], errors="coerce")
    return df


def load_two_f1_csvs(f1_good: str | None, f1_bad: str | None) -> pd.DataFrame:
    dfs = []
    if f1_good:
        dfg = load_f1_csv(f1_good)
        dfg["outcome"] = "good"
        dfs.append(dfg)
    if f1_bad:
        dfb = load_f1_csv(f1_bad)
        dfb["outcome"] = "bad"
        dfs.append(dfb)

    if not dfs:
        raise ValueError("Il faut fournir au moins --f1_good ou --f1_bad pour le plot F1.")

    df = pd.concat(dfs, ignore_index=True)

    n_out = df.groupby("patient_key")["outcome"].nunique()
    conflicted = n_out[n_out > 1].index.tolist()
    if conflicted:
        df.loc[df["patient_key"].isin(conflicted), "outcome"] = "unknown"

    return df.reset_index(drop=True)


def aggregate_f1_per_patient(f1_df: pd.DataFrame, agg: str = "max") -> pd.DataFrame:
    tmp = f1_df.copy()
    tmp = tmp.dropna(subset=["F1@top_10pct"]).copy()

    agg = (agg or "max").lower()
    if agg not in {"max", "mean", "median"}:
        raise ValueError(f"--f1_agg invalide: {agg} (choix: max, mean, median)")

    if agg == "max":
        f = np.max
    elif agg == "mean":
        f = np.mean
    else:
        f = np.median

    out = (
        tmp.groupby("patient_key")
        .agg(
            patient=("patient", lambda x: x.iloc[0]),
            f1_value=("F1@top_10pct", lambda x: float(f(x.to_numpy(dtype=float)))),
            n_rows=("F1@top_10pct", "size"),
            outcome=("outcome", lambda x: x.iloc[0] if x.nunique() == 1 else "unknown"),
        )
        .reset_index()
    )
    return out


def _corr_stats(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    out = {"n": int(x.size)}
    if x.size < 3:
        out.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        return out

    try:
        from scipy.stats import pearsonr, spearmanr
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        out.update(
            {
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            }
        )
    except Exception:
        r = float(np.corrcoef(x, y)[0, 1])
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rho = float(np.corrcoef(rx, ry)[0, 1])
        out.update({"pearson_r": r, "pearson_p": np.nan, "spearman_rho": rho, "spearman_p": np.nan})

    return out


def plot_bce_vs_finalscore_change_global_abs(cohort_scores_df, bce_patient_df, outdir, noshow=False, annotate=False, debug=True):
    os.makedirs(outdir, exist_ok=True)
    ycol = "patient_score__change_global_abs"

    cohort2 = cohort_scores_df.copy()
    cohort2["patient_key"] = cohort2["patient"].map(normalize_patient_id)

    merged = cohort2[["patient", "patient_key", "outcome", ycol]].merge(
        bce_patient_df[["patient_key", "bce_value", "n_rows", "outcome"]].rename(columns={"outcome": "bce_outcome"}),
        on="patient_key",
        how="inner",
    )

    merged[ycol] = pd.to_numeric(merged[ycol], errors="coerce")
    merged["bce_value"] = pd.to_numeric(merged["bce_value"], errors="coerce")
    merged = merged.dropna(subset=["bce_value", ycol]).copy()

    if debug:
        cohort_keys = set(cohort2["patient_key"].dropna().astype(str).tolist())
        bce_keys = set(bce_patient_df["patient_key"].dropna().astype(str).tolist())
        inter = sorted(list(cohort_keys & bce_keys))
        print(f"[DEBUG] cohort patients: {len(cohort_keys)} | bce patients: {len(bce_keys)} | intersection: {len(inter)}")
        print(f"[DEBUG] example intersection: {inter[:10]}")

    if merged.empty:
        print("[WARN] Aucun patient en intersection BCE × cohort_scores (ou valeurs NaN).")
        return merged

    stats_all = _corr_stats(merged[ycol].to_numpy(), merged["bce_value"].to_numpy())

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(
        "Association between Preictal-to-Ictal Change Consistency Score\n"
        "and Patient Level Model 1 Performance (BCE)\n"
        f"Pearson r={stats_all['pearson_r']:.3g} (p={stats_all['pearson_p']:.3g})"
    )







    for outcome in ["good", "bad", "unknown"]:
        sub = merged[merged["outcome"] == outcome]
        if sub.empty:
            continue
        ax.scatter(
            sub[ycol].to_numpy(),
            sub["bce_value"].to_numpy(),
            alpha=0.85,
            s=55,
            label=outcome,
            color=OUTCOME_COLORS.get(outcome, "tab:gray"),
            edgecolors="none",
        )
        if annotate:
            for _, r in sub.iterrows():
                ax.text(float(r[ycol]), float(r["bce_value"]), f" {r['patient']}", fontsize=8, alpha=0.9)

    x = merged[ycol].to_numpy(dtype=float)
    y = merged["bce_value"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size >= 2:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ys = a * xs + b
        ax.plot(xs, ys, linewidth=2)

    ax.set_xlabel("Preictal-to-Ictal Change Consistency Score")
    ax.set_ylabel("Patient-level Model 1 BCE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    outpng = os.path.join(outdir, "bce_vs_finalscore_change_global_abs.png")
    fig.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure BCE corr sauvegardée → {outpng}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    outcsv = os.path.join(outdir, "bce_vs_finalscore_change_global_abs_merged.csv")
    merged.to_csv(outcsv, index=False)
    print(f"[OK] CSV merge BCE×score sauvegardé → {outcsv}")

    return merged


def plot_f1_vs_finalscore_logx(
    cohort_scores_df: pd.DataFrame,
    f1_patient_df: pd.DataFrame,
    outdir: str,
    xcol: str,
    save_tag: str,
    noshow: bool = False,
    annotate: bool = False,
    debug: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    need_cols = {"patient", "outcome", xcol}
    missing = need_cols - set(cohort_scores_df.columns)
    if missing:
        raise ValueError(f"cohort_scores_df manque colonnes: {missing}")

    cohort2 = cohort_scores_df.copy()
    cohort2["patient_key"] = cohort2["patient"].map(normalize_patient_id)

    merged = cohort2[["patient", "patient_key", "outcome", xcol]].merge(
        f1_patient_df[["patient_key", "f1_value", "n_rows", "outcome"]].rename(columns={"outcome": "f1_outcome"}),
        on="patient_key",
        how="inner",
    )

    merged[xcol] = pd.to_numeric(merged[xcol], errors="coerce")
    merged["f1_value"] = pd.to_numeric(merged["f1_value"], errors="coerce")
    merged = merged.dropna(subset=[xcol, "f1_value"]).copy()

    # IMPORTANT: log(x) => x doit être > 0
    n_before = len(merged)
    merged = merged[merged[xcol] > 0].copy()
    if debug:
        print(f"[DEBUG LOGX] rows before filter: {n_before} | after x>0: {len(merged)}")

    if debug:
        cohort_keys = set(cohort2["patient_key"].dropna().astype(str))
        f1_keys = set(f1_patient_df["patient_key"].dropna().astype(str))
        inter = sorted(list(cohort_keys & f1_keys))
        print(f"[DEBUG F1] cohort patients: {len(cohort_keys)} | f1 patients: {len(f1_keys)} | intersection: {len(inter)}")
        print(f"[DEBUG F1] example intersection: {inter[:10]}")
        print("[DEBUG F1] merged outcome counts:")
        print(merged[["outcome", "f1_outcome"]].value_counts(dropna=False))

    if merged.empty:
        print("[WARN] Aucun point restant après filtre x>0 (log).")
        return merged

    stats_all = _corr_stats(merged[xcol].to_numpy(), merged["f1_value"].to_numpy())

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(
        f"Association between Preictal-to-Ictal Change Consistency Score (log X)\n"
        f"and Patient-level Model 2 Performance (F1@top 10%)\n"
        f"Pearson r={stats_all['pearson_r']:.3g} (p={stats_all['pearson_p']:.3g})"
    )

    for outcome in ["good", "bad", "unknown"]:
        sub = merged[merged["outcome"] == outcome]
        if sub.empty:
            continue
        ax.scatter(
            sub[xcol].to_numpy(),
            sub["f1_value"].to_numpy(),
            alpha=0.85,
            s=55,
            label=outcome,
            color=OUTCOME_COLORS.get(outcome, "tab:gray"),
            edgecolors="none",
        )
        if annotate:
            for _, r in sub.iterrows():
                ax.text(float(r[xcol]), float(r["f1_value"]), f" {r['patient']}", fontsize=8, alpha=0.9)

    # Régression sur x (linéaire) mais affichée en axe log (xscale log)
    x = merged[xcol].to_numpy(dtype=float)
    y = merged["f1_value"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[m]
    y = y[m]
    if x.size >= 2:
        a, b = np.polyfit(x, y, 1)
        xs = np.geomspace(float(np.min(x)), float(np.max(x)), 200)  # mieux pour axe log
        ys = a * xs + b
        ax.plot(xs, ys, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Preictal-to-Ictal Change Consistency Score (log scale)")
    ax.set_ylabel("Patient-level Model 2 F1@top 10%")

    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    outpng = os.path.join(outdir, f"f1_vs_finalscore_{save_tag}_logx.png")
    fig.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure F1 corr (log X) sauvegardée → {outpng}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    outcsv = os.path.join(outdir, f"f1_vs_finalscore_{save_tag}_logx_merged.csv")
    merged.to_csv(outcsv, index=False)
    print(f"[OK] CSV merge (log X) sauvegardé → {outcsv}")

    return merged



def plot_subscores_per_feature_across_patients(
    scores_long: pd.DataFrame,
    outdir: str,
    y_col: str,
    patients_order_by: str = "final_score",
    show_all_patient_labels: bool = True,
    max_patients_labels: int = 60,
    label_fontsize: int = 7,
    fig_width_per_patient: float = 0.28,
    noshow: bool = False,
):
    os.makedirs(outdir, exist_ok=True)

    need = {
        "patient", "feature", "outcome",
        "amp_norm", "amp_term", "stability", "sign_term", "final_score"
    }
    missing = need - set(scores_long.columns)
    if missing:
        raise ValueError(f"scores_long manque colonnes: {missing}")

    df = scores_long.copy()
    for c in ["amp_norm", "amp_term", "stability", "sign_term", "final_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    subs = ["amp_norm", "amp_term", "stability", "sign_term", "final_score"]

    for feat, g in df.groupby("feature"):
        g = g.copy()
        g["patient"] = g["patient"].astype(str)

        outcome_map = (
            g.groupby("patient")["outcome"]
            .agg(lambda x: x.iloc[0] if x.nunique() == 1 else "unknown")
            .to_dict()
        )

        if patients_order_by in g.columns:
            order = (
                g.groupby("patient")[patients_order_by]
                .median(numeric_only=True)
                .sort_values(ascending=False)
                .index.astype(str)
                .tolist()
            )
        else:
            order = sorted(g["patient"].unique().tolist())

        g["patient"] = pd.Categorical(g["patient"], categories=order, ordered=True)
        g = g.sort_values("patient")

        n_pat = len(order)

        M = np.full((n_pat, len(subs)), np.nan, dtype=float)
        for i, p in enumerate(order):
            row = g[g["patient"].astype(str) == str(p)]
            if row.empty:
                continue
            M[i, :] = row[subs].median(numeric_only=True).to_numpy(dtype=float)

        bar_colors = [
            OUTCOME_COLORS.get(outcome_map.get(p, "unknown"), OUTCOME_COLORS["unknown"])
            for p in order
        ]

        fig_w = max(14.0, float(fig_width_per_patient) * n_pat)
        fig_h = 2.2 * len(subs)
        fig, axes = plt.subplots(len(subs), 1, figsize=(fig_w, fig_h), sharex=True)
        if len(subs) == 1:
            axes = [axes]

        x = np.arange(n_pat)

        if show_all_patient_labels:
            xticklabels = order
        else:
            if n_pat <= max_patients_labels:
                xticklabels = order
            else:
                step = int(np.ceil(n_pat / max_patients_labels))
                xticklabels = [p if (i % step == 0) else "" for i, p in enumerate(order)]

        for j, subname in enumerate(subs):
            ax = axes[j]
            ax.bar(x, M[:, j], color=bar_colors, alpha=0.9)
            ax.set_ylabel(subname)
            ax.grid(True, axis="y", alpha=0.25)
            if subname == "sign_term":
                ax.axhline(0.0, linewidth=1.2, alpha=0.6)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(xticklabels, rotation=90, fontsize=label_fontsize)
        axes[-1].set_xlabel("patients")

        fig.suptitle(f"{y_col} — feature={feat} | subscores par patient", y=1.01)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=OUTCOME_COLORS["good"], label="good outcome"),
            Patch(facecolor=OUTCOME_COLORS["bad"], label="bad outcome"),
            Patch(facecolor=OUTCOME_COLORS["unknown"], label="unknown/conflict"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        outpng = os.path.join(outdir, f"subscores_{y_col}__feature_{feat}.png")
        fig.savefig(outpng, dpi=200, bbox_inches="tight")
        if not noshow:
            plt.show()
        else:
            plt.close(fig)

        print(f"[OK] {outpng}")


def smooth_value_field_on_grid(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    gridsize: int = 250,
    sigma: float = 0.25,
    padding: float = 0.06,
):
    """
    Estime un champ continu v(x,y) sur une grille via un noyau gaussien.
    sigma est en "unités UMAP" (typiquement 0.15–0.5 selon l'échelle).
    Retourne: Xg, Yg, Z (2D) où Z = valeur lissée, NaN là où pas de support.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[m], y[m], v[m]
    if x.size < 3:
        return None, None, None

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = (xmax - xmin) * padding
    dy = (ymax - ymin) * padding
    xmin -= dx; xmax += dx
    ymin -= dy; ymax += dy

    xs = np.linspace(xmin, xmax, int(gridsize))
    ys = np.linspace(ymin, ymax, int(gridsize))
    Xg, Yg = np.meshgrid(xs, ys)

    # weighted Gaussian smoothing:
    # Z = sum_i w_i * v_i / sum_i w_i, w_i = exp(-d^2/(2*sigma^2))
    Z_num = np.zeros_like(Xg, dtype=float)
    Z_den = np.zeros_like(Xg, dtype=float)

    sig2 = float(sigma) ** 2
    if sig2 <= 0:
        sig2 = 1e-6

    # boucle sur points (OK pour ~100-500 points; sinon on optimisera)
    for xi, yi, vi in zip(x, y, v):
        d2 = (Xg - xi) ** 2 + (Yg - yi) ** 2
        w = np.exp(-0.5 * d2 / sig2)
        Z_num += w * vi
        Z_den += w

    Z = Z_num / np.maximum(Z_den, 1e-12)

    # masque des zones trop loin de tout point (évite “inventer” sur les bords)
    # ici: on met NaN si la somme de poids est trop petite
    thr = np.percentile(Z_den, 2)  # petit seuil adaptatif
    Z[Z_den < thr] = np.nan

    return Xg, Yg, Z


# --- PATCH: quantify how well UMAP separates SOZ groups (silhouette + kNN purity) ---
# Add these imports near the top (or inside the function if you prefer):
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def soz_separation_metrics_from_embedding(
    emb_2d: np.ndarray,
    soz_groups: np.ndarray,
    k: int = 5,
    drop_unknown: bool = True,
) -> dict:
    """
    emb_2d: (N,2) array, UMAP coords
    soz_groups: (N,) array of strings in {"temporal","frontal","insular","mixed","unknown"}
    Returns: {"n":..., "n_groups":..., "silhouette":..., "knn_purity":...}
    """
    emb_2d = np.asarray(emb_2d, float)
    labels = np.asarray(soz_groups).astype(str)

    if drop_unknown:
        m = labels != "unknown"
        emb_2d = emb_2d[m]
        labels = labels[m]

    uniq = np.unique(labels)
    if emb_2d.shape[0] < 3 or uniq.size < 2:
        return {"n": int(emb_2d.shape[0]), "n_groups": int(uniq.size), "silhouette": np.nan, "knn_purity": np.nan}

    # 1) Silhouette (higher = better, ~0 = overlap)
    sil = float(silhouette_score(emb_2d, labels, metric="euclidean"))

    # 2) kNN label purity (higher = better; with 4 groups, random ~0.25)
    kk = int(min(k, max(1, emb_2d.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(emb_2d)
    _, idx = nn.kneighbors(emb_2d)

    pur = []
    for i in range(emb_2d.shape[0]):
        neigh = labels[idx[i][1:]]  # skip self
        pur.append(np.mean(neigh == labels[i]))
    purity = float(np.mean(pur)) if pur else np.nan

    return {"n": int(emb_2d.shape[0]), "n_groups": int(uniq.size), "silhouette": sil, "knn_purity": purity}

def plot_top_features_by_soz_group(
    df: pd.DataFrame,
    outdir: str,
    y_col: str = "change_global_abs",
    topn: int = 15,
    score_mode: str = "importance",   # "median" | "importance"
    min_patients_per_group: int = 3,
    drop_unknown_group: bool = True,
    noshow: bool = False,
    # scoring params (same as your pipeline)
    eps: float = 1e-12,
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.5,
    amp_alpha: float = 0.7,
):
    """
    But: Selon le type d'implant (SOZ group), afficher les features qui ont les meilleurs scores.

    - Calcule scores_long (final_score par patient & feature) via build_feature_scores_table_all_patients()
    - Ajoute soz_group via tes sets SOZ_* (build_soz_group_map)
    - Agrège par (soz_group, feature):
        median_score = median(final_score)
        support      = fraction(final_score > 0)
        n_patients   = nb de patients du groupe ayant une valeur
        importance   = median_score * support
    - Plot: 1 subplot par groupe, barh des top features
    - Sauvegarde:
        * CSV: top_features_by_soz_group__{y_col}.csv (table complète)
        * PNG: top_features_by_soz_group__{y_col}.png

    score_mode:
      - "median": classe par median_score
      - "importance": classe par importance (= median_score * support) -> plus robuste aux outliers
    """
    os.makedirs(outdir, exist_ok=True)
    print(y_col)

    # 1) Long table: patient, outcome, feature, final_score
    scores_long = build_feature_scores_table_all_patients(
        df=df,
        y_col=y_col,
        eps=eps,
        zero_rel=zero_rel,
        zero_abs_thr=zero_abs_thr,
        stability_alpha=stability_alpha,
        amp_alpha=amp_alpha,
    ).copy()

    if scores_long.empty:
        print("[WARN] scores_long vide -> impossible de faire top features par SOZ group.")
        return None

    scores_long["final_score"] = pd.to_numeric(scores_long["final_score"], errors="coerce")
    scores_long = scores_long.dropna(subset=["final_score"]).copy()
    if scores_long.empty:
        print("[WARN] scores_long sans final_score valide -> impossible.")
        return None

    # 2) Add patient_key + soz_group
    scores_long["patient_key"] = scores_long["patient"].astype(str).map(normalize_patient_id)
    soz_map = build_soz_group_map()
    scores_long["soz_group"] = scores_long["patient_key"].map(lambda k: soz_map.get(k, "unknown"))

    if drop_unknown_group:
        scores_long = scores_long[scores_long["soz_group"] != "unknown"].copy()

    if scores_long.empty:
        print("[WARN] Après drop_unknown_group, plus de données.")
        return None

    # 3) Aggregate per (soz_group, feature)
    agg = (
        scores_long
        .groupby(["soz_group", "feature"])
        .agg(
            median_score=("final_score", "median"),
            support=("final_score", lambda x: float(np.mean(np.asarray(x, float) > 0.0))),
            n_patients=("patient_key", "nunique"),
        )
        .reset_index()
    )
    agg["importance"] = agg["median_score"] * agg["support"]

    # filter weak groups/features support (optional)
    agg = agg[agg["n_patients"] >= int(min_patients_per_group)].copy()
    if agg.empty:
        print("[WARN] Aucune feature avec assez de patients par groupe (min_patients_per_group).")
        return None

    # 4) Save CSV (full table)
    outcsv = os.path.join(outdir, f"top_features_by_soz_group__{y_col}.csv")
    agg.sort_values(["soz_group", "importance"], ascending=[True, False]).to_csv(outcsv, index=False)
    print(f"[OK] CSV top-features (par SOZ group) sauvegardé → {outcsv}")

    # 5) Plot
    score_mode = (score_mode or "importance").lower()
    if score_mode not in {"median", "importance"}:
        raise ValueError("score_mode invalide (median | importance)")

    score_col = "median_score" if score_mode == "median" else "importance"
    score_label = "median(final_score)" if score_mode == "median" else "importance = median(final_score) × support"

    # Group ordering (nice, consistent)
    group_order = ["temporal", "medial_temporal", "frontal", "insular", "parietal", "mixed"]
    present_groups = [g for g in group_order if g in set(agg["soz_group"].unique())]
    # If any other labels exist, append them
    for g in sorted(set(agg["soz_group"].unique()) - set(present_groups)):
        present_groups.append(g)

    n_groups = len(present_groups)
    if n_groups == 0:
        print("[WARN] Aucun groupe SOZ à afficher.")
        return None

    # layout: 2 columns (or 1 if only 1 group)
    ncols = 2 if n_groups > 1 else 1
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    fig.suptitle(
        f"Top features par type d'implantation (SOZ group)\n"
        f"y_col={y_col} | score={score_mode} | topn={topn} | min_patients_per_group={min_patients_per_group}\n"
        f"stability_alpha={stability_alpha}, amp_alpha={amp_alpha}, zero_rel={zero_rel}, zero_abs_thr={zero_abs_thr}",
        y=0.995,
        fontsize=14,
    )

    for i, g in enumerate(present_groups):
        ax = axes[i // ncols, i % ncols]
        sub = agg[agg["soz_group"] == g].sort_values(score_col, ascending=False).head(int(topn)).copy()

        if sub.empty:
            ax.set_title(f"{g} (no data)")
            ax.axis("off")
            continue

        # plot in ascending order for nice barh
        sub = sub.sort_values(score_col, ascending=True)

        y = np.arange(sub.shape[0])
        vals = sub[score_col].to_numpy(dtype=float)
        labels = sub["feature"].astype(str).tolist()

        ax.barh(y, vals, alpha=0.9, color=SOZ_COLORS.get(g, "tab:gray"))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{g} — Top {sub.shape[0]} features")
        ax.set_xlabel(score_label)
        ax.grid(True, axis="x", alpha=0.25)

        # annotate support + n_patients
        for yy, (v, sup, npat) in enumerate(zip(vals, sub["support"].to_numpy(float), sub["n_patients"].to_numpy(int))):
            ax.text(
                float(v),
                yy,
                f"  sup={sup:.2f}, n={int(npat)}",
                va="center",
                ha="left",
                fontsize=8,
                alpha=0.85,
            )

    # hide unused axes
    for j in range(n_groups, nrows * ncols):
        ax = axes[j // ncols, j % ncols]
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outpng = os.path.join(outdir, f"top_features_by_soz_group__{y_col}.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure top-features (par SOZ group) sauvegardée → {outpng}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return agg
def plot_top_features_by_soz_group_with_heatmap(
    df: pd.DataFrame,
    outdir: str,
    y_col: str = "change_global_abs",
    # barplots (top row)
    topn_bar: int = 15,
    n_groups_bar: int = 3,  # nb de SOZ groups affichés en barplots (rangés par n_patients)
    # heatmap
    heatmap_top_features: int = 40,  # nb max de features affichées en heatmap
    # scoring/ranking
    score_mode: str = "importance",   # "median" | "importance"
    min_patients_per_group_feature: int = 3,  # filtre n_patients AU NIVEAU (group, feature)
    drop_unknown_group: bool = True,
    noshow: bool = False,
    # scoring params (same as your pipeline)
    eps: float = 1e-12,
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.5,
    amp_alpha: float = 0.7,
    # combined figure options
    include_unknown_outcome_in_outcome_panel: bool = False,  # si False: only good/bad
):
    """
    Fait:
      1) Barplots: TOP features (final_score agrégé) pour les N SOZ groups avec le plus de patients
      2) Heatmap: rows=SOZ groups, cols=features, values=score (median ou importance)
      3) Figure combinée (NOUVELLE version): 3 panneaux empilés (mêmes features):
            - (haut) GLOBAL: 1 barre par feature (pas de segmentation)
            - (milieu) OUTCOME: barres côte-à-côte good vs bad (optionnel unknown)
            - (bas) SOZ: barres côte-à-côte par SOZ implant (comme avant)

    Détails:
      - scores_long = build_feature_scores_table_all_patients(df, y_col=...)
      - Ajoute patient_key, soz_group via build_soz_group_map()
      - Agrège par (soz_group, feature):
          median_score = median(final_score)
          support      = fraction(final_score > 0)
          n_patients   = nb de patients du groupe ayant une valeur pour la feature
          importance   = median_score * support
      - Filtre (group,feature) si n_patients < min_patients_per_group_feature

    IMPORTANT:
      - Dans les barplots groupés (global/outcome/soz), le label feature est "pretty"
        (on enlève "burstint_" du nom affiché), mais les données restent inchangées.

    Outputs:
      - CSV long agrégé:  top_features_by_soz_group__{y_col}.csv
      - CSV pivot heatmap: heatmap_features_by_soz_group__{y_col}.csv
      - PNG figure: top_features_by_soz_group_with_heatmap__{y_col}.png
      - PNG figure combinée 3 panneaux: combined_barplots_global_outcome_soz__{y_col}.png

    Retour:
      (agg_df, heatmap_pivot_df)
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(outdir, exist_ok=True)

    score_mode = (score_mode or "importance").lower()
    if score_mode not in {"median", "importance"}:
        raise ValueError("score_mode invalide (median | importance)")
    score_col = "median_score" if score_mode == "median" else "importance"
    score_label = "median(final_score)" if score_mode == "median" else "importance = median(final_score) × support"

    def _pretty_feat(name: str) -> str:
        return str(name).replace("burstint_", "")

    # ------------------------------------------------------------
    # 1) Compute long scores per patient & feature
    # ------------------------------------------------------------
    scores_long = build_feature_scores_table_all_patients(
        df=df,
        y_col=y_col,
        eps=eps,
        zero_rel=zero_rel,
        zero_abs_thr=zero_abs_thr,
        stability_alpha=stability_alpha,
        amp_alpha=amp_alpha,
    ).copy()

    if scores_long.empty:
        print("[WARN] scores_long vide -> impossible.")
        return None, None

    scores_long["final_score"] = pd.to_numeric(scores_long["final_score"], errors="coerce")
    scores_long = scores_long.dropna(subset=["final_score"]).copy()
    if scores_long.empty:
        print("[WARN] scores_long sans final_score valide -> impossible.")
        return None, None

    # Add patient_key + soz_group
    scores_long["patient_key"] = scores_long["patient"].astype(str).map(normalize_patient_id)
    soz_map = build_soz_group_map()
    scores_long["soz_group"] = scores_long["patient_key"].map(lambda k: soz_map.get(k, "unknown"))

    # ------------------------------------------------------------
    # 2) Aggregate per (SOZ group, feature)
    # ------------------------------------------------------------
    scores_soz = scores_long.copy()
    if drop_unknown_group:
        scores_soz = scores_soz[scores_soz["soz_group"] != "unknown"].copy()
    if scores_soz.empty:
        print("[WARN] Après drop_unknown_group, plus de données.")
        return None, None

    group_sizes = (
        scores_soz.groupby("soz_group")["patient_key"]
        .nunique()
        .sort_values(ascending=False)
    )
    if group_sizes.empty:
        print("[WARN] Aucun groupe SOZ présent.")
        return None, None

    top_groups_for_bar = group_sizes.head(int(n_groups_bar)).index.tolist()

    agg = (
        scores_soz
        .groupby(["soz_group", "feature"])
        .agg(
            median_score=("final_score", "median"),
            support=("final_score", lambda x: float(np.mean(np.asarray(x, float) > 0.0))),
            n_patients=("patient_key", "nunique"),
        )
        .reset_index()
    )
    agg["importance"] = agg["median_score"] * agg["support"]

    # Filter at (group, feature) level
    agg = agg[agg["n_patients"] >= int(min_patients_per_group_feature)].copy()
    if agg.empty:
        print("[WARN] Rien après filtre min_patients_per_group_feature.")
        return None, None

    outcsv = os.path.join(outdir, f"top_features_by_soz_group__{y_col}.csv")
    agg.sort_values(["soz_group", score_col], ascending=[True, False]).to_csv(outcsv, index=False)
    print(f"[OK] CSV agrégé sauvegardé → {outcsv}")

    # ------------------------------------------------------------
    # 3) Heatmap pivot: rows=groups, cols=features (top K globally)
    # ------------------------------------------------------------
    feat_rank = (
        agg.groupby("feature")[score_col]
        .max()
        .sort_values(ascending=False)
    )
    selected_features = feat_rank.head(int(heatmap_top_features)).index.tolist()

    all_groups = group_sizes.index.tolist()
    all_groups = [g for g in all_groups if g in set(agg["soz_group"].unique())]

    heatmap_pivot = (
        agg[agg["feature"].isin(selected_features)]
        .pivot_table(index="soz_group", columns="feature", values=score_col, aggfunc="median")
        .reindex(index=all_groups, columns=selected_features)
    )

    outcsv_hm = os.path.join(outdir, f"heatmap_features_by_soz_group__{y_col}.csv")
    heatmap_pivot.to_csv(outcsv_hm, index=True)
    print(f"[OK] CSV heatmap pivot sauvegardé → {outcsv_hm}")

    # ------------------------------------------------------------
    # 4) Figure: top row = barplots SOZ (top groups), bottom row = heatmap
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.25)

    fig.suptitle(
        f"Top features par type d'implantation + heatmap\n"
        f"y_col={y_col} | score={score_mode} | bar_topn={topn_bar} | heatmap_top_features={heatmap_top_features}\n"
        f"min_patients_per_group_feature={min_patients_per_group_feature} | "
        f"stability_alpha={stability_alpha}, amp_alpha={amp_alpha}",
        y=0.985,
        fontsize=14,
    )

    # ---- Barplots (N groupes max, mais grille 1x3 pour compat)
    for j in range(3):
        ax = fig.add_subplot(gs[0, j])
        if j >= len(top_groups_for_bar):
            ax.axis("off")
            continue

        g = top_groups_for_bar[j]
        sub = (
            agg[agg["soz_group"] == g]
            .sort_values(score_col, ascending=False)
            .head(int(topn_bar))
            .copy()
        )

        n_pat_total = int(group_sizes.loc[g]) if g in group_sizes.index else 0
        ax.set_title(f"{g} (n={n_pat_total}) — Top {sub.shape[0]} features", fontsize=12)

        if sub.empty:
            ax.text(0.5, 0.5, "No features passing filters", ha="center", va="center",
                    transform=ax.transAxes, alpha=0.8)
            ax.grid(True, axis="x", alpha=0.2)
            continue

        sub = sub.sort_values(score_col, ascending=True)  # for barh
        y = np.arange(sub.shape[0])
        vals = sub[score_col].to_numpy(dtype=float)
        labels = [_pretty_feat(x) for x in sub["feature"].astype(str).tolist()]

        ax.barh(y, vals, alpha=0.9, color=SOZ_COLORS.get(g, "tab:gray"))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(score_label)
        ax.grid(True, axis="x", alpha=0.25)

        for yy, (v, sup, npat) in enumerate(zip(vals, sub["support"].to_numpy(float), sub["n_patients"].to_numpy(int))):
            ax.text(
                float(v),
                yy,
                f"  sup={sup:.2f}, nfeat={int(npat)}",
                va="center",
                ha="left",
                fontsize=8,
                alpha=0.85,
            )

    # ---- Heatmap spanning bottom row
    ax_hm = fig.add_subplot(gs[1, :])

    Z = heatmap_pivot.to_numpy(dtype=float)
    Zm = np.ma.masked_invalid(Z)
    im = ax_hm.imshow(Zm, aspect="auto", interpolation="nearest", cmap="viridis")

    ylabels = []
    for g in heatmap_pivot.index.tolist():
        n_pat_total = int(group_sizes.loc[g]) if g in group_sizes.index else 0
        ylabels.append(f"{g} (n={n_pat_total})")

    ax_hm.set_yticks(np.arange(len(ylabels)))
    ax_hm.set_yticklabels(ylabels, fontsize=10)

    xlabels = [_pretty_feat(x) for x in heatmap_pivot.columns.astype(str).tolist()]
    ax_hm.set_xticks(np.arange(len(xlabels)))
    ax_hm.set_xticklabels(xlabels, rotation=90, fontsize=8)

    ax_hm.set_title(f"Heatmap — rows=SOZ group, cols=features (values={score_col})", fontsize=12)
    ax_hm.set_xlabel("features")
    ax_hm.set_ylabel("SOZ implant group")
    ax_hm.grid(False)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.01)
    cbar.set_label(score_label)

    outpng = os.path.join(outdir, f"top_features_by_soz_group_with_heatmap__{y_col}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure barplots+heatmap sauvegardée → {outpng}")

    # ------------------------------------------------------------
    # 5) NEW COMBINED FIGURE: GLOBAL (no segregation) + OUTCOME + SOZ
    #    - Features = union des Top-N de chaque SOZ group (top_groups_for_bar)
    #    - Même ordre de features pour les 3 panels (défini par GLOBAL strength)
    # ------------------------------------------------------------
    if len(top_groups_for_bar) > 0:
        # collect each SOZ group's top features
        top_feats_per_group = {}
        for g in top_groups_for_bar:
            subg = (
                agg[agg["soz_group"] == g]
                .sort_values(score_col, ascending=False)
                .head(int(topn_bar))
            )
            top_feats_per_group[g] = subg["feature"].astype(str).tolist()

        # union of features (preserve order of discovery across groups)
        union_features = []
        seen = set()
        for g in top_groups_for_bar:
            for f in top_feats_per_group.get(g, []):
                if f not in seen:
                    seen.add(f)
                    union_features.append(f)

        if len(union_features) == 0:
            print("[WARN] Combined figure: aucune feature (union) disponible.")
        else:
            # ---------- GLOBAL aggregation (no segregation)
            agg_global = (
                scores_long
                .groupby("feature")
                .agg(
                    median_score=("final_score", "median"),
                    support=("final_score", lambda x: float(np.mean(np.asarray(x, float) > 0.0))),
                    n_patients=("patient_key", "nunique"),
                )
                .reset_index()
            )
            agg_global["importance"] = agg_global["median_score"] * agg_global["support"]
            agg_global = agg_global[agg_global["n_patients"] >= int(min_patients_per_group_feature)].copy()

            global_series = (
                agg_global[agg_global["feature"].astype(str).isin(union_features)]
                .assign(feature=lambda d: d["feature"].astype(str))
                .set_index("feature")[score_col]
                .reindex(union_features)
            )

            # feature order = global strength
            feat_order = global_series.sort_values(ascending=False).index.tolist()

            # ---------- OUTCOME aggregation
            scores_out = scores_long.copy()
            if include_unknown_outcome_in_outcome_panel:
                outcomes_show = ["good", "bad", "unknown"]
            else:
                outcomes_show = ["good", "bad"]
                scores_out = scores_out[scores_out["outcome"].isin(["good", "bad"])].copy()

            if scores_out.empty:
                # still plot outcome panel as "no data"
                outcome_mat = None
                outcome_ns = {}
            else:
                agg_outcome = (
                    scores_out
                    .groupby(["outcome", "feature"])
                    .agg(
                        median_score=("final_score", "median"),
                        support=("final_score", lambda x: float(np.mean(np.asarray(x, float) > 0.0))),
                        n_patients=("patient_key", "nunique"),
                    )
                    .reset_index()
                )
                agg_outcome["importance"] = agg_outcome["median_score"] * agg_outcome["support"]
                agg_outcome = agg_outcome[agg_outcome["n_patients"] >= int(min_patients_per_group_feature)].copy()

                outcome_mat = (
                    agg_outcome[
                        agg_outcome["feature"].astype(str).isin(feat_order)
                        & agg_outcome["outcome"].astype(str).isin(outcomes_show)
                    ]
                    .assign(feature=lambda d: d["feature"].astype(str), outcome=lambda d: d["outcome"].astype(str))
                    .pivot_table(index="feature", columns="outcome", values=score_col, aggfunc="median")
                    .reindex(index=feat_order, columns=outcomes_show)
                )

                # optional: n patients per outcome (for title)
                outcome_ns = (
                    scores_out.groupby("outcome")["patient_key"].nunique().to_dict()
                    if "outcome" in scores_out.columns else {}
                )

            # ---------- SOZ matrix (bottom panel)
            soz_mat = (
                agg[agg["soz_group"].isin(top_groups_for_bar) & agg["feature"].astype(str).isin(feat_order)]
                .assign(feature=lambda d: d["feature"].astype(str))
                .pivot_table(index="feature", columns="soz_group", values=score_col, aggfunc="median")
                .reindex(index=feat_order, columns=top_groups_for_bar)
            )

            # ---------- Plot combined 3 panels
            figc = plt.figure(figsize=(max(14, 0.55 * len(feat_order)), 16))
            gs2 = figc.add_gridspec(3, 1, height_ratios=[0.7, 1.0, 1.0], hspace=0.40)

            ax_global = figc.add_subplot(gs2[0, 0])
            ax_out = figc.add_subplot(gs2[1, 0])
            ax_soz = figc.add_subplot(gs2[2, 0])

            x = np.arange(len(feat_order), dtype=float)
            pretty_labels = [_pretty_feat(f) for f in feat_order]

            # --- TOP: GLOBAL
            vals_g = global_series.reindex(feat_order).to_numpy(dtype=float)
            ax_global.bar(x, vals_g, alpha=0.9, color="tab:gray")
            ax_global.set_title("All patients pooled", fontsize=12)
            ax_global.set_ylabel("Median Feature-level\nPI→I Change Consistency Score")
            ax_global.set_xticks(x)
            ax_global.set_xticklabels(pretty_labels, rotation=90, fontsize=8)
            ax_global.grid(True, axis="y", alpha=0.25)

            # --- MIDDLE: OUTCOME grouped bars
            if outcome_mat is None or outcome_mat.empty:
                ax_out.text(
                    0.5, 0.5, "No outcome data for this panel",
                    transform=ax_out.transAxes, ha="center", va="center", alpha=0.85
                )
                ax_out.axis("off")
            else:
                outs = outcomes_show
                nO = len(outs)
                total_width = 0.80
                bar_w = total_width / max(1, nO)
                offsets = (np.arange(nO) - (nO - 1) / 2.0) * bar_w

                for k, o in enumerate(outs):
                    vals = outcome_mat[o].to_numpy(dtype=float)
                    ax_out.bar(
                        x + offsets[k], vals, width=bar_w,
                        label=f"{o}", alpha=0.9,
                        color=OUTCOME_COLORS.get(o, "tab:gray"),
                    )

                ax_out.set_title(f"Outcome comparison", fontsize=12)
                ax_out.set_ylabel("Median Feature-level\nPI→I Change Consistency Score")
                ax_out.set_xticks(x)
                ax_out.set_xticklabels(pretty_labels, rotation=90, fontsize=8)
                ax_out.grid(True, axis="y", alpha=0.25)
                ax_out.legend(frameon=False, ncol=min(3, nO))

            # --- BOTTOM: SOZ grouped bars (existing logic)
            groups = top_groups_for_bar
            nG = len(groups)
            total_width = 0.80
            bar_w = total_width / max(1, nG)
            offsets = (np.arange(nG) - (nG - 1) / 2.0) * bar_w

            # robust colors: prefer SOZ_COLORS, fallback to tab10 cycle
            cmap_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

            def get_color(g, k):
                if "SOZ_COLORS" in globals() and isinstance(SOZ_COLORS, dict) and g in SOZ_COLORS:
                    return SOZ_COLORS.get(g)
                if cmap_colors:
                    return cmap_colors[k % len(cmap_colors)]
                return None

            for k, g in enumerate(groups):
                vals = soz_mat[g].to_numpy(dtype=float)
                ax_soz.bar(x + offsets[k], vals, width=bar_w, label=f"{g}", alpha=0.9, color=get_color(g, k))

            ax_soz.set_title(
                "SOZ implant localization comparison",
                fontsize=12,
            )
            ax_soz.set_ylabel("Median Feature-level\nPI→I Change Consistency Score")
            ax_soz.set_xticks(x)
            ax_soz.set_xticklabels(pretty_labels, rotation=90, fontsize=8)
            ax_soz.grid(True, axis="y", alpha=0.25)
            ax_soz.legend(frameon=False, ncol=min(2, nG), loc="upper right")

            figc.suptitle(
                f"Feature-level Preictal-to-Ictal Change Consistency Scores across Patient Groups\n"
                f"(Top) Global  •  (Middle) Outcome  •  (Bottom) SOZ implant group",
                y=0.93,
                fontsize=14,
            )

            figc.tight_layout(rect=[0, 0, 1, 0.97])

            outpngc = os.path.join(outdir, f"combined_barplots_global_outcome_soz__{y_col}.png")
            figc.savefig(outpngc, dpi=200, bbox_inches="tight")
            print(f"[OK] Figure combinée (3 panels) sauvegardée → {outpngc}")

            if not noshow:
                plt.show()
            else:
                plt.close(figc)

    else:
        print("[WARN] Combined figure: aucun groupe sélectionné pour les barplots (top_groups_for_bar vide).")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return agg, heatmap_pivot



def plot_cohort_by_soz_group(
    df: pd.DataFrame,
    outdir: str,
    score_col: str = "patient_score__change_global_abs",
    sort_col: str = "patient_score__change_soz_amp_abs",
    aggregate: str = "topk",
    k: int = 15,
    min_seizures: int = 3,
    noshow: bool = False,
    # scoring params (uniforme)
    zero_rel: float = 0.05,
    zero_abs_thr: float = 1e-8,
    stability_alpha: float = 1.5,
    amp_alpha: float = 0.7,
    # filtre top-N
    keep_top_patients_per_feature: int | None = None,
    # multiplier
    multiply_by_n_selected: bool = False,
    # boxplot options
    drop_unknown_soz_in_boxplot: bool = True,
    group_order: list[str] | None = None,
    show_points: bool = True,
    point_size: int = 28,
    random_seed: int = 0,
):
    """
    Figure cohorte (comme plot_cohort), mais colorée par SOZ implant localization.

    LEFT:
      - Barplot horizontal des scores patient (score_col) pour tous les patients
      - Patients triés par sort_col (croissant)
      - Couleurs = SOZ_COLORS[soz_group], sinon gris

    RIGHT:
      - Boxplot par soz_group + points individuels (jitter)
      - Exclut soz_group == "unknown" du boxplot/test si drop_unknown_soz_in_boxplot=True

    Dépendances attendues dans ton script:
      - cohort_patient_scores
      - normalize_patient_id
      - build_soz_group_map
      - SOZ_COLORS
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(outdir, exist_ok=True)

    # -----------------------
    # Helpers robustes
    # -----------------------
    def _get_1d_series(df_: pd.DataFrame, col: str, *, where: str = "") -> pd.Series:
        """Retourne une Series 1D même si colonnes dupliquées."""
        if col not in df_.columns:
            raise ValueError(f"[plot_cohort_by_soz_group] Colonne absente: '{col}' {where}. "
                             f"Colonnes dispo: {list(df_.columns)}")
        x = df_.loc[:, col]  # Series ou DataFrame si dupliquée
        if isinstance(x, pd.DataFrame):
            print(f"[WARN] Colonne dupliquée détectée '{col}' ({x.shape[1]} occurrences) {where} -> on prend la 1ère.")
            x = x.iloc[:, 0]
        return x

    def _ensure_patient_col(df_: pd.DataFrame) -> str:
        """Trouve une colonne patient plausible; sinon raise."""
        for cand in ["patient", "Patient", "patient_id", "subject", "sub"]:
            if cand in df_.columns:
                return cand
        raise ValueError(
            "[plot_cohort_by_soz_group] Impossible de trouver une colonne patient. "
            f"Colonnes dispo: {list(df_.columns)}"
        )

    # -----------------------
    # 1) Compute cohort patient scores
    # -----------------------
    cohort = cohort_patient_scores(
        df,
        aggregate=aggregate,
        k=k,
        min_seizures=min_seizures,
        zero_rel=zero_rel,
        zero_abs_thr=zero_abs_thr,
        stability_alpha=stability_alpha,
        amp_alpha=amp_alpha,
        keep_top_patients_per_feature=keep_top_patients_per_feature,
        multiply_by_n_selected=multiply_by_n_selected,
    )

    if cohort is None or cohort.empty:
        print("[WARN] Cohorte vide (pas assez de données / patients).")
        return cohort

    # colonne patient (normalement "patient")
    pcol = _ensure_patient_col(cohort)

    # -----------------------
    # 2) Build cohort2 + patient_key + soz_group
    # -----------------------
    cohort2 = cohort.copy()
    cohort2[pcol] = cohort2[pcol].astype(str)

    # crée toujours patient_key (même si existait pas)
    cohort2["patient_key"] = cohort2[pcol].map(normalize_patient_id)

    # map SOZ
    soz_map = build_soz_group_map()
    cohort2["soz_group"] = cohort2["patient_key"].map(lambda k: soz_map.get(k, "unknown")).astype(str)

    # -----------------------
    # 3) score_col / sort_col robustes (colonnes dupliquées)
    # -----------------------
    if not isinstance(score_col, str):
        raise TypeError(f"score_col doit être une string, reçu {type(score_col)}: {score_col}")
    if not isinstance(sort_col, str):
        raise TypeError(f"sort_col doit être une string, reçu {type(sort_col)}: {sort_col}")

    # force numeric (via series 1D)
    cohort2[score_col] = pd.to_numeric(_get_1d_series(cohort2, score_col, where="(cohort2)"), errors="coerce")
    cohort2[sort_col] = pd.to_numeric(_get_1d_series(cohort2, sort_col, where="(cohort2)"), errors="coerce")

    outcsv = os.path.join(outdir, "cohort_patient_scores_with_soz_group.csv")
    cohort2.to_csv(outcsv, index=False)
    print(f"[OK] CSV cohorte (+soz_group) sauvegardé → {outcsv}")

    # -----------------------
    # 4) Sort + df_plot
    # -----------------------
    cohort_sorted = cohort2.sort_values(sort_col, ascending=True, na_position="last").reset_index(drop=True)

    cols = [pcol, "patient_key", "soz_group", score_col, sort_col]
    cols = list(dict.fromkeys(cols))  # IMPORTANT: évite score_col==sort_col -> colonne dupliquée
    df_plot = cohort_sorted[cols].copy()

    vals_1d = _get_1d_series(df_plot, score_col, where="(df_plot)")
    df_plot = df_plot[np.isfinite(pd.to_numeric(vals_1d, errors="coerce").to_numpy(dtype=float))].copy()


    # -----------------------
    # 5) Groups for boxplot
    # -----------------------
    if group_order is None:
        group_order = ["temporal", "medial_temporal", "frontal", "insular", "parietal", "mixed", "unknown"]

    present_groups = [g for g in group_order if g in set(df_plot["soz_group"].unique())]
    for g in sorted(set(df_plot["soz_group"].unique()) - set(present_groups)):
        present_groups.append(g)

    box_df = df_plot.copy()
    if drop_unknown_soz_in_boxplot:
        box_df = box_df[box_df["soz_group"] != "unknown"].copy()

    box_groups = [g for g in present_groups if g in set(box_df["soz_group"].unique())]
    if drop_unknown_soz_in_boxplot:
        box_groups = [g for g in box_groups if g != "unknown"]

    # -----------------------
    # 6) Kruskal-Wallis global test (optional)
    # -----------------------
    def _kruskal_pvalue(groups_vals: list[np.ndarray]):
        try:
            from scipy.stats import kruskal
            res = kruskal(*groups_vals)
            return float(res.pvalue), "kruskal_scipy"
        except Exception:
            return np.nan, "no_test"

    group_arrays = []
    for g in box_groups:
        arr = box_df.loc[box_df["soz_group"] == g, score_col].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            group_arrays.append(arr)

    if len(group_arrays) >= 2:
        p_kw, kw_method = _kruskal_pvalue(group_arrays)
    else:
        p_kw, kw_method = np.nan, "insufficient_groups"

    # -----------------------
    # 7) Plot 1x2
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"Cohorte — score coloré par SOZ implant localization\n"
        f"score={score_col} | tri={sort_col}\n"
        f"aggregate={aggregate}, k={k}, min_seizures={min_seizures}, "
        f"keep_top_patients_per_feature={keep_top_patients_per_feature}, "
        f"multiply_by_n_selected={multiply_by_n_selected}, "
        f"stability_alpha={stability_alpha}, amp_alpha={amp_alpha}",
        y=0.98,
        fontsize=14,
    )

    # LEFT barplot
    ax0 = axes[0]


    patients_loc = df_plot[pcol].astype(str).tolist()
    vals = _get_1d_series(df_plot, score_col, where="(barplot)").to_numpy(dtype=float)
    y = np.arange(len(patients_loc))
    colors = [SOZ_COLORS.get(g, "tab:gray") for g in df_plot["soz_group"].astype(str).tolist()]
    ax0.barh(y, vals, alpha=0.9, color=colors)


    ax0.set_yticks(y)
    ax0.set_yticklabels(patients_loc)
    ax0.invert_yaxis()

    xmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    ax0.set_xlim(0, xmax * 1.15 if xmax > 0 else 1.0)

    ax0.set_title("Barplot — tous patients (triés)")
    ax0.set_xlabel(score_col)
    ax0.grid(True, axis="x", alpha=0.3)

    # RIGHT boxplot
    ax1 = axes[1]
    rng = np.random.default_rng(int(random_seed))

    data, labels, facecolors, ns = [], [], [], []
    for g in box_groups:
        arr = box_df.loc[box_df["soz_group"] == g, score_col].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        data.append(arr)
        labels.append(g)
        facecolors.append(SOZ_COLORS.get(g, "tab:gray"))
        ns.append(int(arr.size))

    if len(data) == 0:
        ax1.text(
            0.5, 0.5,
            "No SOZ groups with valid data\n(after filtering unknown?)",
            transform=ax1.transAxes,
            ha="center", va="center",
            alpha=0.85,
        )
        ax1.axis("off")
    else:
        bp = ax1.boxplot(
            data,
            labels=[f"{lab}\n(n={n})" for lab, n in zip(labels, ns)],
            showfliers=False,
            patch_artist=True,
            widths=0.55,
        )
        for b, c in zip(bp["boxes"], facecolors):
            b.set_facecolor(c)
            b.set_alpha(0.30)

        if show_points:
            for i, (arr, c) in enumerate(zip(data, facecolors), start=1):
                jitter = rng.normal(0, 0.06, size=arr.size)
                ax1.scatter(
                    np.full(arr.size, i, dtype=float) + jitter,
                    arr,
                    alpha=0.85,
                    s=int(point_size),
                    color=c,
                    edgecolors="none",
                )

        title = "Boxplot — par SOZ implant localization"
        if np.isfinite(p_kw):
            title += f"\nKruskal–Wallis p={p_kw:.3g}"
        else:
            title += f"\nKruskal–Wallis: {kw_method}"
        ax1.set_title(title)
        ax1.set_ylabel(score_col)
        ax1.grid(True, axis="y", alpha=0.3)

    # legend
    legend_groups = [g for g in present_groups if g in SOZ_COLORS]
    legend_handles = [Patch(facecolor=SOZ_COLORS.get(g, "tab:gray"), label=g, alpha=0.6) for g in legend_groups]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(6, max(2, len(legend_handles))), frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    outpng = os.path.join(outdir, f"cohort_by_soz_group__{score_col}.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure cohorte (SOZ colors + boxplot) sauvegardée → {outpng}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return cohort2



def _get_1d_series(df: pd.DataFrame, col: str, *, name_for_logs: str = "") -> pd.Series:
    """
    Retourne une Series 1D même si df a des colonnes dupliquées.
    Si col apparaît plusieurs fois, prend la 1ère occurrence.
    """
    if col not in df.columns:
        raise ValueError(f"Colonne absente: {col}")

    x = df.loc[:, col]   # peut être Series ou DataFrame (si colonnes dupliquées)
    if isinstance(x, pd.DataFrame):
        # colonnes dupliquées -> on prend la première
        print(f"[WARN] Colonne dupliquée détectée '{col}' ({x.shape[1]} occurrences). "
              f"On utilise la première. {name_for_logs}".strip())
        x = x.iloc[:, 0]
    return x


def plot_bce_and_f1_by_soz_group(
    outdir: str,
    # inputs
    bce_good: str | None = None,
    bce_bad: str | None = None,
    bce_agg: str = "median",
    bce_drop_unknown_seizure: bool = True,
    f1_good: str | None = None,
    f1_bad: str | None = None,
    f1_agg: str = "max",
    # plot options
    drop_unknown_soz: bool = True,
    group_order: list[str] | None = None,
    show_points: bool = True,
    point_size: int = 28,
    random_seed: int = 0,
    noshow: bool = False,
    # stats options
    do_pairwise_posthoc: bool = False,
):
    """
    Figure 1x2:
      - gauche: boxplots BCE (par SOZ implant localization)
      - droite: boxplots F1@top10pct (par SOZ implant localization)

    Dépendances attendues dans ton script:
      - normalize_patient_id
      - build_soz_group_map
      - SOZ_COLORS
      - load_two_bce_csvs, aggregate_bce_per_patient
      - load_two_f1_csvs, aggregate_f1_per_patient
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(int(random_seed))

    if group_order is None:
        group_order = ["temporal", "medial_temporal", "frontal", "insular", "parietal", "mixed", "unknown"]

    # -----------------------
    # Helpers stats
    # -----------------------
    def _kruskal_pvalue(groups_vals: list[np.ndarray]):
        try:
            from scipy.stats import kruskal
            res = kruskal(*groups_vals)
            return float(res.pvalue), "kruskal_scipy"
        except Exception:
            return np.nan, "no_scipy"

    def _mann_whitney_p(x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, float); y = np.asarray(y, float)
        x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
        if x.size < 2 or y.size < 2:
            return np.nan
        try:
            from scipy.stats import mannwhitneyu
            res = mannwhitneyu(x, y, alternative="two-sided")
            return float(res.pvalue)
        except Exception:
            return np.nan

    def _holm_correction(pvals: list[float]) -> list[float]:
        """Holm-Bonferroni, retourne p_adj alignées sur l'ordre d'entrée."""
        p = np.asarray(pvals, float)
        m = p.size
        order = np.argsort(p)
        p_sorted = p[order]
        adj_sorted = np.empty_like(p_sorted)

        # Holm: adj_i = max_{j<=i} ( (m-j)*p_j )
        running_max = 0.0
        for i in range(m):
            factor = (m - i)
            val = factor * p_sorted[i]
            running_max = max(running_max, val)
            adj_sorted[i] = min(1.0, running_max)

        adj = np.empty_like(adj_sorted)
        adj[order] = adj_sorted
        return adj.tolist()

    def _add_boxplot(ax, df_vals: pd.DataFrame, value_col: str, title: str, ylabel: str):
        """
        df_vals columns: patient_key, patient(optional), soz_group, value_col
        """
        # filter unknown group if requested
        d = df_vals.copy()
        if drop_unknown_soz:
            d = d[d["soz_group"] != "unknown"].copy()

        # keep only finite values
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        d = d[np.isfinite(d[value_col].to_numpy(float))].copy()
        if d.empty:
            ax.text(0.5, 0.5, "No data after filtering", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.85)
            ax.axis("off")
            return {"p_kw": np.nan, "kw_method": "no_data", "posthoc": None}

        # decide group order present
        present = [g for g in group_order if g in set(d["soz_group"].unique())]
        for g in sorted(set(d["soz_group"].unique()) - set(present)):
            present.append(g)

        # build per-group arrays
        data, labels, facecolors, ns = [], [], [], []
        for g in present:
            arr = d.loc[d["soz_group"] == g, value_col].to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            data.append(arr)
            labels.append(g)
            facecolors.append(SOZ_COLORS.get(g, "tab:gray"))
            ns.append(int(arr.size))

        if len(data) == 0:
            ax.text(0.5, 0.5, "No SOZ groups with valid data", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.85)
            ax.axis("off")
            return {"p_kw": np.nan, "kw_method": "no_groups", "posthoc": None}

        # global Kruskal test
        p_kw, kw_method = (np.nan, "insufficient_groups")
        if len(data) >= 2:
            p_kw, kw_method = _kruskal_pvalue(data)

        # plot box
        bp = ax.boxplot(
            data,
            labels=[f"{lab}\n(n={n})" for lab, n in zip(labels, ns)],
            showfliers=False,
            patch_artist=True,
            widths=0.58,
        )
        for b, c in zip(bp["boxes"], facecolors):
            b.set_facecolor(c)
            b.set_alpha(0.30)

        # points
        if show_points:
            for i, (arr, c) in enumerate(zip(data, facecolors), start=1):
                jitter = rng.normal(0, 0.06, size=arr.size)
                ax.scatter(
                    np.full(arr.size, i, dtype=float) + jitter,
                    arr,
                    alpha=0.85,
                    s=int(point_size),
                    color=c,
                    edgecolors="none",
                )

        # titles
        t = title
        if np.isfinite(p_kw):
            t += f"\nKruskal–Wallis p={p_kw:.3g}"
        else:
            t += f"\nKruskal–Wallis: {kw_method}"
        ax.set_title(t, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

        # optional post-hoc
        posthoc = None
        if do_pairwise_posthoc and len(data) >= 2:
            pairs = []
            pvals = []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    p = _mann_whitney_p(data[i], data[j])
                    if np.isfinite(p):
                        pairs.append((labels[i], labels[j]))
                        pvals.append(p)

            if pvals:
                p_adj = _holm_correction(pvals)
                posthoc = pd.DataFrame({
                    "group_a": [a for a, b in pairs],
                    "group_b": [b for a, b in pairs],
                    "p_raw": pvals,
                    "p_holm": p_adj,
                }).sort_values("p_holm", ascending=True).reset_index(drop=True)

        return {"p_kw": p_kw, "kw_method": kw_method, "posthoc": posthoc}

    # -----------------------
    # Load + aggregate BCE
    # -----------------------
    bce_vals = None
    if (bce_good is not None) or (bce_bad is not None):
        bce_raw = load_two_bce_csvs(bce_good, bce_bad)
        bce_pat = aggregate_bce_per_patient(
            bce_raw,
            agg=bce_agg,
            drop_unknown_seizure=bce_drop_unknown_seizure,
        )
        # ensure patient_key + soz_group
        bce_pat["patient_key"] = bce_pat["patient_key"].astype(str)
        soz_map = build_soz_group_map()
        bce_pat["soz_group"] = bce_pat["patient_key"].map(lambda k: soz_map.get(k, "unknown")).astype(str)
        bce_vals = bce_pat[["patient_key", "patient", "soz_group", "bce_value"]].copy()
    else:
        bce_vals = pd.DataFrame(columns=["patient_key", "patient", "soz_group", "bce_value"])

    # -----------------------
    # Load + aggregate F1
    # -----------------------
    f1_vals = None
    if (f1_good is not None) or (f1_bad is not None):
        f1_raw = load_two_f1_csvs(f1_good, f1_bad)
        f1_pat = aggregate_f1_per_patient(f1_raw, agg=f1_agg)
        f1_pat["patient_key"] = f1_pat["patient_key"].astype(str)
        soz_map = build_soz_group_map()
        f1_pat["soz_group"] = f1_pat["patient_key"].map(lambda k: soz_map.get(k, "unknown")).astype(str)
        f1_vals = f1_pat[["patient_key", "patient", "soz_group", "f1_value"]].copy()
    else:
        f1_vals = pd.DataFrame(columns=["patient_key", "patient", "soz_group", "f1_value"])

    # -----------------------
    # Plot figure 1x2
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Patient-level performance by SOZ implant localization\n"
        f"BCE agg={bce_agg} | F1 agg={f1_agg} | drop_unknown_soz={drop_unknown_soz}",
        y=0.98,
        fontsize=14,
    )

    res_bce = _add_boxplot(
        axes[0],
        bce_vals,
        value_col="bce_value",
        title="Model 1 performance (BCE) by SOZ group",
        ylabel="BCE (patient-level)",
    )

    res_f1 = _add_boxplot(
        axes[1],
        f1_vals,
        value_col="f1_value",
        title="Model 2 performance (F1@top 10%) by SOZ group",
        ylabel="F1@top_10pct (patient-level)",
    )

    # legend
    legend_groups = [g for g in group_order if g in SOZ_COLORS]
    if drop_unknown_soz and "unknown" in legend_groups:
        legend_groups = [g for g in legend_groups if g != "unknown"]
    handles = [Patch(facecolor=SOZ_COLORS.get(g, "tab:gray"), label=g, alpha=0.6) for g in legend_groups]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, max(2, len(handles))), frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    outpng = os.path.join(outdir, "bce_and_f1_by_soz_group.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure sauvegardée → {outpng}")

    # save tidy CSVs
    bce_csv = os.path.join(outdir, "bce_by_soz_group_patient_level.csv")
    f1_csv = os.path.join(outdir, "f1_by_soz_group_patient_level.csv")
    bce_vals.to_csv(bce_csv, index=False)
    f1_vals.to_csv(f1_csv, index=False)
    print(f"[OK] CSV BCE sauvegardé → {bce_csv}")
    print(f"[OK] CSV F1 sauvegardé → {f1_csv}")

    # save posthoc if asked
    if do_pairwise_posthoc:
        if res_bce.get("posthoc") is not None:
            outp = os.path.join(outdir, "posthoc_pairwise_bce_by_soz_group.csv")
            res_bce["posthoc"].to_csv(outp, index=False)
            print(f"[OK] Posthoc BCE sauvegardé → {outp}")
        if res_f1.get("posthoc") is not None:
            outp = os.path.join(outdir, "posthoc_pairwise_f1_by_soz_group.csv")
            res_f1["posthoc"].to_csv(outp, index=False)
            print(f"[OK] Posthoc F1 sauvegardé → {outp}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return {"bce": res_bce, "f1": res_f1}


def plot_outcome_proportions_and_metric_by_soz_group(
    df_features: pd.DataFrame,
    outdir: str,
    # metric on the RIGHT panel
    right_metric: str = "f1",  # "f1" or "bce"
    f1_good: str | None = None,
    f1_bad: str | None = None,
    f1_agg: str = "max",
    bce_good: str | None = None,
    bce_bad: str | None = None,
    bce_agg: str = "median",
    bce_drop_unknown_seizure: bool = True,
    # plotting options
    group_order: list[str] | None = None,
    drop_unknown_soz_in_right: bool = True,
    include_unknown_outcome_in_left: bool = False,  # if False: only good/bad in proportions
    show_points: bool = True,
    point_size: int = 28,
    random_seed: int = 0,
    noshow: bool = False,
):
    """
    LEFT: stacked barplot per SOZ group = proportions of outcome (good vs bad [+optional unknown])
    RIGHT: boxplot per SOZ group of patient-level metric (F1 or BCE) + jittered points
           + Kruskal–Wallis p-value across groups (if >=2 groups have data).

    Depends on:
      - normalize_patient_id
      - build_soz_group_map
      - SOZ_COLORS
      - OUTCOME_COLORS
      - load_two_f1_csvs, aggregate_f1_per_patient
      - load_two_bce_csvs, aggregate_bce_per_patient
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(int(random_seed))

    if group_order is None:
        group_order = ["temporal", "medial_temporal", "frontal", "insular", "parietal", "mixed", "unknown"]

    # -----------------------
    # Kruskal helper
    # -----------------------
    def _kruskal_pvalue(groups_vals: list[np.ndarray]):
        """
        Kruskal–Wallis p-value.
        SciPy si dispo; sinon retourne NaN.
        """
        try:
            from scipy.stats import kruskal
            res = kruskal(*groups_vals)
            return float(res.pvalue), "kruskal_scipy"
        except Exception:
            return np.nan, "no_scipy"

    # -----------------------
    # Build patient -> outcome mapping from df_features
    # -----------------------
    tmp = df_features.copy()
    if "patient" not in tmp.columns or "outcome" not in tmp.columns:
        raise ValueError(
            "df_features doit contenir les colonnes: patient, outcome (tes CSV mean_change_metrics_per_seizure)"
        )

    tmp["patient"] = tmp["patient"].astype(str)
    tmp["patient_key"] = tmp["patient"].map(normalize_patient_id)

    outcome_map = (
        tmp.groupby("patient_key")["outcome"]
        .agg(lambda x: x.iloc[0] if x.nunique() == 1 else "unknown")
        .to_dict()
    )

    # SOZ mapping
    soz_map = build_soz_group_map()
    patient_soz = (
        pd.DataFrame({"patient_key": sorted(outcome_map.keys())})
        .assign(
            outcome=lambda d: d["patient_key"].map(lambda k: outcome_map.get(k, "unknown")).astype(str),
            soz_group=lambda d: d["patient_key"].map(lambda k: soz_map.get(k, "unknown")).astype(str),
        )
    )

    # -----------------------
    # LEFT panel: proportions good/bad (optionally unknown)
    # -----------------------
    left_df = patient_soz.copy()

    # Option: exclude unknown outcomes from proportion denominator (default)
    if not include_unknown_outcome_in_left:
        left_df = left_df[left_df["outcome"].isin(["good", "bad"])].copy()

    # order groups present
    present_groups = [g for g in group_order if g in set(left_df["soz_group"].unique())]
    for g in sorted(set(left_df["soz_group"].unique()) - set(present_groups)):
        present_groups.append(g)

    # counts per (group, outcome)
    ct = (
        left_df.groupby(["soz_group", "outcome"])["patient_key"]
        .nunique()
        .unstack("outcome")
        .fillna(0.0)
    )

    # ensure columns exist
    for col in ["good", "bad", "unknown"]:
        if col not in ct.columns:
            ct[col] = 0.0

    ct = ct.reindex(index=present_groups)

    denom = ct.sum(axis=1).replace(0, np.nan)
    prop = ct.div(denom, axis=0).fillna(0.0)

    # -----------------------
    # RIGHT panel: load metric (F1 or BCE) and add soz_group
    # -----------------------
    def _prepare_metric_df(metric: str) -> pd.DataFrame:
        metric = (metric or "f1").lower()
        if metric == "f1":
            if (f1_good is None) and (f1_bad is None):
                return pd.DataFrame(columns=["patient_key", "patient", "value", "soz_group"])
            raw = load_two_f1_csvs(f1_good, f1_bad)
            pat = aggregate_f1_per_patient(raw, agg=f1_agg)
            pat["patient_key"] = pat["patient_key"].astype(str)
            pat["value"] = pd.to_numeric(pat["f1_value"], errors="coerce")
        elif metric == "bce":
            if (bce_good is None) and (bce_bad is None):
                return pd.DataFrame(columns=["patient_key", "patient", "value", "soz_group"])
            raw = load_two_bce_csvs(bce_good, bce_bad)
            pat = aggregate_bce_per_patient(raw, agg=bce_agg, drop_unknown_seizure=bce_drop_unknown_seizure)
            pat["patient_key"] = pat["patient_key"].astype(str)
            pat["value"] = pd.to_numeric(pat["bce_value"], errors="coerce")
        else:
            raise ValueError("right_metric doit être 'f1' ou 'bce'.")

        pat["soz_group"] = pat["patient_key"].map(lambda k: soz_map.get(k, "unknown")).astype(str)
        return pat[["patient_key", "patient", "value", "soz_group"]].copy()

    metric_df = _prepare_metric_df(right_metric)

    # -----------------------
    # Plot 1x2
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Surgical outcome composition and Model 2 performance by SOZ implant localization.",
        y=0.98,
        fontsize=18,
    )

    # LEFT: stacked bar proportions
    ax0 = axes[0]
    x = np.arange(len(prop.index))
    bottoms = np.zeros(len(prop.index), dtype=float)

    # choose which outcome stacks to show
    stacks = ["good", "bad"] + (["unknown"] if include_unknown_outcome_in_left else [])
    for o in stacks:
        vals = prop[o].to_numpy(float)
        ax0.bar(
            x,
            vals,
            bottom=bottoms,
            alpha=0.9,
            color=OUTCOME_COLORS.get(o, "tab:gray"),
            label=o,
        )
        bottoms += vals

    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{g}\n(n={int(ct.loc[g].sum())})" for g in prop.index], rotation=0, fontsize=10)
    ax0.set_ylim(0, 1.0)
    ax0.set_ylabel("Proportion of patients")
    ax0.set_title("Surgical outcome distribution by SOZ implantation group")
    ax0.grid(True, axis="y", alpha=0.25)

    # RIGHT: boxplot metric by soz
    ax1 = axes[1]
    d = metric_df.copy()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d[np.isfinite(d["value"].to_numpy(float))].copy()

    if drop_unknown_soz_in_right:
        d = d[d["soz_group"] != "unknown"].copy()

    if d.empty:
        ax1.text(
            0.5, 0.5, "No metric data after filtering",
            transform=ax1.transAxes, ha="center", va="center", alpha=0.85
        )
        ax1.axis("off")
    else:
        groups_right = [g for g in group_order if g in set(d["soz_group"].unique())]
        for g in sorted(set(d["soz_group"].unique()) - set(groups_right)):
            groups_right.append(g)

        data, labels, facecolors, ns = [], [], [], []
        for g in groups_right:
            arr = d.loc[d["soz_group"] == g, "value"].to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            data.append(arr)
            labels.append(g)
            facecolors.append(SOZ_COLORS.get(g, "tab:gray"))
            ns.append(int(arr.size))

        # ---- Kruskal–Wallis across groups
        p_kw, kw_method = np.nan, "insufficient_groups"
        if len(data) >= 2:
            p_kw, kw_method = _kruskal_pvalue(data)

        bp = ax1.boxplot(
            data,
            labels=[f"{lab}\n(n={n})" for lab, n in zip(labels, ns)],
            showfliers=False,
            patch_artist=True,
            widths=0.58,
        )
        for b, c in zip(bp["boxes"], facecolors):
            b.set_facecolor(c)
            b.set_alpha(0.30)

        if show_points:
            for i, (arr, c) in enumerate(zip(data, facecolors), start=1):
                jitter = rng.normal(0, 0.06, size=arr.size)
                ax1.scatter(
                    np.full(arr.size, i, dtype=float) + jitter,
                    arr,
                    alpha=0.85,
                    s=int(point_size),
                    color=c,
                    edgecolors="none",
                )

        metric_title = "F1@top_10pct" if right_metric.lower() == "f1" else "BCE"

        title_right = f"Model 2 SOZ channel ranking performance ({metric_title}) by SOZ implantation group"
        if np.isfinite(p_kw):
            title_right += f"\nKruskal–Wallis p={p_kw:.3g}"
        else:
            title_right += f"\nKruskal–Wallis: {kw_method}"

        ax1.set_title(title_right)
        ax1.set_ylabel(metric_title)
        ax1.grid(True, axis="y", alpha=0.25)

        if kw_method != "kruskal_scipy":
            ax1.text(
                0.02, 0.02,
                f"test={kw_method}",
                transform=ax1.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                alpha=0.85,
            )

    # Legends
    # left legend: outcomes
    handles_out = [Patch(facecolor=OUTCOME_COLORS.get(o, "tab:gray"), label=o, alpha=0.8) for o in stacks]
    # right legend: soz colors
    legend_groups = [g for g in group_order if g in SOZ_COLORS]
    if drop_unknown_soz_in_right and "unknown" in legend_groups:
        legend_groups = [g for g in legend_groups if g != "unknown"]
    handles_soz = [Patch(facecolor=SOZ_COLORS.get(g, "tab:gray"), label=g, alpha=0.6) for g in legend_groups]

    fig.legend(
        handles=handles_out,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.01),
        ncol=len(handles_out),
        frameon=False,
        fontsize = 12
    )
    fig.legend(
        handles=handles_soz,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.01),
        ncol=min(6, len(handles_soz)),
        frameon=False,
        fontsize = 12
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    tag = "f1" if right_metric.lower() == "f1" else "bce"
    outpng = os.path.join(outdir, f"outcome_props_and_{tag}_by_soz_group.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure sauvegardée → {outpng}")

    # save CSVs
    prop_csv = os.path.join(outdir, "outcome_proportions_by_soz_group.csv")
    prop.reset_index().rename(columns={"soz_group": "group"}).to_csv(prop_csv, index=False)
    print(f"[OK] CSV proportions sauvegardé → {prop_csv}")

    metric_csv = os.path.join(outdir, f"{tag}_patient_level_with_soz_group.csv")
    metric_df.to_csv(metric_csv, index=False)
    print(f"[OK] CSV metric sauvegardé → {metric_csv}")

    if not noshow:
        plt.show()
    else:
        plt.close(fig)

    return {"proportions": prop, "counts": ct, "metric_df": metric_df}





def main():
    parser = argparse.ArgumentParser(
        description="Plot patient curves + feature scores + optional cohort comparison (2 CSV good/bad supported)"
    )
    parser.add_argument("--umap_plot", action="store_true", help="UMAP/PCA embedding 2D des patients depuis final_score par feature")
    parser.add_argument("--umap_ycol", default="change_soz_amp_abs", choices=CHANGE_COLS)
    parser.add_argument("--umap_n_features", type=int, default=15)
    parser.add_argument("--umap_feature_select", default="most_common", choices=["most_common", "top_global_score"])
    parser.add_argument("--umap_impute", default="zero", choices=["zero", "median"])
    parser.add_argument("--umap_neighbors", type=int, default=20)
    parser.add_argument("--umap_min_dist", type=float, default=0.15)
    parser.add_argument("--umap_metric", default="cosine", choices=["euclidean", "cosine"])
    parser.add_argument("--umap_annotate", action="store_true")

    # Input CSVs
    parser.add_argument("--csv", default=None, help="CSV unique contenant tous les patients (outcome=unknown)")
    parser.add_argument("--csv_good", default=None, help="CSV des patients 'good outcome'")
    parser.add_argument("--csv_bad", default=None, help="CSV des patients 'bad outcome'")

    # Outputs / patient
    parser.add_argument("--patient", default=None, help="Nom du patient (ex: Patient_01). Si omis: pas de figure patient.")
    parser.add_argument("--outdir", default="figures", help="Dossier de sortie (default: ./figures)")
    parser.add_argument("--noshow", action="store_true", help="Ne pas afficher les figures (mode batch)")
    parser.add_argument("--topn", type=int, default=12, help="Top N features à afficher (default: 12)")

    # Cohorte
    parser.add_argument("--cohort", action="store_true", help="Générer aussi la figure de comparaison entre patients")
    parser.add_argument("--aggregate", default="sum", choices=["sum", "max", "topk", "mean"], help="Agrégation feature->patient (default: sum).")
    parser.add_argument("--k", type=int, default=15, help="k pour l'agrégation topk (default: 15)")
    parser.add_argument("--min_seizures", type=int, default=3, help="Minimum de crises par patient (default: 3)")

    # hyperparams score (tuning)
    parser.add_argument("--zero_rel", type=float, default=0.05)
    parser.add_argument("--zero_abs_thr", type=float, default=1e-8)
    parser.add_argument("--stability_alpha", type=float, default=1.5)
    parser.add_argument("--amp_alpha", type=float, default=0.7)

    # Top-N + multiplication
    parser.add_argument(
        "--keep_top_patients_per_feature",
        type=int,
        default=None,
        help="Si défini (ex: 10), ne garde une feature pour un patient que si ce patient est dans le top-N patients pour cette feature.",
    )
    parser.add_argument(
        "--multiply_by_n_selected",
        action="store_true",
        help="Si activé, multiplie le score patient par le nb de features sélectionnées (top-N) pour ce patient.",
    )

    # BCE
    parser.add_argument("--bce_good", default=None, help="CSV BCE pour patients good outcome")
    parser.add_argument("--bce_bad", default=None, help="CSV BCE pour patients bad outcome")
    parser.add_argument("--bce_agg", default="median", choices=["median", "mean", "min", "max"], help="Agrégation BCE par patient (default: median)")
    parser.add_argument("--bce_drop_unknown_seizure", action="store_true", help="Drop les lignes BCE avec seizure_id == '?'")
    parser.add_argument("--bce_plot", action="store_true", help="Génère la figure BCE vs final_score(change_global_abs)")
    parser.add_argument("--bce_annotate", action="store_true", help="Annoter les points avec le nom patient")

    # F1
    parser.add_argument("--f1_good", default=None, help="CSV F1 pour patients good outcome")
    parser.add_argument("--f1_bad", default=None, help="CSV F1 pour patients bad outcome")
    parser.add_argument("--f1_agg", default="max", choices=["max", "mean", "median"], help="Agrégation F1@top_10pct par patient (default: max)")
    parser.add_argument("--f1_plot", action="store_true", help="Génère la figure F1@top_10pct vs final_score")
    parser.add_argument("--f1_annotate", action="store_true", help="Annoter les points avec le nom patient")

    # Feature subscores plots
    parser.add_argument("--feature_subscores_plot", action="store_true", help="Génère 1 figure par feature avec les subscores par patient")
    parser.add_argument(
        "--feature_subscores_ycol",
        default="change_soz_amp_abs",
        choices=CHANGE_COLS,
        help="Quelle colonne change_* utiliser pour les subscores (default: change_soz_amp_abs)",
    )
    parser.add_argument("--umap2_plot", action="store_true", help="UMAP 2 panneaux: color SOZ group + color F1")
    parser.add_argument("--umap2_ycol", default="change_soz_amp_abs", choices=CHANGE_COLS)
    parser.add_argument("--umap2_n_features", type=int, default=15)
    parser.add_argument("--umap2_feature_select", default="most_common", choices=["most_common", "top_global_score"])
    parser.add_argument("--umap2_impute", default="zero", choices=["zero", "median"])
    parser.add_argument("--umap2_neighbors", type=int, default=20)
    parser.add_argument("--umap2_min_dist", type=float, default=0.15)
    parser.add_argument("--umap2_metric", default="cosine", choices=["euclidean", "cosine"])
    parser.add_argument("--umap2_annotate", action="store_true")



    parser.add_argument("--top_features_by_soz", action="store_true")
    parser.add_argument("--top_features_ycol", default="change_global_abs", choices=CHANGE_COLS)
    parser.add_argument("--top_features_topn", type=int, default=15)
    parser.add_argument("--top_features_mode", default="importance", choices=["importance", "median"])
    parser.add_argument("--top_features_min_patients", type=int, default=3)


    args = parser.parse_args()

    # Validation input
    if (args.csv_good or args.csv_bad) and args.csv:
        raise ValueError("Choisis soit --csv, soit (--csv_good/--csv_bad), mais pas les deux en même temps.")
    if not (args.csv or args.csv_good or args.csv_bad):
        raise ValueError("Il faut fournir --csv OU bien --csv_good/--csv_bad.")

    df = load_two_csvs(args.csv_good, args.csv_bad, args.csv)

    cohort_df = None

    if args.cohort:
        cohort_df = plot_cohort(
            df,
            outdir=args.outdir,
            aggregate=args.aggregate,
            k=args.k,
            min_seizures=args.min_seizures,
            noshow=args.noshow,
            zero_rel=args.zero_rel,
            zero_abs_thr=args.zero_abs_thr,
            stability_alpha=args.stability_alpha,
            amp_alpha=args.amp_alpha,
            keep_top_patients_per_feature=args.keep_top_patients_per_feature,
            multiply_by_n_selected=args.multiply_by_n_selected,
        )

    # BCE correlation plot
    if args.bce_plot:
        if cohort_df is None:
            cohort_df = cohort_patient_scores(
                df,
                aggregate=args.aggregate,
                k=args.k,
                min_seizures=args.min_seizures,
                zero_rel=args.zero_rel,
                zero_abs_thr=args.zero_abs_thr,
                stability_alpha=args.stability_alpha,
                amp_alpha=args.amp_alpha,
                keep_top_patients_per_feature=args.keep_top_patients_per_feature,
                multiply_by_n_selected=args.multiply_by_n_selected,
            )

        bce_raw = load_two_bce_csvs(args.bce_good, args.bce_bad)
        bce_pat = aggregate_bce_per_patient(
            bce_raw,
            agg=args.bce_agg,
            drop_unknown_seizure=args.bce_drop_unknown_seizure,
        )

        plot_bce_vs_finalscore_change_global_abs(
            cohort_scores_df=cohort_df,
            bce_patient_df=bce_pat,
            outdir=args.outdir,
            noshow=args.noshow,
            annotate=args.bce_annotate,
        )

    # Patient plot
    if args.patient is not None:
        plot_patient(
            df,
            args.patient,
            outdir=args.outdir,
            noshow=args.noshow,
            topn=args.topn,
            zero_rel=args.zero_rel,
            zero_abs_thr=args.zero_abs_thr,
            stability_alpha=args.stability_alpha,
            amp_alpha=args.amp_alpha,
        )

    # F1 correlation plot
    if args.f1_plot:
        if cohort_df is None:
            cohort_df = cohort_patient_scores(
                df,
                aggregate=args.aggregate,
                k=args.k,
                min_seizures=args.min_seizures,
                zero_rel=args.zero_rel,
                zero_abs_thr=args.zero_abs_thr,
                stability_alpha=args.stability_alpha,
                amp_alpha=args.amp_alpha,
                keep_top_patients_per_feature=args.keep_top_patients_per_feature,
                multiply_by_n_selected=args.multiply_by_n_selected,
            )

        f1_raw = load_two_f1_csvs(args.f1_good, args.f1_bad)
        f1_pat = aggregate_f1_per_patient(f1_raw, agg=args.f1_agg)

        plot_f1_vs_finalscore_logx(
            cohort_scores_df=cohort_df,
            f1_patient_df=f1_pat,
            outdir=args.outdir,
            xcol="patient_score__change_global_abs",
            save_tag="allchange",
            noshow=args.noshow,
            annotate=args.f1_annotate,
            debug=True,
        )

        plot_f1_vs_finalscore_logx(
            cohort_scores_df=cohort_df,
            f1_patient_df=f1_pat,
            outdir=args.outdir,
            xcol="patient_score__change_soz_amp_abs",
            save_tag="SOZchange",
            noshow=args.noshow,
            annotate=args.f1_annotate,
            debug=True,
        )

    if args.feature_subscores_plot:
        scores_long = build_feature_scores_table_all_patients(
            df,
            y_col=args.feature_subscores_ycol,
            eps=1e-12,
            zero_rel=args.zero_rel,
            zero_abs_thr=args.zero_abs_thr,
            stability_alpha=args.stability_alpha,
            amp_alpha=args.amp_alpha,
        )
        """
        plot_subscores_per_feature_across_patients(
            scores_long,
            outdir=os.path.join(args.outdir, "subscores_by_feature"),
            y_col=args.feature_subscores_ycol,
            patients_order_by="final_score",
            show_all_patient_labels=True,
            label_fontsize=6,
            fig_width_per_patient=0.35,
            noshow=args.noshow,
        )
        """



    if args.umap_plot:
        plot_patient_umap_from_feature_scores(
            df,
            outdir=args.outdir,
            y_col=args.umap_ycol,
            n_features=args.umap_n_features,
            feature_select=args.umap_feature_select,
            impute=args.umap_impute,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            noshow=args.noshow,
            annotate=args.umap_annotate,
        )

    if args.umap2_plot:
        plot_umap_soz_and_f1_side_by_side(
            df=df,
            outdir=args.outdir,
            y_col=args.umap2_ycol,
            n_features=args.umap2_n_features,
            feature_select=args.umap2_feature_select,
            impute=args.umap2_impute,
            standardize=True,
            umap_neighbors=args.umap2_neighbors,
            umap_min_dist=args.umap2_min_dist,
            umap_metric=args.umap2_metric,
            f1_good=args.f1_good,
            f1_bad=args.f1_bad,
            f1_agg=args.f1_agg,
            annotate=args.umap2_annotate,
            noshow=args.noshow,
        )

    """
    plot_top_features_by_soz_group(
        df=df,
        outdir=args.outdir,
        y_col=args.top_features_ycol,
        topn=args.top_features_topn,
        score_mode=args.top_features_mode,
        min_patients_per_group=args.top_features_min_patients,
        drop_unknown_group=True,
        noshow=args.noshow,
        stability_alpha=args.stability_alpha,
        amp_alpha=args.amp_alpha,
        zero_rel=args.zero_rel,
        zero_abs_thr=args.zero_abs_thr,
    )
    """
    plot_top_features_by_soz_group_with_heatmap(
        df=df,
        outdir=args.outdir,
        y_col="change_global_abs",
        topn_bar=15,
        n_groups_bar=6,                 # <- 3 groupes max
        heatmap_top_features=40,
        score_mode="median",
        min_patients_per_group_feature=3,
        noshow=args.noshow,
        stability_alpha=args.stability_alpha,
        amp_alpha=args.amp_alpha,
        zero_rel=args.zero_rel,
        zero_abs_thr=args.zero_abs_thr,
    )


    plot_cohort_by_soz_group(
        df,
        outdir=args.outdir,
        score_col="patient_score__change_global_abs",
        sort_col="patient_score__change_global_abs",
        aggregate=args.aggregate,
        k=args.k,
        min_seizures=args.min_seizures,
        noshow=args.noshow,
        zero_rel=args.zero_rel,
        zero_abs_thr=args.zero_abs_thr,
        stability_alpha=args.stability_alpha,
        amp_alpha=args.amp_alpha,
        keep_top_patients_per_feature=args.keep_top_patients_per_feature,
        multiply_by_n_selected=args.multiply_by_n_selected,
    )

    plot_bce_and_f1_by_soz_group(
        outdir=args.outdir,
        bce_good=args.bce_good,
        bce_bad=args.bce_bad,
        bce_agg=args.bce_agg,
        bce_drop_unknown_seizure=args.bce_drop_unknown_seizure,
        f1_good=args.f1_good,
        f1_bad=args.f1_bad,
        f1_agg=args.f1_agg,
        drop_unknown_soz=True,
        show_points=True,
        noshow=args.noshow,
        do_pairwise_posthoc=False,  # mets True si tu veux posthoc
    )


    plot_outcome_proportions_and_metric_by_soz_group(
        df_features=df,
        outdir=args.outdir,
        right_metric="f1",      # ou "bce"
        f1_good=args.f1_good,
        f1_bad=args.f1_bad,
        f1_agg=args.f1_agg,
        bce_good=args.bce_good,
        bce_bad=args.bce_bad,
        bce_agg=args.bce_agg,
        bce_drop_unknown_seizure=args.bce_drop_unknown_seizure,
        include_unknown_outcome_in_left=False,  # good/bad seulement dans les proportions
        drop_unknown_soz_in_right=True,
        noshow=args.noshow,
    )



if __name__ == "__main__":
    main()
