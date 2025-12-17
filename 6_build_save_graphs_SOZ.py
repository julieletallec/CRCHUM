# build_pyg_graphs.py
# Convertit les DF master (nodes + edges) en objets PyG (.pt)
# Version "single edge feature": utilise UNE SEULE métrique de connectivité (ex: psi_12_45)

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric est requis. Installe-le: pip install torch-geometric") from e

# =================== CONFIG ===================
ROOT = "//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2"
NODES_PATH = os.path.join(ROOT, "master_node_features_SOZ_normalized_aug_20_10_burst_norm.parquet")
EDGES_PATH = os.path.join(ROOT, "master_edge_features_SOZ_aug_20_10_burst.parquet")
OUT_DIR = os.path.join(ROOT, "gnn", "pyg")

# colonne meta côté nodes
NODE_META_COLS = {
    "dataset","outcome","phase","patient","seizure","epoch","is_avg",
    "electrode","file_path","is_SOZ","first_ictal_epoch"
}
OUTCOME_TO_Y = {"bad outcome": 0, "good outcome": 1, None: -1}


# ---------- Sélection explicite des features noeud ----------
# Si USE_EXPLICIT_NODE_FEATURES = False :
#   -> on garde le comportement actuel : toutes les colonnes numériques
#      (hors NODE_META_COLS) sont utilisées comme features.
# Si USE_EXPLICIT_NODE_FEATURES = True :
#   -> on ne garde que les features listées dans EXPLICIT_NODE_FEATURES
USE_EXPLICIT_NODE_FEATURES = True


EXPLICIT_NODE_FEATURES = [
    'ratio_bg_ta',
    'ratio_gamma_delta',
    'sef95_Hz',
    'spike_sharpness',
    'line_length',
    'tkeo_energy',
    'hg_power_80_150',
    'hg_over_gamma',
    'spec_slope_2_80',
    'spec_intercept_2_80',
    'hjorth_activity',
    'hjorth_mobility',
    'hjorth_complexity',
    'kurtosis',
    'skewness']

EXPLICIT_NODE_FEATURES = [
        "burstint_ratio_bg_ta",
        "burstint_ratio_gamma_delta",
        "burstint_sef95_Hz",
        "burstint_spike_sharpness",
        "burstint_line_length",
        "burstint_tkeo_energy",
        "burstint_hg_power_80_150",
        "burstint_hg_over_gamma",
        "burstint_spec_slope_2_80",
        "burstint_spec_intercept_2_80",
        "burstint_hjorth_activity",
        "burstint_hjorth_mobility",
        "burstint_hjorth_complexity",
        "burstint_kurtosis",
        "burstint_skewness",
    ]

NORMALIZE_NODE_X_PER_GRAPH = False

# ---------- Réglages connectivité (UNE SEULE FEATURE) ----------
# Choisis la colonne de connectivité présente dans master_edge_features.parquet
# Recommandé pour directionnalité: "psi_12_45"
# Recommandé pour non-dirigé (cohésion): "coh_12_45"
EDGE_FEATURE_NAME = "psi_12_45"
# Si abs() doit être appliqué sur la métrique sélectionnée (utile pour corrélation)
EDGE_USE_ABS = False

# Sparsification basée sur CETTE métrique
SPARSIFY_MODE = "knn"  # "knn" | "percentile" | "topk" | "none"
K_OUT = 10              # pour knn
MAX_IN = None           # limite du in-degree (optionnel) pour knn
PERCENTILE_Q = 95.0     # pour percentile
TOPK_GLOBAL = 500       # pour topk global

REMOVE_SELF_LOOPS = True

# Debug
DEBUG = False
NAN_BEHAVIOR = "warn"  # "allow" | "warn" | "skip"
# =====================================================

# ---------------------- Utils ----------------------
def _clean_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in ("dataset", "outcome", "phase", "patient", "electrode"):
        if c in df.columns:
            s = df[c]
            mask = s.notna() & s.map(lambda v: isinstance(v, str))
            if mask.any():
                df.loc[mask, c] = s[mask].str.strip()
    return df

def _load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Format non supporté: {ext}")

def _node_feature_cols_all_features(df_nodes):
    num_cols = [c for c in df_nodes.columns
                if c not in NODE_META_COLS and pd.api.types.is_numeric_dtype(df_nodes[c])]
    if not num_cols:
        raise ValueError("Aucune colonne de features noeud détectée.")
    return num_cols

def _node_feature_cols(df_nodes):
    """
    Retourne la liste des colonnes à utiliser comme features de noeud.

    - Si USE_EXPLICIT_NODE_FEATURES = True:
        On intersecte EXPLICIT_NODE_FEATURES avec les colonnes présentes
        et numériques.
    - Sinon:
        On prend toutes les colonnes numériques qui ne sont pas dans NODE_META_COLS.
    """
    if USE_EXPLICIT_NODE_FEATURES:
        cols = []
        for f in EXPLICIT_NODE_FEATURES:
            if f in df_nodes.columns and pd.api.types.is_numeric_dtype(df_nodes[f]):
                cols.append(f)
        if not cols:
            raise ValueError(
                "Aucune feature de noeud trouvée parmi EXPLICIT_NODE_FEATURES. "
                "Vérifie les noms de colonnes ou mets USE_EXPLICIT_NODE_FEATURES=False."
            )
        print(f"[INFO] Node features (explicit): {cols}")
        return cols
    else:
        num_cols = [
            c for c in df_nodes.columns
            if c not in NODE_META_COLS and pd.api.types.is_numeric_dtype(df_nodes[c])
        ]
        if not num_cols:
            raise ValueError("Aucune colonne de features noeud détectée.")
        print(f"[INFO] Node features (auto numeric): {num_cols}")
        return num_cols


# ---------------------- Sparsification ----------------------
def _knn_directed(S, k_out=10, max_in=None):
    n = S.shape[0]
    keep = np.zeros_like(S, dtype=bool)
    k = min(k_out, max(0, n-1))
    if k > 0:
        idx = np.argpartition(-S, kth=k-1, axis=1)[:, :k]
        rows = np.repeat(np.arange(n)[:, None], idx.shape[1], axis=1)
        # tri des k meilleurs pour stabilité
        ord_cols = np.take_along_axis(S, idx, axis=1).argsort(axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, ord_cols, axis=1)
        keep[rows, idx] = True
    np.fill_diagonal(keep, False)
    if max_in is not None:
        in_deg = keep.sum(axis=0)
        over = np.where(in_deg > max_in)[0]
        for j in over:
            srcs = np.where(keep[:, j])[0]
            if len(srcs) > max_in:
                ord_src = srcs[np.argsort(S[srcs, j])]
                to_drop = ord_src[:len(srcs) - max_in]
                keep[to_drop, j] = False
    return keep

def _quantile_then_fill(S, q=0.95, k_out_min=5):
    n = S.shape[0]
    if n == 0:
        return np.zeros_like(S, dtype=bool)
    mask_off = ~np.eye(n, dtype=bool)
    flat = S[mask_off]
    if flat.size == 0:
        return np.zeros_like(S, dtype=bool)
    thr = np.quantile(flat, q)
    keep = (S >= thr)
    np.fill_diagonal(keep, False)
    # garantit au moins k_out_min sorties par nœud
    for i in range(n):
        need = max(0, k_out_min - keep[i].sum())
        if need:
            # complète avec les plus forts restants
            cand = np.argpartition(-S[i], kth=min(need, n-1)-1)[:min(need, n-1)]
            cand = cand[cand != i]
            keep[i, cand] = True
    return keep

def _topk_global(S, k=500):
    n = S.shape[0]
    mask_off = ~np.eye(n, dtype=bool)
    scores = S[mask_off]
    if scores.size == 0:
        return np.zeros_like(S, dtype=bool)
    k = min(k, scores.size)
    top = np.argpartition(-scores, k-1)[:k]
    keep = np.zeros_like(S, dtype=bool)
    idxs = np.argwhere(mask_off)
    keep[idxs[top, 0], idxs[top, 1]] = True
    return keep

# ---------------------- Build graph ----------------------
def _map_is_soz_to_tensor(df_nodes_g_indexed, node_names):
    arr = np.full((len(node_names), 1), -1, dtype=np.int8)
    if "is_SOZ" not in df_nodes_g_indexed.columns:
        return torch.tensor(arr, dtype=torch.int8)
    ser = df_nodes_g_indexed["is_SOZ"]
    for i, name in enumerate(node_names):
        if name in df_nodes_g_indexed.index:
            v = ser.loc[name]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if isinstance(v, (bool, np.bool_)):
                arr[i, 0] = 1 if bool(v) else 0
            elif pd.isna(v):
                arr[i, 0] = -1
            else:
                sval = str(v).strip().lower()
                if sval in ("true", "1", "yes"):
                    arr[i, 0] = 1
                elif sval in ("false", "0", "no"):
                    arr[i, 0] = 0
                else:
                    arr[i, 0] = -1
    return torch.tensor(arr, dtype=torch.int8)

def _get_first_ictal_epoch_flag(df_nodes_g_indexed):
    if "first_ictal_epoch" not in df_nodes_g_indexed.columns:
        return False
    v = df_nodes_g_indexed["first_ictal_epoch"].iloc[0]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ("true", "1", "yes")


    # Si des arêtes valides existent, tu continues la procédure comme d'habitude
    # Suite de ton code pour traiter les arêtes...

def _build_one_graph(df_nodes_g, df_edges_g, node_feat_cols, edge_feature=EDGE_FEATURE_NAME, use_abs=EDGE_USE_ABS):
    # ordre stable des nœuds
    node_names = sorted(df_nodes_g["electrode"].unique().tolist())
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    df_nodes_g = df_nodes_g.set_index("electrode")

    # === X (features) ===
    X = np.zeros((len(node_names), len(node_feat_cols)), dtype=np.float32)
    rows_present = [n for n in node_names if n in df_nodes_g.index]
    if rows_present:
        sub = df_nodes_g.loc[rows_present, node_feat_cols]
        X_idx = np.array([name_to_idx[n] for n in rows_present], dtype=np.int64)
        X[X_idx] = sub.to_numpy(dtype=np.float32, copy=False)
    if np.isnan(X).any():
        # Localisation exacte des NaN
        nan_locs = np.argwhere(np.isnan(X))
        print("\n[WARN] NaN détectés dans les features noeud :")
        #for (i, j) in nan_locs:
            #print(f" - Node: {node_names[i]}, feature: {node_feat_cols[j]}, "
                #f"value = {df_nodes_g.loc[node_names[i], node_feat_cols[j]]}")

        # Contexte patient / crise
        try:
            print(f"   -> Patient: {df_nodes_g['patient'].iloc[0]}, "
                f"Seizure: {df_nodes_g['seizure'].iloc[0]}, "
                f"Epoch: {df_nodes_g['epoch'].iloc[0]}")
            cols_with_nan = np.where(np.isnan(X).any(axis=0))[0]
            print("\n[WARN] NaN détectés dans les colonnes :", [node_feat_cols[c] for c in cols_with_nan])

        except Exception:
            pass

        if NAN_BEHAVIOR == "skip":
            raise ValueError("NaN détectés dans les features noeud.")




    #if NORMALIZE_NODE_X_PER_GRAPH:
        #mu = X.mean(0, keepdims=True)
        #sd = X.std(0, keepdims=True) + 1e-8
        #X = (X - mu) / sd
    #x = torch.tensor(X, dtype=torch.float32)




    if NORMALIZE_NODE_X_PER_GRAPH:
        # Normalisation Min-Max
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)  # Normalisation des features entre 0 et 1
    x = torch.tensor(X, dtype=torch.float32)




    # === Edges: UNE SEULE métrique ===
    if edge_feature not in df_edges_g.columns:
        # petit fallback si la colonne a un alias typique
        aliases = {
            "psi_12_45": ["psi12_45", "psi1245", "psi"],
            "coh_12_45": ["coh12_45", "coh1245", "coh"],
            "pearson_r": ["corr", "r", "pearson"],
            "psi": ["psi"],
            "granger_f": ["granger", "gc", "granger_stat"]
        }
        found = None
        for cand in aliases.get(edge_feature, []):
            if cand in df_edges_g.columns and pd.api.types.is_numeric_dtype(df_edges_g[cand]):
                found = cand
                break
        if found is None:
            raise ValueError(f"La colonne d’arête demandée '{edge_feature}' est absente du DF edges.")
        edge_col = found
    else:
        edge_col = edge_feature

    # map noms -> indices
    i_idx = df_edges_g["source"].map(name_to_idx)
    j_idx = df_edges_g["target"].map(name_to_idx)
    valid = i_idx.notna() & j_idx.notna() & df_edges_g[edge_col].notna()
    if not valid.any():
        # pas d'arêtes valides
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        outcome = df_nodes_g["outcome"].iloc[0] if "outcome" in df_nodes_g.columns else None
        y_val = OUTCOME_TO_Y.get(outcome, -1)
        y = torch.tensor([y_val], dtype=torch.long)
        node_is_soz = _map_is_soz_to_tensor(df_nodes_g, node_names)
        first_ictal_epoch = _get_first_ictal_epoch_flag(df_nodes_g)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.node_names = node_names
        data.meta = {
            "dataset": df_nodes_g["dataset"].iloc[0],
            "outcome": outcome,
            "phase": df_nodes_g["phase"].iloc[0],
            "patient": df_nodes_g["patient"].iloc[0],
            "seizure": int(df_nodes_g["seizure"].iloc[0]),
            "epoch": int(df_nodes_g["epoch"].iloc[0]),
            "first_ictal_epoch": bool(first_ictal_epoch),
            "n_nodes": len(node_names),
            "n_edges": 0,
            "edge_feature_name": edge_feature,
        }
        data.node_is_soz = node_is_soz
        data.first_ictal_epoch = bool(first_ictal_epoch)
        return data

    dfE = df_edges_g.loc[valid, ["source", "target", edge_col]].copy()
    dfE["_i"] = i_idx.loc[valid].astype(np.int64).values
    dfE["_j"] = j_idx.loc[valid].astype(np.int64).values
    dfE = dfE.drop_duplicates(subset=["_i", "_j"], keep="last")

    n = len(node_names)
    M = np.zeros((n, n), dtype=np.float32)
    vals = dfE[edge_col].astype(np.float32).to_numpy(copy=False)
    if use_abs:
        vals = np.abs(vals)
    M[dfE["_i"].to_numpy(), dfE["_j"].to_numpy()] = vals
    np.fill_diagonal(M, 0.0)

    # === Sparsification avec CETTE matrice M (scores) ===
    if SPARSIFY_MODE == "none":
        keep = ~np.eye(n, dtype=bool)
    elif SPARSIFY_MODE == "knn":
        keep = _knn_directed(M, k_out=K_OUT, max_in=MAX_IN)
    elif SPARSIFY_MODE == "percentile":
        keep = _quantile_then_fill(M, q=PERCENTILE_Q/100.0, k_out_min=K_OUT)
    elif SPARSIFY_MODE == "topk":
        keep = _topk_global(M, k=TOPK_GLOBAL)
    else:
        raise ValueError("SPARSIFY_MODE invalide")

    if REMOVE_SELF_LOOPS:
        np.fill_diagonal(keep, False)

    src, dst = np.where(keep)
    if src.size:
        edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64, copy=False))
        edge_attr  = torch.from_numpy(M[src, dst][:, None].astype(np.float32, copy=False))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float32)

    # === y & meta
    outcome = df_nodes_g["outcome"].iloc[0] if "outcome" in df_nodes_g.columns else None
    y_val = OUTCOME_TO_Y.get(outcome, -1)
    y = torch.tensor([y_val], dtype=torch.long)

    node_is_soz = _map_is_soz_to_tensor(df_nodes_g, node_names)
    first_ictal_epoch = _get_first_ictal_epoch_flag(df_nodes_g)

    meta = {
        "dataset": df_nodes_g["dataset"].iloc[0],
        "outcome": outcome,
        "phase": df_nodes_g["phase"].iloc[0],
        "patient": df_nodes_g["patient"].iloc[0],
        "seizure": int(df_nodes_g["seizure"].iloc[0]),
        "epoch": int(df_nodes_g["epoch"].iloc[0]),
        "first_ictal_epoch": bool(first_ictal_epoch),
        "n_nodes": len(node_names),
        "n_edges": edge_index.shape[1],
        "edge_feature_name": edge_feature,
    }

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.node_names = node_names
    data.meta = meta
    data.node_is_soz = node_is_soz          # int8 [N,1]
    data.first_ictal_epoch = bool(first_ictal_epoch)
    return data

# ---------------------- Main ----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(1)
    nodes, edges = _load_df(NODES_PATH), _load_df(EDGES_PATH)
    print(2)
    nodes, edges = _clean_keys(nodes), _clean_keys(edges)
    print(3)
    # petite clé outcome robuste
    def _mk_outcome_key(s: pd.Series) -> pd.Series:
        # If it's categorical, either add "__NA__" then fill, or just cast to object first.
        if pd.api.types.is_categorical_dtype(s):
            # Option A: keep categorical
            if "__NA__" not in s.cat.categories:
                s = s.cat.add_categories(["__NA__"])
            return s.fillna("__NA__").astype("object")
        # Non-categorical path
        return s.astype("object").where(s.notna(), "__NA__")

    nodes["outcome_key"] = _mk_outcome_key(nodes["outcome"]) if "outcome" in nodes.columns else "__NA__"
    print(4)
    edges["outcome_key"] = _mk_outcome_key(edges["outcome"]) if "outcome" in edges.columns else "__NA__"
    print(5)
    # Filtre epoch != -1 (on prend les epochs "réels", pas les moyennes)
    nodes = nodes[nodes["epoch"].astype(int) != -1]
    print(6)
    edges = edges[edges["epoch"].astype(int) != -1]
    print(7)

    node_feat_cols = _node_feature_cols(nodes)
    key_cols = ["dataset", "outcome_key", "phase", "patient", "seizure", "epoch"]

    # indexation par groupe
    edges_by_key = {k: g for k, g in edges.groupby(key_cols, sort=False, dropna=False)}

    index_rows, n_ok, n_skip = [], 0, 0

    for key_vals, dfN in nodes.groupby(key_cols, sort=False, dropna=False):
        dataset, outcome_key, phase, patient, seizure, epoch = key_vals
        dfE = edges_by_key.get(key_vals, None)
        if dfE is None or dfE.empty:
            if DEBUG:
                print(f"[SKIP] pas d'edges pour {key_vals!r}")
            n_skip += 1
            continue

        try:
            data = _build_one_graph(
                dfN.reset_index(drop=True),
                dfE.reset_index(drop=True),
                node_feat_cols,
                edge_feature=EDGE_FEATURE_NAME,
                use_abs=EDGE_USE_ABS
            )
        except Exception as e:
            print(f"[SKIP] build graph failed for {key_vals}: {e}")
            n_skip += 1
            continue

        out_dir = os.path.join(str(dataset)+"_20_10_burst", str(phase), str(patient))
        os.makedirs(out_dir, exist_ok=True)
        fname = f"graph_{patient}_{seizure}_{epoch}.pt"
        fpath = os.path.join(out_dir, fname)
        torch.save(data, fpath)

        index_rows.append({
            "path": fpath,
            "dataset": dataset,
            "phase": phase,
            "patient": patient,
            "seizure": seizure,
            "epoch": epoch,
            "first_ictal_epoch": data.first_ictal_epoch,
            "n_nodes": int(data.meta["n_nodes"]),
            "n_edges": int(data.meta["n_edges"]),
            "y": int(data.y.item()),
            "edge_feature_name": data.meta["edge_feature_name"],
        })
        n_ok += 1

    if index_rows:
        idx_df = pd.DataFrame(index_rows).sort_values(["dataset", "phase", "patient", "seizure", "epoch"])
        idx_csv = os.path.join("graphs_index.csv")
        idx_df.to_csv(idx_csv, index=False)
        print(f"[OK] Saved {n_ok} graphs. Index: {idx_csv} (skipped: {n_skip})")
    else:
        print("[WARN] Aucun graphe sauvegardé.")

if __name__ == "__main__":
    main()
