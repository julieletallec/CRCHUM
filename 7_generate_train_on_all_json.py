# split_sequences_train_test_patient_specific.py
# - Deux splits par patient: train et test
# - TRAIN (toutes les seizures): fenêtres glissantes + éventuelle séquence centrée onset
#   * chaque séquence conservée doit contenir préictal **et** ictal
# - TEST (toutes les seizures): exactement 1 séquence par seizure
#   * aussi longue que possible
#   * premier ictal (onset) exactement au milieu
#   * la séquence doit contenir préictal **et** ictal
# - CONTRAINTE SPLIT: par patient, chaque split (train **et** test)
#   doit contenir au moins 1 epoch préictal et au moins 1 epoch ictal au total.
#   Sinon on supprime ce patient des sorties.
# - CONTRAINTE PATIENT: ne garder que les patients avec >= MIN_CRISES crises
# - Sorties: splits_patient_train_test.json + splits_patient_train_test_summary.csv

import json
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ========= CONFIG =========
INDEX_CSV = "graphs_index.csv"



# Fenêtres glissantes (TRAIN)
L_TARGET = 10      # longueur visée par séquence
S_TARGET = 1       # stride

# Garder les séquences plus courtes que L_TARGET si crise courte (TRAIN)
KEEP_SHORT = True
MIN_SEQ_EPOCHS = 8   # mini pour garder une séquence courte

# Séquence centrée onset (TRAIN)
ADD_ONSET_SEQ = True
ONSET_PRE_TARGET = 5
ONSET_POST_TARGET = 5
ONSET_MIN_TOTAL = 1

# Séquence centrée onset (TEST)
#CONTROLLED = None
CONTROLLED = (5, 5)


OUT_SPLIT_JSON = f"splits/splits_trainonall_L{L_TARGET}_S{S_TARGET}.json"
OUT_SUMMARY_CSV = f"splits/splits_trainonall_L{L_TARGET}_S{S_TARGET}_summary.csv"



# Exiger que chaque split (train/test) ait au moins 1 préictal et 1 ictal (au total)
MUST_HAVE_BOTH_PHASES_PER_SPLIT = True

# Patients éligibles: au moins 4 crises
MIN_CRISES = 4

# Logs
PRINT_PER_PATIENT_STATS = True
# =========================


def _require_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {INDEX_CSV}: {missing}")


def _paths_ordered_with_trel(df_seiz: pd.DataFrame):
    """
    Timeline t_rel par crise :
      - préictal trié par epoch -> t_rel = -n_pre, ..., -1
      - ictal   trié par epoch -> t_rel = 0, 1, 2, ...
    Retourne:
      paths_sorted (list[str]), onset_idx (int|None), trel_sorted (np.ndarray[int])
    """
    df_pre = df_seiz[df_seiz["phase"].eq("preictal")].sort_values("epoch")
    df_ict = df_seiz[df_seiz["phase"].eq("ictal")].sort_values("epoch")

    n_pre = len(df_pre)
    n_ict = len(df_ict)

    trel_pre = np.arange(-n_pre, 0, dtype=int) if n_pre > 0 else np.array([], dtype=int)
    trel_ict = np.arange(0, n_ict, dtype=int) if n_ict > 0 else np.array([], dtype=int)

    paths, trel = [], []
    if n_pre > 0:
        paths += df_pre["path"].tolist()
        trel  += trel_pre.tolist()
    if n_ict > 0:
        paths += df_ict["path"].tolist()
        trel  += trel_ict.tolist()

    if not paths:
        return [], None, np.array([], dtype=int)

    order = np.argsort(trel)
    paths_sorted = [paths[i] for i in order]
    trel_sorted  = np.asarray(trel, dtype=int)[order]

    onset_pos = np.where(trel_sorted == 0)[0]
    onset_idx = int(onset_pos[0]) if onset_pos.size > 0 else None
    return paths_sorted, onset_idx, trel_sorted


# ---- Phases helpers (robust: match sur segments de chemin) ----
SEP_RE = re.compile(r"[\\/]+")

def _path_has_phase(p: str, phase: str) -> bool:
    parts = [x for x in SEP_RE.split(p.lower()) if x]
    return phase in parts  # segment exact (évite 'ictal' dans 'preictal')

def _seq_has_both_phases(seq_paths: List[str]) -> bool:
    has_pre = any(_path_has_phase(p, "preictal") for p in seq_paths)
    has_ict = any(_path_has_phase(p, "ictal")   for p in seq_paths)
    return has_pre and has_ict

def _paths_list_has_phase(paths: List[str], phase: str) -> bool:
    return any(_path_has_phase(p, phase) for p in paths)

def _split_has_both_phases(seqs: List[List[str]]) -> bool:
    all_paths = (p for seq in seqs for p in seq)
    has_pre = any(_path_has_phase(p, "preictal") for p in all_paths)
    all_paths = (p for seq in seqs for p in seq)  # recréer l'itérateur
    has_ict = any(_path_has_phase(p, "ictal") for p in all_paths)
    return has_pre and has_ict


# ---- Génération de séquences ----
def make_sliding_sequences(paths_sorted: List[str],
                           L_target: int,
                           S_target: int,
                           keep_short: bool,
                           min_short: int) -> List[List[str]]:
    """
    Fenêtres glissantes adaptatives ; on garde uniquement
    celles qui contiennent préictal & ictal.
    """
    out = []
    n = len(paths_sorted)
    if n == 0:
        return out

    L_eff = min(L_target, n)
    S_eff = max(1, min(S_target, L_eff // 2))

    if n < L_target:
        if keep_short and n >= min_short:
            cand = paths_sorted
            if _seq_has_both_phases(cand):
                out.append(cand)
        return out

    i = 0
    while i + L_eff <= n:
        cand = paths_sorted[i:i + L_eff]
        if _seq_has_both_phases(cand):
            out.append(cand)
        i += S_eff
    return out


def onset_centered_sequence_adaptive(paths_sorted: List[str],
                                     onset_idx: Optional[int],
                                     pre_target: int = 10,
                                     post_target: int = 10,
                                     min_total: int = 8) -> List[str]:
    """
    TRAIN: UNE séquence centrée t0, puis filtrée pour exiger préictal & ictal.
    """
    if onset_idx is None:
        return []
    n = len(paths_sorted)
    if n == 0:
        return []

    pre = min(pre_target, onset_idx)
    post = min(post_target, n - 1 - onset_idx)

    lack_pre = pre_target - pre
    lack_post = post_target - post
    if lack_pre > 0:
        extra = min(lack_pre, (n - 1 - onset_idx) - post)
        post += max(0, extra)
    if lack_post > 0:
        extra = min(lack_post, onset_idx - pre)
        pre += max(0, extra)

    lo = max(0, onset_idx - pre)
    hi = min(n, onset_idx + post + 1)
    seq = paths_sorted[lo:hi]

    if len(seq) < min_total:
        return []
    return seq if _seq_has_both_phases(seq) else []

from typing import List, Optional, Tuple

def best_centered_onset_sequence_maxlen(
    paths_sorted: List[str],
    onset_idx: Optional[int],
    controlled: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """
    TEST (par défaut): retourne **une seule** séquence par seizure, aussi longue que possible,
    avec l'onset (premier ictal) **exactement au milieu** (index central).
    Longueur = 2*k + 1 avec k maximal tel que:
      - k <= nb d'epochs préictal dispos (à gauche de l'onset)
      - k <= nb d'epochs ictal dispos **après** l'onset (droite), en comptant l'onset lui-même
    => k_max = min(onset_idx, (n - onset_idx) - 1)
    Si pas de préictal OU pas d'ictal: renvoie [].

    Mode contrôlé:
      controlled = (n_pre, n_ictal)
        - n_pre   : nombre d'éléments avant l'onset (préictal)
        - n_ictal : nombre d'éléments à partir de l'onset (ictal), onset inclus
      La séquence renvoyée aura alors exactement n_pre + n_ictal éléments
      (l'onset se trouvera à l'index n_pre dans la séquence).
    """
    if onset_idx is None:
        return []
    n = len(paths_sorted)
    if n == 0:
        return []

    # ----- Mode contrôlé -----
    if controlled is not None:
        n_pre, n_ictal = controlled
        # Paramètres invalides
        if n_pre <= 0 or n_ictal <= 0:
            return []

        # Vérifie qu'on peut prélever n_pre à gauche et n_ictal à droite (onset inclus)
        lo = onset_idx - n_pre
        hi = onset_idx + (n_ictal - 1)
        if lo < 0 or hi >= n:
            return []

        seq = paths_sorted[lo:hi + 1]  # inclusif
        return seq if _seq_has_both_phases(seq) else []

    # ----- Mode original (non contrôlé) -----
    pre_available = onset_idx                  # éléments à gauche
    ictal_count = n - onset_idx                # éléments à droite en incluant l'onset
    if pre_available <= 0 or ictal_count <= 0:
        return []

    k_max = min(pre_available, ictal_count - 1)
    if k_max <= 0:
        return []

    lo = onset_idx - k_max
    hi = onset_idx + k_max
    seq = paths_sorted[lo:hi + 1]  # inclusif

    return seq if _seq_has_both_phases(seq) else []


def three_offset_test_sequences_same_length(
    paths_sorted: List[str],
    onset_idx: Optional[int],
    controlled: Optional[Tuple[int, int] | int] = None,
) -> List[List[str]]:
    """
    TEST: renvoie jusqu'à 3 séquences (25%, 50%, 75%) de même longueur L.
    Contrôle:
      - controlled = (n_pre_center, n_ictal_center)  # comme avant
          => L = n_pre_center + n_ictal_center
          => la séquence '50%' aura l'onset à l'index n_pre_center
      - controlled = L (int)                         # longueur imposée
      - controlled = None                            # cherche L max faisable

    Chaque séquence doit contenir préictal ET ictal.
    """
    if onset_idx is None:
        return []
    n = len(paths_sorted)
    if n == 0:
        return []

    pre_available = onset_idx         # nb éléments à gauche (préictal)
    ictal_count   = n - onset_idx     # nb éléments à droite en incluant l'onset

    # Besoin d'au moins 1 pré et >=1 ictal (onset inclus) au global
    if pre_available <= 0 or ictal_count <= 1:
        return []

    # ----- helpers -----
    def has_both(seq: List[str]) -> bool:
        return _seq_has_both_phases(seq)

    def slice_by_pos(L: int, pos_onset_in_seq: int) -> List[str]:
        lo = onset_idx - pos_onset_in_seq
        hi = lo + L
        if lo < 0 or hi > n:
            return []
        seq = paths_sorted[lo:hi]
        return seq if has_both(seq) else []

    def ok_length(L: int, p50_override: Optional[int] = None) -> bool:
        if L < 3:
            return False

        # positions théoriques de l'onset dans la séquence
        p50 = (L - 1) // 2 if p50_override is None else p50_override
        p25 = int(np.floor(0.25 * (L - 1)))
        p75 = int(np.floor(0.75 * (L - 1)))

        # chaque séquence doit contenir >=1 pré et >=1 ictal
        if not (1 <= p25 <= L - 2): return False
        if not (1 <= p50 <= L - 2): return False
        if not (1 <= p75 <= L - 2): return False

        # disponibilité globale suffisante pour extraire L autour de l'onset
        # (gauche = pos_onset_in_seq; droite = L - pos)
        if p25 > pre_available or (L - p25) > ictal_count: return False
        if p50 > pre_available or (L - p50) > ictal_count: return False
        if p75 > pre_available or (L - p75) > ictal_count: return False
        return True

    # ----- déterminer L cible -----
    L_target: Optional[int] = None
    p50_override: Optional[int] = None

    if isinstance(controlled, tuple):
        n_pre_c, n_ict_c = controlled
        if n_pre_c <= 0 or n_ict_c <= 0:
            return []
        L_try = n_pre_c + n_ict_c
        # on impose que la séquence "50%" place l'onset à l'index n_pre_c
        p50_override = n_pre_c
        if ok_length(L_try, p50_override=p50_override):
            L_target = L_try
        else:
            return []  # longueur demandée impossible proprement
    elif isinstance(controlled, int):
        L_try = controlled
        if ok_length(L_try):
            L_target = L_try
        else:
            return []
    else:
        # mode auto: on part du max "centre" possible puis on décrémente
        L_center_max = 2 * min(pre_available, ictal_count - 1) + 1
        for L_try in range(L_center_max, 2, -1):
            if ok_length(L_try):
                L_target = L_try
                break
        if L_target is None:
            return []

    # ----- construire les 3 séquences -----
    if p50_override is None:
        p50 = (L_target - 1) // 2
    else:
        p50 = p50_override
    p25 = int(np.floor(0.25 * (L_target - 1)))
    p75 = int(np.floor(0.75 * (L_target - 1)))

    seq25 = slice_by_pos(L_target, p25)
    seq50 = slice_by_pos(L_target, p50)
    seq75 = slice_by_pos(L_target, p75)

    out = []
    if seq25: out.append(seq25)
    if seq50: out.append(seq50)
    if seq75: out.append(seq75)
    return out


def main():
    # Lecture & checks
    df = pd.read_csv(INDEX_CSV)
    _require_columns(df, ["dataset", "phase", "patient", "seizure", "epoch", "path"])

    # tri + filtre éventuel epoch = -1 (comme l'original)
    df = df.sort_values(["dataset", "patient", "seizure", "epoch"]).reset_index(drop=True)
    if "epoch" in df.columns:
        df = df[df["epoch"].astype(int) != -1].reset_index(drop=True)

    splits: Dict[str, Dict[str, List[List[str]]]] = {}
    summary_rows = []

    by_pat = df.groupby(["dataset", "patient"], dropna=False, sort=False)
    n_pat_all = len(by_pat)
    n_pat_kept, n_pat_skipped, n_pat_dropped = 0, 0, 0

    for (dataset, patient), dfP in by_pat:
        crises = sorted(dfP["seizure"].unique().tolist())

        # --- Filtre patient: au moins MIN_CRISES crises disponibles ---
        if len(crises) < MIN_CRISES:
            n_pat_skipped += 1
            if PRINT_PER_PATIENT_STATS:
                print(f"[SKIP] {dataset}::{patient} (crises={len(crises)} < {MIN_CRISES})")
            continue

        bucket_train: List[List[str]] = []
        bucket_test:  List[List[str]] = []

        # Toutes les seizures -> TRAIN + TEST
        for sid in crises:
            dfS = dfP[dfP["seizure"] == sid].copy()
            paths, onset_idx, _ = _paths_ordered_with_trel(dfS)

            # TRAIN
            train_seqs = make_sliding_sequences(
                paths, L_target=L_TARGET, S_target=S_TARGET,
                keep_short=KEEP_SHORT, min_short=MIN_SEQ_EPOCHS
            )
            bucket_train.extend(train_seqs)

            if ADD_ONSET_SEQ:
                onset_seq = onset_centered_sequence_adaptive(
                    paths, onset_idx,
                    pre_target=ONSET_PRE_TARGET, post_target=ONSET_POST_TARGET,
                    min_total=ONSET_MIN_TOTAL
                )
                if onset_seq:
                    bucket_train.append(onset_seq)

            # TEST
            test_seq = best_centered_onset_sequence_maxlen(paths, onset_idx, controlled = CONTROLLED)
            if test_seq:
                bucket_test.append(test_seq)
            #test_seqs = three_offset_test_sequences_same_length(paths, onset_idx, controlled=CONTROLLED)
            #bucket_test.extend(test_seqs)

        # Dédoublonnage exact
        def _dedup(list_of_seqs: List[List[str]]) -> List[List[str]]:
            seen, clean = set(), []
            for seq in list_of_seqs:
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    clean.append(seq)
            return clean

        bucket_train = _dedup(bucket_train)
        bucket_test  = _dedup(bucket_test)

        # Garde-fou global par split
        keep_patient = True
        if MUST_HAVE_BOTH_PHASES_PER_SPLIT:
            if not bucket_train or not _split_has_both_phases(bucket_train):
                keep_patient = False
            if not bucket_test or not _split_has_both_phases(bucket_test):
                keep_patient = False

        if not keep_patient:
            n_pat_dropped += 1
            if PRINT_PER_PATIENT_STATS:
                print(f"[DROP] {dataset}::{patient} — split(s) sans les deux phases "
                      f"(train={len(bucket_train)} seqs, test={len(bucket_test)} seqs)")
            continue

        n_pat_kept += 1
        pat_key = f"{dataset}::{patient}"
        splits[pat_key] = {"train": bucket_train, "test": bucket_test}

        summary_rows.append({
            "patient_key": pat_key,
            "dataset": dataset,
            "patient": patient,
            "n_crises": len(crises),
            "n_train_seqs": len(bucket_train),
            "n_test_seqs": len(bucket_test),
            "L_target": L_TARGET, "S_target": S_TARGET,
            "keep_short": bool(KEEP_SHORT),
            "add_onset_seq": bool(ADD_ONSET_SEQ),
            "onset_pre_target": ONSET_PRE_TARGET,
            "onset_post_target": ONSET_POST_TARGET,
            "onset_min_total": ONSET_MIN_TOTAL,
            "both_phases_required": bool(MUST_HAVE_BOTH_PHASES_PER_SPLIT),
            "min_crises_required": MIN_CRISES,
        })

        if PRINT_PER_PATIENT_STATS:
            print(f"[OK] {pat_key}: crises={len(crises)} | train_seqs={len(bucket_train)} | test_seqs={len(bucket_test)}")

    # Écriture
    with open(OUT_SPLIT_JSON, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\n[OK] wrote train/test splits per patient: {OUT_SPLIT_JSON}")
    print(f"Patients kept: {n_pat_kept} / {n_pat_all} (skipped: {n_pat_skipped}, dropped: {n_pat_dropped})")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "patient"])
        summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
        print(f"[OK] summary: {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
