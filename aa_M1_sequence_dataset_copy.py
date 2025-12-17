# sequence_dataset.py
import os
import json
import torch
from typing import List, Dict, Tuple, Any
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class SequenceDataset(Dataset):
    """
    Lit les chemins de graphes (.pt) séquencés (par patient & split).
    Supporte 2 formats :
      A) par patient -> dict {"train": [...], "val": [...], "test": [...]}
      B) par patient -> liste de folds: [ {"train": [...], "val": [...], "test": [...]}, ... ]
         => on sélectionne le fold via fold_idx (par défaut 0).
    """
    def __init__(self, splits_json_path: str, patient_key: str, split_key: str, fold_idx: int = 0):
        with open(splits_json_path, "r") as f:
            all_splits: Dict[str, Any] = json.load(f)

        if patient_key not in all_splits:
            raise KeyError(f"Patient '{patient_key}' introuvable dans {splits_json_path}")

        entry = all_splits[patient_key]
        # B) cas CV: liste de folds
        if isinstance(entry, list):
            if not (0 <= fold_idx < len(entry)):
                raise IndexError(f"fold_idx={fold_idx} hors limites (0..{len(entry)-1}) pour {patient_key}")
            fold = entry[fold_idx]
            if split_key not in fold:
                raise KeyError(f"Split '{split_key}' introuvable au fold {fold_idx} pour {patient_key}. "
                               f"Clés dispo: {list(fold.keys())}")
            self.seq_paths: List[List[str]] = fold[split_key]
        # A) cas simple: dict direct
        elif isinstance(entry, dict):
            if split_key not in entry:
                raise KeyError(f"Split '{split_key}' introuvable pour {patient_key}. "
                               f"Clés dispo: {list(entry.keys())}")
            self.seq_paths: List[List[str]] = entry[split_key]
        else:
            raise TypeError(f"Format non supporté pour {patient_key}: {type(entry)}")
        
                
        
        # Détection automatique du nombre total d’électrodes pour ce patient
        self.num_electrodes_patient = None
        try:
            # On cherche le premier fichier existant
            for seq in self.seq_paths:
                for p in seq:
                    if os.path.exists(p):
                        g = torch.load(p, weights_only=False)
                        if hasattr(g, "x"):
                            self.num_electrodes_patient = int(g.x.size(0))
                        # Si on a des IDs d’électrode explicites (recommandé)
                        if hasattr(g, "elec_ids"):
                            self.num_electrodes_patient = int(g.elec_ids.max().item() + 1)
                        break
                if self.num_electrodes_patient is not None:
                    print(self.num_electrodes_patient)
                    break

            if self.num_electrodes_patient is None:
                print(f"[WARN] Impossible d'inférer le nombre d’électrodes pour {patient_key}.")
                self.num_electrodes_patient = 1
        except Exception as e:
            print(f"[WARN] Erreur pendant détection du nombre d’électrodes pour {patient_key}: {e}")
            self.num_electrodes_patient = 1
        self._cache = {}    # <--- ajoute cette ligne
        
    def __len__(self) -> int:
        return len(self.seq_paths)

    #def __getitem__(self, idx: int) -> List[Data]:
        #paths = self.seq_paths[idx]
        #return [torch.load(p, weights_only=False) for p in paths]


    def __getitem__(self, idx: int) -> List[Data]:
        paths = self.seq_paths[idx]
        graphs = []
        for p in paths:
            if p not in self._cache:
                self._cache[p] = torch.load(p, weights_only=False)
            graphs.append(self._cache[p])
        return graphs









def collate_sequences(list_of_seqs: List[List[Data]]):
    """
    Agrège un batch de séquences de longueurs potentiellement différentes.

    Retourne:
      - batches_time: List[Batch]        # un Batch PyG par pas de temps t (pour les graphes présents à t)
      - seq_ids_time: List[List[int]]    # pour chaque Batch_t, l’index de séquence (0..B-1) de chaque graphe
      - y_graph_time: List[Tensor]       # pour chaque Batch_t, labels (0=preictal,1=ictal) par graphe
      - B: int                           # taille du batch (nb de séquences)
    """
    if not list_of_seqs:
        return [], [], [], 0

    B = len(list_of_seqs)
    T = max(len(seq) for seq in list_of_seqs)

    batches_time, seq_ids_time, y_graph_time = [], [], []

    for t in range(T):
        graphs_t, seq_ids_t, y_t = [], [], []
        for b_idx, seq in enumerate(list_of_seqs):
            if t < len(seq):
                g = seq[t]
                graphs_t.append(g)
                seq_ids_t.append(b_idx)
                # Label par graphe depuis meta.phase
                phase = str(getattr(g, "meta", {}).get("phase", "preictal")).lower()
                y_t.append(1.0 if phase == "ictal" else 0.0)
        if graphs_t:
            batches_time.append(Batch.from_data_list(graphs_t))
            seq_ids_time.append(seq_ids_t)
            y_graph_time.append(torch.tensor(y_t, dtype=torch.float32))

    return batches_time, seq_ids_time, y_graph_time, B
