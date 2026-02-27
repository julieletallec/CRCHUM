# -------------------------------------------------------------------------
# model_backbone.py — GNN + GRU + k-WTA persistant + attention + noisy-OR
# -------------------------------------------------------------------------
from typing import List, Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
def _simple_pool(
    p_node: torch.Tensor,
    att: Optional[torch.Tensor] = None,
    mode: str = "active_mean",
    k_ratio: float = 0.2
) -> torch.Tensor:
    """
    Poolings simples pour p_node in [0,1].

    mode ∈ {"mean", "max", "att_mean", "active_mean", "topk_mean"}

    - "mean":        moyenne non pondérée sur tous les nœuds
    - "max":         max des p
    - "att_mean":    moyenne pondérée par att (softmax spatial)
    - "active_mean": moyenne sur les nœuds actifs (p>0)
    - "topk_mean":   moyenne des k% plus grandes valeurs de p
    """
    p = torch.clamp(p_node.view(-1), 0.0, 1.0)
    if p.numel() == 0:
        return p.new_tensor(0.0)

    if mode == "max":
        return p.max()

    if mode == "att_mean":
        if att is None:
            return p.mean()
        w = torch.clamp(att.view(-1), min=0.0)
        s = w.sum()
        if s <= 0:
            return p.mean()
        w = w / s
        return (w * p).sum()

    if mode == "active_mean":
        mask = (p > 0.0)
        if mask.any():
            return p[mask].mean()
        else:
            return p.new_tensor(0.0)

    if mode == "topk_mean":
        k = max(1, int(math.ceil(k_ratio * p.numel())))
        vals, _ = torch.topk(p, k)
        return vals.mean()

    # défaut: moyenne simple
    return p.mean()


def _noisy_or_pool(p_node: torch.Tensor, att: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    p_node: (N,) in [0,1]
    att   : (N,) >=0, somme=1 (optionnel)
    p_graph = 1 - Π_i (1 - att_i * p_i)
    """
    p_node = torch.clamp(p_node.view(-1), 0.0, 1.0)
    if p_node.numel() == 0:
        return p_node.new_tensor(0.0)
    if att is None:
        att = torch.full_like(p_node, 1.0 / float(p_node.numel()))
    else:
        att = torch.clamp(att.view(-1), min=0.0)
        att = att / (att.sum() + 1e-12)
    term = torch.clamp(1.0 - att * p_node, 1e-6, 1.0)
    return 1.0 - torch.prod(term)


def _kwta_ste(p: torch.Tensor, k: int, keep_idx=None) -> torch.Tensor:
    """
    k-WTA "dur" (top-k) avec STE: forward: masque binaire; backward: identité.
    p: (N,)
    keep_idx: indices à forcer "on" (optionnel)
    """
    N = p.numel()
    if N == 0:
        return torch.zeros_like(p)
    k = int(min(max(1, k), N))
    # top-k
    _, topk_idx = torch.topk(p, k, largest=True, sorted=False)
    mask = torch.zeros_like(p)
    mask[topk_idx] = 1.0
    if keep_idx is not None and len(keep_idx) > 0:
        mask[keep_idx] = 1.0
    p_hard = p * mask
    return p + (p_hard - p).detach()  # STE


def _kwta_persistent_ste(
    P_bt: torch.Tensor,       # (N, T) probas après sigmoid
    K_max: int,               # capacité max par pas
    K_new: int,               # nb max de nouveaux entrants par pas
    alpha_stay: float = 0.8,  # critère relatif : rester si p_t >= alpha * p_{t-1}
    abs_min: float = 0.0      # seuil absolu minimal pour avoir le droit de rester/entrer
) -> List[torch.Tensor]:
    """
    Implémente un k-WTA PERSISTANT par pas de temps, avec STE.

    Stratégie:
      - t=0: k-WTA simple avec K_max.
      - t>0: on garde d'abord les "anciens gagnants" i ∈ A_{t-1} qui vérifient p_t[i] >= max(abs_min, alpha_stay*p_{t-1}[i]),
              puis on autorise au plus K_new nouveaux i parmi le reste (en top score), sans dépasser K_max au total.
      - sortie: liste [p_t_masked] de longueur T, chacun (N,)
    """
    N, T = P_bt.shape
    if N == 0 or T == 0:
        return [torch.zeros((N,), device=P_bt.device, dtype=P_bt.dtype) for _ in range(max(T,1))]

    P_bt = torch.clamp(P_bt, 0.0, 1.0)
    out_list: List[torch.Tensor] = []

    # t=0: simple k-WTA (K_max)
    p0 = P_bt[:, 0]
    p0_kwta = _kwta_ste(p0, K_max)
    out_list.append(p0_kwta)
    prev_active = (p0_kwta > 0).nonzero(as_tuple=True)[0]  # indices actifs à t=0

    for t in range(1, T):
        p_prev = P_bt[:, t-1]
        p_curr = P_bt[:, t]

        # candidats "stay" = anciens actifs qui restent assez hauts
        stay_mask = torch.zeros(N, device=P_bt.device, dtype=torch.bool)
        if prev_active.numel() > 0:
            stay_thresh = torch.maximum(alpha_stay * p_prev[prev_active], torch.full_like(prev_active.float(), abs_min)).to(P_bt.dtype)
            ok = p_curr[prev_active] >= stay_thresh
            stay_idx = prev_active[ok]
            stay_mask[stay_idx] = True
        else:
            stay_idx = torch.tensor([], device=P_bt.device, dtype=torch.long)

        # places restantes
        nb_stay = int(stay_mask.sum().item())
        capacity_left = max(0, K_max - nb_stay)

        # nb de nouveaux autorisés
        nb_new_allowed = min(K_new, capacity_left)

        # sélection de nouveaux entrants
        if nb_new_allowed > 0:
            # scores des non-stay
            not_stay = (~stay_mask)
            # filtre par seuil absolu
            cand_idx = torch.nonzero(not_stay & (p_curr >= abs_min), as_tuple=True)[0]
            if cand_idx.numel() > 0:
                cand_scores = p_curr[cand_idx]
                # top nb_new_allowed parmi ces candidats
                _, ord_idx = torch.topk(cand_scores, k=min(nb_new_allowed, cand_idx.numel()), largest=True, sorted=False)
                new_idx = cand_idx[ord_idx]
            else:
                new_idx = torch.tensor([], device=P_bt.device, dtype=torch.long)
        else:
            new_idx = torch.tensor([], device=P_bt.device, dtype=torch.long)

        keep_idx = torch.unique(torch.cat([stay_idx, new_idx], dim=0))
        # si, pour une raison quelconque, on est en dessous de K_max, on peut compléter avec les meilleurs restants
        remaining_slots = max(0, K_max - keep_idx.numel())
        if remaining_slots > 0:
            mask_keep = torch.zeros(N, device=P_bt.device, dtype=torch.bool)
            mask_keep[keep_idx] = True
            rest_idx = torch.nonzero(~mask_keep, as_tuple=True)[0]
            if rest_idx.numel() > 0:
                _, compl_top = torch.topk(p_curr[rest_idx], k=min(remaining_slots, rest_idx.numel()), largest=True, sorted=False)
                keep_idx = torch.unique(torch.cat([keep_idx, rest_idx[compl_top]], dim=0))

        p_t_kwta = _kwta_ste(p_curr, k=K_max, keep_idx=keep_idx)
        out_list.append(p_t_kwta)
        prev_active = (p_t_kwta > 0).nonzero(as_tuple=True)[0]

    return out_list


class GraphBackboneNodeGRU(nn.Module):
    """
    Spatial: 2x GCNConv -> h_v(t)
    Temporel: GRU par nœud -> logits_v(t) -> sigmoid -> k-WTA persistant
    Pooling graphe: attention softmax + noisy-OR sur p_node_kWTA(t)
    """
    def __init__(self,
                 in_dim: int,
                 hid: int = 128,
                 # --- contrôle parcimonie persistante ---
                 #k_max_ratio: float = 0.15,     # fraction d'électrodes actives max par pas
                 k_max_ratio: float = 1.0,
                 k_max_fixed: Optional[int] = None,  # sinon K_max = ceil(k_max_ratio*N)
                 k_new_ratio: float = 0.05,     # nb max de NOUVEAUX entrants par pas (en fraction)
                 k_new_fixed: Optional[int] = None,
                 alpha_stay: float = 0.8,       # rester si p_t >= max(abs_min, alpha_stay*p_{t-1})
                 abs_min: float = 0.0,
                 pooling_mode: str = "topk_mean",
                 #pooling_mode: str = "mean",
                 k_ratio: float = 0.2):
        super().__init__()

        self.pooling_mode = pooling_mode
        self.k_ratio = float(k_ratio)
        # ----- GNN spatial -----
        self.conv1 = GCNConv(in_dim, hid)
        self.bn1   = nn.BatchNorm1d(hid)
        self.conv2 = GCNConv(hid, hid)
        self.bn2   = nn.BatchNorm1d(hid)

        # ----- GRU temporel (par nœud) -----
        self.gru = nn.GRU(input_size=hid, hidden_size=hid, batch_first=True)

        # ----- têtes -----
        self.node_head = nn.Linear(hid, 1)      # logits -> sigmoid plus tard
        # Créer att_head seulement si le mode de pooling en a besoin
        if self.pooling_mode in {"att_mean", "noisy_or"}:
            self.att_head = nn.Linear(hid, 1)   # logits attention (softmax spatial)
        else:
            self.att_head = None

        # hyperparams k-WTA persistant
        self.k_max_ratio = float(k_max_ratio)
        self.k_max_fixed = int(k_max_fixed) if (k_max_fixed is not None) else None
        self.k_new_ratio = float(k_new_ratio)
        self.k_new_fixed = int(k_new_fixed) if (k_new_fixed is not None) else None
        self.alpha_stay  = float(alpha_stay)
        self.abs_min     = float(abs_min)

        

    # encode un Batch_t en h_v(t)
    def _encode_nodes(self, bt: Batch) -> torch.Tensor:
        h = self.conv1(bt.x, bt.edge_index); h = self.bn1(h); h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, bt.edge_index); h = self.bn2(h); h = F.relu(h)
        return h
    
    @classmethod
    def from_config(cls, cfg: dict, inferred_in_dim: int | None = None):
        """
        Instancie le modèle à partir d'un dictionnaire de configuration.
        Si 'in_dim' vaut "auto", il faut fournir 'inferred_in_dim'.
        """
        m = cfg.get("model", {})

        in_dim = m.get("in_dim", "auto")
        if in_dim == "auto":
            if inferred_in_dim is None:
                raise ValueError("in_dim is 'auto' but inferred_in_dim is None")
            in_dim = inferred_in_dim

        return cls(
            in_dim=in_dim,
            hid=int(m.get("hid", 128)),
            k_max_ratio=float(m.get("k_max_ratio", 1.0)),
            k_max_fixed=(
                None if m.get("k_max_fixed") in (None, "null") else int(m["k_max_fixed"])
            ),
            k_new_ratio=float(m.get("k_new_ratio", 0.05)),
            k_new_fixed=(
                None if m.get("k_new_fixed") in (None, "null") else int(m["k_new_fixed"])
            ),
            alpha_stay=float(m.get("alpha_stay", 0.8)),
            abs_min=float(m.get("abs_min", 0.0)),
            pooling_mode=str(m.get("pooling_mode", "topk_mean")),
            k_ratio=float(m.get("k_ratio", 0.2)),
        )

    @torch.no_grad()
    def _split_by_graph(self, h_nodes: torch.Tensor, bt: Batch) -> List[torch.Tensor]:
        out = []
        if bt.batch.numel() == 0:
            return out
        num_graphs = int(bt.batch.max().item()) + 1
        for g in range(num_graphs):
            out.append(h_nodes[bt.batch == g])
        return out

    def forward_sequence(
        self,
        batches_time: List[Batch],
        seq_ids_time: Optional[List[List[int]]] = None,
        return_node_probs: bool = False
    ):
        device = batches_time[0].x.device

        # 1) encode GNN par temps
        per_seq_nodes: Dict[int, List[torch.Tensor]] = {}
        time_index_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for t, bt in enumerate(batches_time):
            h_nodes_t = self._encode_nodes(bt)              # (sum_N_t, hid)
            h_by_graph = self._split_by_graph(h_nodes_t, bt)  # [ (N_g, hid) ]
            num_graphs_t = len(h_by_graph)
            b_ids = list(range(num_graphs_t)) if (seq_ids_time is None) else seq_ids_time[t]
            for g_idx in range(num_graphs_t):
                b = int(b_ids[g_idx])
                per_seq_nodes.setdefault(b, [])
                tau = len(per_seq_nodes[b])
                per_seq_nodes[b].append(h_by_graph[g_idx])
                time_index_map[(t, g_idx)] = (b, tau)

        # 2) GRU -> logits -> sigmoid -> k-WTA persistant ; attention softmax
        per_seq_node_probs: Dict[int, List[torch.Tensor]] = {}
        per_seq_att_w:     Dict[int, List[torch.Tensor]] = {}

        for b, list_h in per_seq_nodes.items():
            N_b = list_h[0].size(0)
            for hstep in list_h:
                if hstep.size(0) != N_b:
                    raise ValueError(f"Incohérence N dans séquence {b}: {N_b} vs {hstep.size(0)}.")

            H_bt = torch.stack(list_h, dim=0).permute(1,0,2).contiguous()  # (N_b, T_b, hid)
            H_out, _ = self.gru(H_bt)                                      # (N_b, T_b, hid)

            logits = self.node_head(H_out).squeeze(-1)  # (N_b, T_b)
            P_bt   = torch.sigmoid(logits)                  # (N_b, T_b) in [0,1]

            if self.att_head is not None:
                attlog = self.att_head(H_out).squeeze(-1)   # (N_b, T_b)
                A_bt   = torch.softmax(attlog, dim=0)       # (N_b, T_b), somme=1 sur N_b
            else:
                A_bt = None   

            # paramètres K_max / K_new
            if self.k_max_fixed is not None:
                K_max = int(min(max(1, self.k_max_fixed), N_b))
            else:
                K_max = int(min(max(1, math.ceil(self.k_max_ratio * N_b)), N_b))

            if self.k_new_fixed is not None:
                K_new = int(min(max(0, self.k_new_fixed), K_max))
            else:
                K_new = int(min(max(0, math.ceil(self.k_new_ratio * N_b)), K_max))

            # applique k-WTA persistant (STE) sur la séquence (N_b, T_b)
            p_list_kwta = _kwta_persistent_ste(
                P_bt, K_max=K_max, K_new=K_new,
                alpha_stay=self.alpha_stay, abs_min=self.abs_min
            )  # liste de T_b tenseurs (N_b,)

            per_seq_node_probs[b] = p_list_kwta
            if A_bt is not None:
                per_seq_att_w[b] = [A_bt[:, t] for t in range(A_bt.size(1))]
            else:
                per_seq_att_w[b] = [None for _ in range(P_bt.size(1))]
        # 3) reconstruction par temps global -> p_g(t) + p_nodes(t)
        p_graph_time: List[torch.Tensor] = []
        p_node_time:  List[List[torch.Tensor]] = []

        for t, bt in enumerate(batches_time):
            num_graphs_t = int(bt.batch.max().item()) + 1 if bt.batch.numel() > 0 else 0
            p_g_list, p_nodes_list = [], []
            for g_idx in range(num_graphs_t):
                b, tau = time_index_map[(t, g_idx)]
                p_nodes = per_seq_node_probs[b][tau]  # (N_b,)
                att_w   = per_seq_att_w[b][tau] if A_bt is not None else None  # (N_b,) ou None

                p_nodes_list.append(p_nodes)
                #p_g_list.append(_noisy_or_pool(p_nodes, att=att_w))
                #p_g_list.append(_simple_pool(p_nodes, att=att_w, mode="topk_mean"))
                p_g_list.append(
                    _simple_pool(
                        p_nodes, 
                        att=att_w,
                        mode=self.pooling_mode,
                        k_ratio=self.k_ratio
                    )
                )
            if len(p_g_list) == 0:
                p_graph_time.append(torch.empty(0, device=device))
                p_node_time.append([])
            else:
                p_graph_time.append(torch.stack(p_g_list))
                p_node_time.append(p_nodes_list)


        if return_node_probs:
            return p_graph_time, p_node_time, per_seq_node_probs
        return p_graph_time
