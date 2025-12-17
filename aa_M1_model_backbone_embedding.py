import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch

def _topk_mean(x: torch.Tensor, k_ratio: float) -> torch.Tensor:
    x = x.view(-1)
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    k = max(1, math.ceil(k_ratio * x.numel()))
    topk_vals, _ = torch.topk(x, k, largest=True, sorted=False)
    return topk_vals.mean()


class GraphBackboneNodeGRU(nn.Module):
    """
    Ajouts majeurs vs. version initiale :
      - Embedding d'identité d'électrode par patient (self.elec_emb), ajouté après conv1
      - Bloc d'attention inter-électrodes après GRU (self.node_mha) pour communication spatiale
      - Perte de diversité spatiale (variance des proba nodales intra-graphe) renvoyée dans aux_losses

    Exigences de données :
      - Chaque Batch_t doit contenir un tenseur bt.elec_ids (long) de taille (sum_N_t,)
        qui indexe les électrodes du PATIENT dans [0, num_electrodes_patient-1].
        Si manquant, on considère un vecteur d'ids à 0 (déconseillé mais fonctionnera).
      - Au sein d'une même séquence (crise) b, le nombre d'électrodes doit rester constant
        sur les pas de temps (comme avant). D'une séquence à l'autre, il peut varier.

    Sorties :
      - p_graph_time : List[Tensor(num_graphs_t,)]
      - p_node_time  : List[List[Tensor(N_g,)]] par temps puis par graphe
      - per_seq_node_probs : Dict[b] -> [Tensor(N_b,), ...] (si return_node_probs=True)
      - aux_losses (optionnel si return_aux_losses=True) : dict avec "loss_diversity"
    """
    def __init__(
        self,
        in_dim: int,
        hid: int = 128,
        topk_ratio: float = 0.2,
        num_electrodes_patient: Optional[int] = None,
        mha_heads: int = 4,
        node_dropout_p: float = 0.0,   # dropout de nœuds (facultatif)
    ):
        super().__init__()
        # ----- GNN spatial (par fenêtre) -----
        self.conv1 = GCNConv(in_dim, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        self.conv2 = GCNConv(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid)

        # ----- Embeddings d'identité d'électrodes (par patient) -----
        # Note: si num_electrodes_patient est None, on créera un embedding de taille 1.
        self.num_elec = int(num_electrodes_patient) if num_electrodes_patient is not None else 1
        self.elec_emb = nn.Embedding(self.num_elec, hid)

        # ----- GRU temporel (par noeud) -----
        self.gru = nn.GRU(input_size=hid, hidden_size=hid, batch_first=True)  # (N_nodes, T, hid)

        # ----- Communication spatiale post-GRU (attention entre électrodes au même t) -----
        # On traite chaque temps t comme un "batch" et la dimension "séquence" = N_b (électrodes)
        self.node_mha = nn.MultiheadAttention(embed_dim=hid, num_heads=mha_heads, batch_first=True)

        # ----- tête nodale (après GRU + MHA) -----
        self.node_head = nn.Sequential(nn.Linear(hid, 1), nn.Sigmoid())

        self.topk_ratio = topk_ratio
        self.node_dropout_p = float(node_dropout_p)

    # -- encode un Batch_t en embeddings nodaux h_v(t) --
    def _encode_nodes(self, bt: Batch) -> torch.Tensor:
        # bt.x: (sum_N, in_dim), bt.edge_index
        h = self.conv1(bt.x, bt.edge_index)          # (sum_N, hid)
        h = self.bn1(h)
        h = F.relu(h)

        # Embedding d'identité d'électrode ajouté après la première projection
        if hasattr(bt, "elec_ids") and bt.elec_ids is not None:
            elec_ids = bt.elec_ids
            # sécurité de type
            if not torch.is_floating_point(elec_ids):
                elec_ids = elec_ids.long()
        else:
            # fallback : un seul id 0 (toutes les électrodes partagent le même embedding)
            elec_ids = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # clamp si jamais certaines ids dépassent num_elec-1
        if self.num_elec > 0:
            elec_ids = elec_ids.clamp(min=0, max=self.num_elec - 1)
        elec_e = self.elec_emb(elec_ids)             # (sum_N, hid)

        h = h + elec_e                                # injection d'identité
        h = F.dropout(h, p=0.3, training=self.training)

        h = self.conv2(h, bt.edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h  # (sum_N, hid)

    @torch.no_grad()
    def _split_by_graph(self, h_nodes: torch.Tensor, bt: Batch) -> List[torch.Tensor]:
        """Découpe (sum_N, hid) en liste [ (N_g, hid) ] selon bt.batch."""
        out = []
        num_graphs = int(bt.batch.max().item()) + 1 if bt.batch.numel() > 0 else 0
        for g in range(num_graphs):
            mask = (bt.batch == g)
            out.append(h_nodes[mask])
        return out

    def _apply_node_dropout(self, H_bt: torch.Tensor) -> torch.Tensor:
        """
        Dropout 'par nœud' optionnel sur H_bt de forme (N_b, T_b, hid).
        On supprime (met à 0) aléatoirement des électrodes, identiquement sur tous les temps d'une crise.
        """
        if not self.training or self.node_dropout_p <= 0.0:
            return H_bt
        N_b = H_bt.size(0)
        keep_mask = (torch.rand(N_b, device=H_bt.device) > self.node_dropout_p).float()  # (N_b,)
        keep_mask = keep_mask.view(N_b, 1, 1)  # broadcast sur T_b et hid
        return H_bt * keep_mask

    def forward_sequence(
        self,
        batches_time: List[Batch],
        seq_ids_time: Optional[List[List[int]]] = None,
        return_node_probs: bool = False,
        return_aux_losses: bool = False,
    ):
        device = batches_time[0].x.device
        T = len(batches_time)

        # 1) Passe GNN par temps -> récupère embeddings par graphe et par séquence
        per_seq_nodes: Dict[int, List[torch.Tensor]] = {}
        time_index_map: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (t, g) -> (b, tau)

        for t, bt in enumerate(batches_time):
            h_nodes_t = self._encode_nodes(bt)            # (sum_N_t, hid)
            h_by_graph = self._split_by_graph(h_nodes_t, bt)  # list len=num_graphs_t
            num_graphs_t = len(h_by_graph)

            if seq_ids_time is None:
                b_ids = list(range(num_graphs_t))
            else:
                b_ids = seq_ids_time[t]  # longueur == num_graphs_t

            for g_idx in range(num_graphs_t):
                b = int(b_ids[g_idx])
                per_seq_nodes.setdefault(b, [])
                tau = len(per_seq_nodes[b])   # position temporelle locale dans la séquence b
                per_seq_nodes[b].append(h_by_graph[g_idx])  # (N_b, hid)
                time_index_map[(t, g_idx)] = (b, tau)

        # 2) GRU par noeud + Attention inter-électrodes -> probas nodales par séquence
        per_seq_node_probs: Dict[int, List[torch.Tensor]] = {}  # b -> [ (N_b,) @ t ]
        total_diversity_loss = 0.0
        num_div_terms = 0

        for b, list_h in per_seq_nodes.items():
            # cohérence du nombre de noeuds (N_b) au fil du temps de cette séquence
            N_b = list_h[0].size(0)
            for hstep in list_h:
                if hstep.size(0) != N_b:
                    raise ValueError(
                        f"Incohérence du nombre de noeuds dans la séquence {b}: "
                        f"{N_b} vs {hstep.size(0)}. Aligne tes graphes (même ordre d'électrodes) avant le training."
                    )

            # Empilement temporel
            H_bt = torch.stack(list_h, dim=0)          # (T_b, N_b, hid)
            H_bt = H_bt.permute(1, 0, 2).contiguous()  # (N_b, T_b, hid)

            # Dropout par nœud (optionnel)
            H_bt = self._apply_node_dropout(H_bt)      # (N_b, T_b, hid)

            # GRU par nœud
            H_bt_out, _ = self.gru(H_bt)               # (N_b, T_b, hid)

            # Attention inter-électrodes à chaque temps t
            # On revoit en (T_b, N_b, hid) pour traiter chaque temps comme un 'batch'
            H_time_first = H_bt_out.permute(1, 0, 2).contiguous()  # (T_b, N_b, hid)
            # MHA batch_first=True => (batch, seq, embed) = (T_b, N_b, hid)
            H_time_att, _ = self.node_mha(H_time_first, H_time_first, H_time_first)  # (T_b, N_b, hid)
            H_att = H_time_att.permute(1, 0, 2).contiguous()       # (N_b, T_b, hid)

            # Tête nodale -> probas
            P_bt = self.node_head(H_att)               # (N_b, T_b, 1)
            p_list = [P_bt[:, t, :].squeeze(-1) for t in range(P_bt.size(1))]  # [(N_b,), ...]
            per_seq_node_probs[b] = p_list

            # Perte de diversité spatiale : variance des proba nodales par temps
            # (on veut de la variance -> on maximise var -> on MINIMISE -var)
            for p_nodes_t in p_list:
                if p_nodes_t.numel() > 1:
                    var_t = torch.var(p_nodes_t, unbiased=False)
                    total_diversity_loss = total_diversity_loss + (-var_t)  # négatif pour l'ajouter à la loss
                    num_div_terms += 1

        if num_div_terms > 0:
            loss_diversity = total_diversity_loss / num_div_terms
        else:
            loss_diversity = torch.tensor(0.0, device=device)

        # 3) Reconstitution par temps global t -> p_g(t) et p_nodes(t) par graphe
        p_graph_time: List[torch.Tensor] = []
        p_node_time: List[List[torch.Tensor]] = []

        for t, bt in enumerate(batches_time):
            num_graphs_t = int(bt.batch.max().item()) + 1 if bt.batch.numel() > 0 else 0
            p_g_list, p_nodes_list = [], []
            for g_idx in range(num_graphs_t):
                b, tau = time_index_map[(t, g_idx)]
                p_nodes = per_seq_node_probs[b][tau]  # (N_b,)
                p_nodes_list.append(p_nodes)
                p_g = _topk_mean(p_nodes, self.topk_ratio)
                p_g_list.append(p_g)

            if len(p_g_list) == 0:
                p_graph_time.append(torch.empty(0, device=device))
                p_node_time.append([])
            else:
                p_graph_time.append(torch.stack(p_g_list))  # (num_graphs_t,)
                p_node_time.append(p_nodes_list)

            del bt  # libère un peu

        if return_node_probs and return_aux_losses:
            return p_graph_time, p_node_time, per_seq_node_probs, {"loss_diversity": loss_diversity}
        if return_node_probs:
            return p_graph_time, p_node_time, per_seq_node_probs
        if return_aux_losses:
            return p_graph_time, {"loss_diversity": loss_diversity}
        return p_graph_time
