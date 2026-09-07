import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy
import os
import hashlib
import dgl
from dgllife.model.gnn import GCN
from attention import MultiHeadAttentionLayer
from torch.nn.utils.weight_norm import weight_norm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

def masked_mean_pooling(x, mask):
    """Applies mask and then computes mean."""
    if mask.dtype == torch.bool:
        mask = mask.float()

    masked_x = x * mask
    sum_x = masked_x.sum(dim=1)
    sum_mask = mask.sum(dim=1)

    sum_mask = torch.max(sum_mask, torch.tensor(1e-6, device=sum_mask.device))

    mean_x = sum_x / sum_mask
    return mean_x

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None): #loss function BCE in two steps of cdan, this part is broken
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    y_hat = class_output.max(1)[1]

    if label.dtype != y_hat.dtype:
         label = label.type_as(y_hat)

    if weights is None:
        loss = nn.NLLLoss()(class_output, label.view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.view(label.size(0)))
        weights = weights.float().to(losses.device)
        loss = torch.sum(weights * losses) / (torch.sum(weights) + 1e-6)

    return n, loss


def entropy_logits(linear_output):   #calculates entropy ,so less confidence get less eightage for clauclating loss
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-10)), dim=1)
    return loss_ent


# feature-cache: on-disk memoisation for the two frozen encoders.
#
# ESM-2 and ChemBERTa are both frozen - eval() is pinned by the train() overrides below (the
# SW-04 fix) and every forward runs under no_grad - so the embedding of a given sequence string
# is deterministic and identical on every epoch. A DTI dataset repeats each protein and each
# SMILES across many pairs, so recomputing them per batch is pure waste. These helpers key each
# sequence by MD5 of its raw string and reuse the saved embedding instead.
#
# Caching is per-sequence, but the encoders return a *batched, padded* tensor. So the cached
# record stores the UNPADDED [L, D] embedding plus its real length L, and _encode_with_cache
# re-pads the batch here to reproduce exactly the shape, mask and valid-position values that
# the uncached batch-tokenizer path produces.
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _sequence_cache_key(sequence):                                        # feature-cache
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def _atomic_torch_save(record, cache_path):                               # feature-cache
    """Write via a temp file + rename so a crash mid-write cannot leave a half-written .pt
    that a later run would happily load as a valid embedding."""
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(record, tmp_path)
    os.replace(tmp_path, cache_path)


def _encode_with_cache(sequences, cache_dir, encode_one, device):         # feature-cache
    """Build (padded_features, attention_mask) from per-sequence cached embeddings.

    encode_one(sequence) -> [L, D] tensor for a single unpadded sequence. Misses are computed
    with it and written to cache_dir/{md5}.pt as {"features": cpu [L, D], "length": L}.

    Padded positions are filled with zeros. In the uncached path those positions instead hold
    the encoder's output for the pad token, but they are masked out of every downstream
    consumer (both MultiHeadAttentionLayer calls masked_fill them to -1e10 before the softmax,
    which underflows to exactly 0 weight), so the two paths agree everywhere it is read.
    """
    os.makedirs(cache_dir, exist_ok=True)

    records = []
    for sequence in sequences:
        cache_path = os.path.join(cache_dir, f"{_sequence_cache_key(sequence)}.pt")
        record = None
        if os.path.exists(cache_path):
            try:
                record = torch.load(cache_path, map_location="cpu", weights_only=True)
            except Exception:
                record = None       # unreadable/truncated entry: fall through and recompute

        if record is None:
            features = encode_one(sequence)
            record = {"features": features.detach().to("cpu"), "length": int(features.shape[0])}
            _atomic_torch_save(record, cache_path)

        records.append(record)

    max_len = max(record["length"] for record in records)
    feature_dim = records[0]["features"].shape[-1]
    feature_dtype = records[0]["features"].dtype

    padded_features = torch.zeros(len(records), max_len, feature_dim, dtype=feature_dtype,
                                 device=device)
    attention_mask = torch.zeros(len(records), max_len, dtype=torch.long, device=device)
    for i, record in enumerate(records):
        length = record["length"]
        padded_features[i, :length] = record["features"].to(device=device, dtype=feature_dtype)
        attention_mask[i, :length] = 1

    return padded_features, attention_mask


class ProtBertProteinEncoder(nn.Module):
    def __init__(self, esm_model_path, device, use_cache=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(esm_model_path).to(device).eval()
        self.output_dim = 1280
        # feature-cache: off by default - use_cache False reproduces the original path exactly.
        self.use_cache = use_cache
        self.cache_dir = os.path.join(_CACHE_ROOT, "esm")

    # iter1 - FIXED (SW-04): this encoder is frozen, but it is a submodule of CMA, so the
    # self.model.train() at the top of every training epoch used to recursively flip it back
    # into train mode and silently switch ESM-2 dropout on during training forward passes -
    # making the "frozen" features stochastic and different from what eval() later saw.
    # Pinning train() to eval keeps the encoder deterministic in both phases.
    def train(self, mode=True):
        return super().train(False)

    def _encode_one(self, protein_sequence):                              # feature-cache
        """Run ESM-2 on a single unpadded sequence and return its [L, 1280] embedding."""
        encoded_inputs = self.tokenizer(protein_sequence, padding=False, truncation=True,
                                        return_tensors='pt', max_length=512)
        encoded_inputs = {key: value.to(self.model.device) for key, value in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_inputs, output_hidden_states=True)

        return outputs.hidden_states[-1][0]

    def forward(self, protein_sequences):
        # feature-cache: cached branch re-pads per-sequence embeddings into the same batched
        # (features, attention_mask) pair the uncached branch below returns.
        if self.use_cache:
            return _encode_with_cache(protein_sequences, self.cache_dir, self._encode_one,
                                      self.model.device)

        encoded_inputs = self.tokenizer(protein_sequences, padding=True, truncation=True, return_tensors='pt',
                                        max_length=512)
        encoded_inputs = {key: value.to(self.model.device) for key, value in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_inputs, output_hidden_states=True)

        features = outputs.hidden_states[-1]
        attention_mask = encoded_inputs['attention_mask']

        return features, attention_mask


class ChemBERTaEncoder(nn.Module):
    def __init__(self, chemberta_model_path, device, use_cache=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(chemberta_model_path)
        self.model = AutoModel.from_pretrained(chemberta_model_path).to(device).eval()
        self.output_dim = self.model.config.hidden_size
        # feature-cache: off by default - use_cache False reproduces the original path exactly.
        self.use_cache = use_cache
        self.cache_dir = os.path.join(_CACHE_ROOT, "chemberta")

    # iter1 - FIXED (SW-04): same as ProtBertProteinEncoder - pinned to eval so the parent
    # CMA.train() cannot re-enable ChemBERTa dropout on the frozen encoder.
    def train(self, mode=True):
        return super().train(False)

    def _encode_one(self, smiles_sequence):                               # feature-cache
        """Run ChemBERTa on a single unpadded SMILES and return its [L, D] embedding."""
        encoded_inputs = self.tokenizer(smiles_sequence, padding=False, truncation=True,
                                        return_tensors='pt', max_length=512)
        encoded_inputs = {key: value.to(self.model.device) for key, value in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_inputs, output_hidden_states=True)

        return outputs.last_hidden_state[0]

    def forward(self, smiles_sequences):
        # feature-cache: cached branch re-pads per-sequence embeddings into the same batched
        # (features, attention_mask) pair the uncached branch below returns.
        if self.use_cache:
            return _encode_with_cache(smiles_sequences, self.cache_dir, self._encode_one,
                                      self.model.device)

        encoded_inputs = self.tokenizer(smiles_sequences, padding=True, truncation=True, return_tensors='pt',
                                        max_length=512)
        encoded_inputs = {key: value.to(self.model.device) for key, value in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_inputs, output_hidden_states=True)

        token_features = outputs.last_hidden_state
        attention_mask = encoded_inputs['attention_mask']

        return token_features, attention_mask

class CMA(nn.Module):
    def __init__(self, protbert_model_path, chemberta_model_path, device, **config):
        super().__init__()
        self.device = device
        self.init_params = {
            "protbert_model_path": protbert_model_path,
            "chemberta_model_path": chemberta_model_path,
            "device": device,
            **copy.deepcopy(config)
        }
        self.config = config

        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        max_drug_nodes = config["DRUG"]["MAX_NODES"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=config["DRUG"]["PADDING"],
                                           hidden_feats=drug_hidden_feats,
                                           max_nodes=max_drug_nodes)
        self.gcn_feature_dim = drug_hidden_feats[-1]

        # feature-cache: SOLVER.USE_CACHE (default False) turns on the on-disk embedding cache
        # for both frozen encoders. Read defensively so a partial config still constructs.
        use_cache = bool(config.get("SOLVER", {}).get("USE_CACHE", False))

        self.protein_extractor = ProtBertProteinEncoder(protbert_model_path, device, use_cache=use_cache)
        self.protein_feature_dim = self.protein_extractor.output_dim

        self.chemberta_encoder = ChemBERTaEncoder(chemberta_model_path, device, use_cache=use_cache)
        self.chemberta_feature_dim = self.chemberta_encoder.output_dim

        self.gcn_proj_for_cross_attn = nn.Linear(self.gcn_feature_dim, self.chemberta_feature_dim)

        cross_attn_heads = config["BCN"]["HEADS"]
        cross_attn_dropout = config["BCN"]["DROPOUT"]
        self.cross_attn_gc = MultiHeadAttentionLayer(input_dim=self.chemberta_feature_dim, num_heads=cross_attn_heads,
                                                    dropout=cross_attn_dropout, device=device)

        self.fused_nodes_proj_for_protein_attn = nn.Linear(self.chemberta_feature_dim, self.protein_feature_dim)

        attention_input_dim = self.protein_feature_dim
        ban_heads = config["BCN"]["HEADS"]
        ban_dropout = config["BCN"]["DROPOUT"]
        self.bcn = MultiHeadAttentionLayer(input_dim=attention_input_dim, num_heads=ban_heads,
                                           dropout=ban_dropout, device=device)

        mlp_in_dim = self.protein_feature_dim
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

        self.random_layer = None

        # ablation-residual: ABLATION.FUSION_RESIDUAL (default False). Read defensively so a
        # partial config still constructs, and built LAST so that when the flag is off no
        # module is created at all - the parameter set, the init RNG stream and therefore
        # every downstream number are bit-identical to the baseline.
        self.use_fusion_residual = bool(config.get("ABLATION", {}).get("FUSION_RESIDUAL", False))
        if self.use_fusion_residual:
            self.fusion_residual_norm = nn.LayerNorm(self.chemberta_feature_dim)

    def forward(self, bg_d, smiles_sequences, protein_sequences, mode="train"):
        v_d_graph_nodes = self.drug_extractor(bg_d)
        # mask-fix: this used to be (arange(max_nodes) < bg_d.batch_num_nodes()). The dataloader
        # pads every molecule to DRUG.MAX_NODES *before* batching, so batch_num_nodes() returns
        # max_nodes for every graph and that comparison was unconditionally true: the mask was
        # all-ones, masked_mean_pooling averaged ~275 padding rows alongside the real atoms, and
        # the query axis of both attention layers ran from virtual nodes as well as real ones.
        # The real per-molecule count is only known at graph construction, so the dataloader now
        # records it per node as ndata['node_mask'] (bool, True for a real atom); it survives
        # MolecularGCN's ndata.pop('h'). Same [B, N, 1] shape and float/bool pair as before, so
        # every downstream consumer is unchanged.
        batch_size, max_gcn_nodes = v_d_graph_nodes.shape[0], v_d_graph_nodes.shape[1]
        gcn_node_mask_bool = bg_d.ndata['node_mask'].view(batch_size, max_gcn_nodes, 1).to(self.device)
        gcn_node_mask = gcn_node_mask_bool.to(v_d_graph_nodes.dtype)

        v_d_chembl_tokens, chemberta_mask = self.chemberta_encoder(smiles_sequences)
        max_seq_len_c = v_d_chembl_tokens.shape[1]
        chemberta_mask_expanded_bool = chemberta_mask.unsqueeze(1).bool().to(self.device)

        v_d_graph_proj = self.gcn_proj_for_cross_attn(v_d_graph_nodes)
        cross_attn_mask_bool = torch.bmm(gcn_node_mask_bool.float(), chemberta_mask_expanded_bool.float()).bool()

        v_d_fused_nodes, cross_attention_weights = self.cross_attn_gc(
            v_d_graph_proj, v_d_chembl_tokens, v_d_chembl_tokens, mask=cross_attn_mask_bool.unsqueeze(1)
        )

        # ablation-residual: v_d_fused_nodes above is attention(A) @ value(V), and V comes only
        # from the ChemBERTa tokens - v_d_graph_proj enters solely as the query, so none of the
        # GCN's structural representation survives into the fused nodes. Adding the projected
        # GCN embedding back as a residual (then LayerNorm, to keep the scale of the sum in the
        # range the downstream projection was trained for) preserves that signal.
        if self.use_fusion_residual:
            v_d_fused_nodes = self.fusion_residual_norm(v_d_fused_nodes + v_d_graph_proj)

        v_d_nodes_proj = self.fused_nodes_proj_for_protein_attn(v_d_fused_nodes)

        v_p, protein_attention_mask = self.protein_extractor(protein_sequences)
        max_seq_len_p = v_p.shape[1]
        protein_token_mask_expanded_bool = protein_attention_mask.unsqueeze(1).bool().to(self.device)


        ban_mask_bool = torch.bmm(gcn_node_mask_bool.float(), protein_token_mask_expanded_bool.float()).bool()


        f_seq, ban_attention_weights = self.bcn(
            v_d_nodes_proj,
            v_p,
            v_p,
            mask=ban_mask_bool.unsqueeze(1)
        )

        f_pooled = masked_mean_pooling(f_seq, gcn_node_mask)
        
        score = self.mlp_classifier(f_pooled)

        if mode == "train":
             return v_d_nodes_proj, v_p, f_pooled, score
        elif mode == "eval":
            return score, ban_attention_weights


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None, max_nodes=None):
        super().__init__()
        if max_nodes is None:
             raise ValueError("max_nodes must be provided to MolecularGCN")
        self.max_nodes = max_nodes

        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                padding_idx = in_feats - 1
                if self.init_transform.weight.shape[0] > padding_idx:
                     self.init_transform.weight[padding_idx].fill_(0)


        if activation is None:
            activation = F.relu
        elif isinstance(activation, str):
             if activation.lower() == 'relu':
                  activation = F.relu
             else:
                  raise ValueError(f"Unsupported activation string: {activation}")

        if hidden_feats is None:
             hidden_feats = [dim_embedding]
        gnn_activation = [activation] * len(hidden_feats)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=gnn_activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = node_feats.to(self.init_transform.weight.dtype)
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)

        batch_size = batch_graph.batch_size
        total_num_nodes = node_feats.shape[0]

        expected_total_nodes = batch_size * self.max_nodes
        if total_num_nodes != expected_total_nodes:
             raise ValueError(f"Total nodes in batch graph ({total_num_nodes}) "
                              f"does not match batch size ({batch_size}) * max_nodes ({self.max_nodes}). "
                              f"Check dataloader padding.")

        node_feats = node_feats.view(batch_size, self.max_nodes, self.output_feats)

        return node_feats


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):                      # iter1 - DEAD CODE (CO-07): unused, kept intentionally
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):                         # cdan
    def __init__(self, input_dim_list, output_dim=256, device=None):
        super().__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = nn.ParameterList([nn.Parameter(torch.randn(input_dim_list[i], output_dim, device=device), requires_grad=False) for i in range(self.input_num)])


    def forward(self, input_list):
        return_list = [torch.mm(input_list[i].to(self.random_matrix[j].device), self.random_matrix[j]) for j, i in enumerate(range(self.input_num))]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single.to(return_tensor.device))
        return return_tensor