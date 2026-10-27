from yacs.config import CfgNode as CN

_C = CN()

_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75

_C.DRUG.PADDING = True

_C.DRUG.HIDDEN_LAYERS = [128, 128, 1280]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290
# iter6 - what the GCN atoms are fused with before the protein cross-attention.
#   chemberta     baseline: cross-attention with ChemBERTa SMILES tokens (unchanged)
#   gcn_only      no ChemBERTa, D = LN(GCN)
#   unimol_gated  no ChemBERTa, D = z*LN(GCN) + (1-z)*U, scalar gate z per atom,
#                 U = frozen Uni-Mol per-atom features through a trainable adapter
# Defaults to chemberta so existing configs and runs are unchanged.
_C.DRUG.FUSION = "chemberta"
# Heavy-atom Uni-Mol cache written by scripts/precompute_unimol_heavy.py
# (relative paths are taken from the repo root).
_C.DRUG.UNIMOL_CACHE_DIR = "cache/unimol_heavy"
# A SMILES with no Uni-Mol record: "error" stops the run, "zeros" gives it an
# all-zero Uni-Mol view (the gate can still use its GCN view).
_C.DRUG.UNIMOL_MISSING = "error"

_C.PROTEIN = CN()
# iter1 - DEAD CODE (CO-07): NUM_FILTERS/KERNEL_SIZE/EMBEDDING_DIM/PADDING are CNN-era, unused, kept intentionally
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True
_C.PROTEIN.ESM_FEATURE_DIM = 1280
# iter4 - 3D-04: swap the protein encoder for the cache-backed Protein3DEncoder
# (ESM-IF1 structure embeddings, with an ESM-2 fallback for proteins that have no
# usable AlphaFold model). False keeps the ESM-2 baseline path unchanged. Both
# encoders emit ESM_FEATURE_DIM (1280), so nothing downstream resizes.
_C.PROTEIN.USE_3D = False
# iter4 - 3D-05: which protein encoder to build.
#   esm2    ESM-2 650M per-residue embeddings (the baseline)
#   esmif1  cache-backed Protein3DEncoder (ESM-IF1 structures + ESM-2 fallback)
#   saprot  cache-backed SaProt over Foldseek 3Di tokens (encoders/saprot.py)
# Defaults to esm2 so existing runs are unchanged. USE_3D above is still honoured
# when ENCODER is left at its default, so configs written before this flag
# existed keep selecting esmif1.
_C.PROTEIN.ENCODER = "esm2"
# SaProt cache directory, repo-relative unless absolute. cache/saprot is the full-3Di
# arm; cache/saprot_masked (precompute_saprot.py --mask-all) is the sequence-only arm.
_C.PROTEIN.SAPROT_CACHE_DIR = "cache/saprot"

_C.BCN = CN()
_C.BCN.HEADS = 8
_C.BCN.DROPOUT = 0.2

_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 1280  # iter1 - DEAD CODE (CO-07): unused, kept intentionally (decoder sizes itself from ESM_FEATURE_DIM)
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048
# feature-cache: reuse on-disk embeddings for the frozen ESM-2 / ChemBERTa encoders
# (cache/esm/, cache/chemberta/). False keeps the original recompute-every-batch path.
_C.SOLVER.USE_CACHE = False

_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = False

_C.DA = CN()
_C.DA.TASK = False
_C.DA.METHOD = "CDAN"
_C.DA.USE = False
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
# iter1 - FIXED (C-10): was None, which reached nn.Linear(out_features=None) and
# Discriminator(input_size=None) whenever DA.RANDOM_LAYER was enabled.
_C.DA.RANDOM_DIM = 256
_C.DA.USE_ENTROPY = True

_C.COMET = CN()
_C.COMET.WORKSPACE = "pz-white"
_C.COMET.PROJECT_NAME = "CMA"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()