try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False

from trainer import Trainer
from models import CMA
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from domain_adaptator import Discriminator
from models import RandomLayer


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import time
from time import time
import argparse
import warnings, os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="AttnESM-DTI for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
# iter1 - FIXED (C-07, C-08): added cold_drug/cold_protein; dropped the dangling 'cold' choice,
# which argparse accepted even though no datasets/*/cold directory has ever existed.
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'cold_drug', 'cold_protein', 'cluster'])
parser.add_argument('--num_runs', default=1, type=int, help="Number of independent runs")
# iter3 - FIXED (C-11): the default was a hardcoded 2048 that always won, so SOLVER.SEED in the
# yaml was silently dead. None means "not passed", and the seed then comes from the config;
# passing --start_seed explicitly still overrides it.
parser.add_argument('--start_seed', default=None, type=int,
                    help="Starting seed for independent runs (default: SOLVER.SEED from --cfg)")
# iter3 - FIXED (C-12): escape hatch for the auto <OUTPUT_DIR>/<data>_<split> layout below.
parser.add_argument('--output_dir', default=None, type=str,
                    help="explicit output directory (default: <RESULT.OUTPUT_DIR>/<data>_<split>)")
# iter1 - FIXED (C-05): encoder checkpoints come from the CLI; they were hardcoded to
# /home/qinchi/... absolute paths that exist only on the original author's machine.
parser.add_argument('--esm_path', required=True, type=str,
                    help="path to the local ESM-2 checkpoint directory")
parser.add_argument('--chemberta_path', required=True, type=str,
                    help="path to the local ChemBERTa checkpoint directory")
# iter1 - FIXED (C-06): optional explicit override, e.g. --device cuda:1 on a multi-GPU host
parser.add_argument('--device', default=None, type=str,
                    help="torch device override (default: cuda if available, else cpu)")
# iter3 - FIXED (C-13): there was no way to override any config key from the command line;
# --cfg was the only config input, so every ablation needed a hand-edited yaml. Trailing
# KEY VALUE pairs are merged over the yaml, e.g.  ... --data biosnap ABLATION.ATTN_POOLING True
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                    help="config overrides as trailing KEY VALUE pairs, e.g. ABLATION.ATTN_POOLING True")

args = parser.parse_args()

if args.opts and len(args.opts) % 2 != 0:
    parser.error(f"config overrides must be KEY VALUE pairs, got an odd number: {args.opts}")

# iter1 - FIXED (C-06): was hardcoded to 'cuda:1', an invalid ordinal on single-GPU hosts
# (the original ternary's middle branch was unreachable).
device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# iter1 - FIXED (C-03): the CDAN domain discriminator is a 2-way source/target classifier.
# cross_entropy_logits does log_softmax(dim=1) and indexes [:, 1], so the old 1-logit head
# raised IndexError and would have produced an identically-zero loss. DECODER.BINARY still
# sizes the DTI prediction head and the CDAN multilinear map; only the domain head changes.
DA_DOMAIN_CLASSES = 2


def resolve_output_dir(cfg, args, seed):
    """Give every (dataset, split) its own results folder.

    iter3 - FIXED (C-12): RESULT.OUTPUT_DIR used to be the full, hardcoded destination, so
    back-to-back runs over different datasets/splits all wrote their checkpoints, metrics and
    markdown tables into the same directory and silently overwrote each other. It is now the
    *base* results directory and the per-run leaf is appended here.
    """
    if args.output_dir is not None:
        base = args.output_dir
    else:
        base = os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}_{args.split}")
    # within a sweep each seed needs its own leaf for the same reason
    return os.path.join(base, f"seed{seed}") if args.num_runs > 1 else base


def run_single_experiment(cfg, args, device, seed, esm_model_path, chemberta_model_path):
    # iter3 - the caller now hands us a fully resolved cfg (yaml merged, OUTPUT_DIR namespaced),
    # so re-merging args.cfg here would just undo the OUTPUT_DIR it computed.
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    set_seed(seed)
    print(f"\nRunning experiment with seed: {seed}")

    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)

    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}")
    print(f"Writing results to: {cfg.RESULT.OUTPUT_DIR}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, 'train.csv')
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
    else:
        train_source_path = os.path.join(dataFolder, 'source_train.csv')
        train_target_path = os.path.join(dataFolder, 'target_train.csv')
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
            "Seed": seed,
            "ESM_path": esm_model_path,
            "ChemBERTa_path": chemberta_model_path,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LR,
                "DA_lambda_init": cfg.DA.LAMB_DA
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{args.split}_{suffix}_seed{seed}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    model = CMA(protbert_model_path=esm_model_path,
                    chemberta_model_path=chemberta_model_path,
                    device=device, **cfg).to(device)

    opt_da = None
    domain_dmm = None

    # iter1 - FIXED (C-03): Discriminator is built with n_class=DA_DOMAIN_CLASSES (2) below.
    if cfg.DA.USE:
        fused_feature_dim_for_da = cfg["PROTEIN"]["ESM_FEATURE_DIM"]
        n_class_for_da = cfg["DECODER"]["BINARY"]
        cdan_h_dim_for_da = fused_feature_dim_for_da * n_class_for_da

        if cfg["DA"]["RANDOM_LAYER"]:
             if cfg["DA"]["ORIGINAL_RANDOM"]:
                 random_layer = RandomLayer(input_dim_list=[fused_feature_dim_for_da, n_class_for_da],
                                            output_dim=cfg["DA"]["RANDOM_DIM"],
                                            device=device)
             else:
                  random_layer = nn.Linear(in_features=cdan_h_dim_for_da,
                                           out_features=cfg["DA"]["RANDOM_DIM"],
                                           bias=False).to(device)
                  torch.nn.init.normal_(random_layer.weight, mean=0, std=1)
                  for param in random_layer.parameters():
                      param.requires_grad = False
             domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=DA_DOMAIN_CLASSES).to(device)
             model.random_layer = random_layer
        else:
             domain_dmm = Discriminator(input_size=cdan_h_dim_for_da,
                                       n_class=DA_DOMAIN_CLASSES).to(device)
             model.random_layer = None

        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = None
        domain_dmm = None


    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device,
                      training_generator if not cfg.DA.USE else multi_generator,
                      val_generator,
                      test_generator,
                      opt_da=opt_da,
                      discriminator=domain_dmm,
                      experiment=experiment,
                      alpha=cfg["DA"]["LAMB_DA"] if cfg.DA.USE else 1.0,
                      **cfg)

    result = trainer.train()

    return result


if __name__ == '__main__':
    s = time()

    # iter1 - FIXED (C-05): sourced from --esm_path / --chemberta_path
    esm_local_model_path = args.esm_path
    chemberta_local_model_path = args.chemberta_path


    all_results = []


    for i in range(args.num_runs):
        cfg_for_run = get_cfg_defaults()
        cfg_for_run.merge_from_file(args.cfg)
        # iter3 - FIXED (C-13): CLI overrides win over the yaml. Merged before the seed and
        # OUTPUT_DIR are read below, so SOLVER.SEED and RESULT.OUTPUT_DIR are overridable too.
        if args.opts:
            cfg_for_run.merge_from_list(args.opts)

        # iter3 - FIXED (C-11): fall back to the config's seed when --start_seed is not given
        base_seed = args.start_seed if args.start_seed is not None else cfg_for_run.SOLVER.SEED
        current_seed = base_seed + i

        # iter3 - FIXED (C-12)
        cfg_for_run.RESULT.OUTPUT_DIR = resolve_output_dir(cfg_for_run, args, current_seed)

        single_run_result = run_single_experiment(cfg_for_run,
                                                  args,
                                                  device,
                                                  current_seed,
                                                  esm_local_model_path,
                                                  chemberta_local_model_path)
        all_results.append(single_run_result)

    e = time()
    print(f"\nTotal running time for {args.num_runs} runs: {round(e - s, 2)}s")

    if args.num_runs > 1:
        print("\nOverall Results (Mean ± Standard Deviation):")

        metrics_to_report = ['auroc', 'auprc', 'accuracy', 'F1', 'sensitivity', 'specificity', 'Precision', 'test_loss']
        other_metrics_to_report = ['thred_optim', 'best_epoch']

        for metric_key in metrics_to_report:
            if all_results and metric_key in all_results[0]:
                scores = [res[metric_key] for res in all_results]
                if all(isinstance(s, (int, float)) for s in scores):
                    mean_score = np.mean(scores)
                    std_score = np.std(scores, ddof=1) if args.num_runs > 1 else 0.0
                    print(f"{metric_key}: {mean_score:.4f} ± {std_score:.4f}")
                else:
                     print(f"{metric_key}: {scores}")
            else:
                 print(f"Warning: Metric '{metric_key}' not found in trainer results.")

        for metric_key in other_metrics_to_report:
             if all_results and metric_key in all_results[0]:
                  scores = [res[metric_key] for res in all_results]
                  print(f"{metric_key}: {scores}")
             else:
                 print(f"Warning: Metric '{metric_key}' not found in trainer results.")


    elif args.num_runs == 1 and all_results:
         print("\nSingle Run Result:")
         for key, value in all_results[0].items():
             print(f"{key}: {value}")