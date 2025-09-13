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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="AttnESM-DTI for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
parser.add_argument('--num_runs', default=1, type=int, help="Number of independent runs")
parser.add_argument('--start_seed', default=2048, type=int, help="Starting seed for independent runs")

args = parser.parse_args()

ESM_LOCAL_MODEL_PATH = "/home/qinchi/test-dti4/esm2_t33_650M_UR50D"
CHEMBERTA_LOCAL_MODEL_PATH = "/home/qinchi/test-dti3/chemberta_local_model"


def run_single_experiment(cfg, args, device, seed, esm_model_path, chemberta_model_path):
    cfg.merge_from_file(args.cfg)
    print("Hyperparameters:", dict(cfg))

    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    set_seed(seed)
    print(f"\nRunning experiment with seed: {seed}")

    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)

    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

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
             domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
             model.random_layer = random_layer
        else:
             domain_dmm = Discriminator(input_size=cdan_h_dim_for_da,
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
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

    esm_local_model_path = ESM_LOCAL_MODEL_PATH
    chemberta_local_model_path = CHEMBERTA_LOCAL_MODEL_PATH


    all_results = []


    for i in range(args.num_runs):
        current_seed = args.start_seed + i

        cfg_for_run = get_cfg_defaults()

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