import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score, accuracy_score, recall_score, f1_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from domain_adaptator import ReverseLayerF

from prettytable import PrettyTable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha

        self.n_class = config["DECODER"]["BINARY"]

        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            self.random_layer = getattr(model, 'random_layer', None)

        self.fused_feature_dim = config["PROTEIN"]["ESM_FEATURE_DIM"]
        self.cdan_h_dim = self.fused_feature_dim * self.n_class


        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "F1", "Recall", "Precision", "Accuracy", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "DA_Loss", "DA_Lambda"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        if non_init_epoch <= 0:
             return self.init_lamb_da if self.current_epoch >= self.da_init_epoch else 0.0
        
        p = (self.current_epoch - self.da_init_epoch) / non_init_epoch
        p = max(0.0, min(1.0, p))
        
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        grow_fact = (grow_fact + 1) / 2.0

        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        da_loss, epoch_lamb]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                         self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
                         self.experiment.log_metric("train_epoch da lambda", epoch_lamb, epoch=self.current_epoch)


            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss, accuracy, precision, recall, f1 = self.test(dataloader="val")

            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch accuracy", accuracy, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch precision", precision, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch recall", recall, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch f1", f1, epoch=self.current_epoch)


            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, f1, recall, precision, accuracy, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            if auroc >= self.best_auroc:
                self.best_model = type(self.model)(**self.model.init_params).to(self.device)
                self.best_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                print(f"New best model found at epoch {self.best_epoch} with AUROC: {self.best_auroc:.4f}")


            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss),
                  " AUROC " + str(auroc) + " AUPRC " + str(auprc) +
                  " F1 " + str(f1) + " Recall " + str(recall) + " Precision " + str(precision) +
                  " Accuracy " + str(accuracy))

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")

        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim) + " F1 " + str(f1) + " Precision " + str(precision))

        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision

        self.save_result()

        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])

        return self.test_metrics


    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []

        if dataloader == "test":
            data_loader = self.test_dataloader
            model_to_eval = self.best_model if self.best_model else self.model
            if model_to_eval is None:
                 print("Warning: No model available for testing.")
                 return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        elif dataloader == "val":
            data_loader = self.val_dataloader
            model_to_eval = self.model
        else:
            raise ValueError(f"Error key value {dataloader}. Must be 'val' or 'test'.")

        model_to_eval.eval()
        with torch.no_grad():
            num_batches = len(data_loader)
            if num_batches == 0:
                 print(f"Warning: {dataloader} dataloader is empty.")
                 if dataloader == "test":
                      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0
                 else:
                      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


            for i, (v_d, smiles_sequences, v_p_strings, labels) in enumerate(data_loader):
                v_d = v_d.to(self.device)
                labels = labels.float().to(self.device)

                score, attention_weights = model_to_eval(v_d, smiles_sequences, v_p_strings, mode="eval")

                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)

                test_loss += loss.item()
                y_label.extend(labels.to("cpu").tolist())
                y_pred.extend(n.to("cpu").tolist())

        eval_loss = test_loss / num_batches
        if len(y_label) == 0:
             print(f"Warning: No samples processed in {dataloader} evaluation.")
             if dataloader == "test":
                  return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, eval_loss, 0.5, 0.0
             else:
                  return 0.0, 0.0, eval_loss, 0.0, 0.0, 0.0, 0.0


        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)

        y_pred_binary_05 = [1 if prob >= 0.5 else 0 for prob in y_pred]
        accuracy_05 = accuracy_score(y_label, y_pred_binary_05)
        precision_05 = precision_score(y_label, y_pred_binary_05, zero_division=0)
        recall_05 = recall_score(y_label, y_pred_binary_05, zero_division=0)
        f1_05 = f1_score(y_label, y_pred_binary_05, zero_division=0)


        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec_pr_curve, recall_pr_curve, thresholds_pr_curve = precision_recall_curve(y_label, y_pred)

            precision_at_threshold = tpr / (tpr + fpr + 1e-6)
            f1_at_threshold = 2 * precision_at_threshold * tpr / (tpr + precision_at_threshold + 1e-6)

            valid_f1_at_threshold = f1_at_threshold[:-1] if len(f1_at_threshold) > 1 else f1_at_threshold
            valid_thresholds = thresholds[:-1] if len(thresholds) > 1 else thresholds

            if valid_f1_at_threshold.size > 0:
                 thred_optim = valid_thresholds[np.argmax(valid_f1_at_threshold)]
                 f1_optimal = np.max(valid_f1_at_threshold)
            else:
                 thred_optim = 0.5
                 f1_optimal = f1_05

            y_pred_optimal_binary = [1 if prob >= thred_optim else 0 for prob in y_pred]
            cm_optimal = confusion_matrix(y_label, y_pred_optimal_binary)

            accuracy_optimal = (cm_optimal[0, 0] + cm_optimal[1, 1]) / (sum(sum(cm_optimal)) + 1e-6)
            sensitivity_optimal = cm_optimal[0, 0] / (cm_optimal[0, 0] + cm_optimal[0, 1] + 1e-6)
            specificity_optimal = cm_optimal[1, 1] / (cm_optimal[1, 0] + cm_optimal[1, 1] + 1e-6)
            precision_optimal = precision_score(y_label, y_pred_optimal_binary, zero_division=0)


            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall_pr_curve, prec_pr_curve)

            return auroc, auprc, f1_optimal, sensitivity_optimal, specificity_optimal, accuracy_optimal, eval_loss, thred_optim, precision_optimal

        else:
            return auroc, auprc, eval_loss, accuracy_05, precision_05, recall_05, f1_05

    def save_result(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if self.config["RESULT"]["SAVE_MODEL"] and self.best_model:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))

        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
            "best_epoch": self.best_epoch
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        if self.current_epoch >= self.da_init_epoch:
             entropy = ReverseLayerF.apply(entropy, self.alpha)
             current_lambda = self.da_lambda_decay() if self.current_epoch >= self.da_init_epoch else 0.0
             entropy = ReverseLayerF.apply(entropy, current_lambda)
             entropy_w = 1.0 + torch.exp(-entropy)
        else:
             entropy_w = 1.0 + torch.exp(-entropy)

        return entropy_w


    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, smiles_sequences, v_p, labels) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} Training")):
            self.step += 1
            v_d = v_d.to(self.device)
            labels = labels.float().to(self.device)

            self.optim.zero_grad()

            v_d_fused_aligned, v_p_out, f_pooled, score = self.model(v_d, smiles_sequences, v_p, mode="train")

            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)

        loss_epoch = loss_epoch / num_batches
        print(f'Training at Epoch {self.current_epoch} with training loss {loss_epoch:.4f}')
        return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        if self.domain_dmm:
            self.domain_dmm.train()

        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        da_loss_D_epoch = 0

        epoch_lamb_da = self.da_lambda_decay() if self.current_epoch >= self.da_init_epoch else 0.0
        if self.experiment:
             self.experiment.log_metric("train_epoch da lambda", epoch_lamb_da, epoch=self.current_epoch)


        num_batches = len(self.train_dataloader)
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} DA Training")):
            self.step += 1

            batch_source_data = batch_s[0]
            v_d_s, smiles_sequences_s, v_p_s_strings, labels_s = batch_source_data

            v_d_s = v_d_s.to(self.device)
            labels_s = labels_s.float().to(self.device)


            batch_target_data = batch_t[0]
            v_d_t, smiles_sequences_t, v_p_t_strings, _ = batch_target_data


            v_d_t = v_d_t.to(self.device)


            if self.current_epoch >= self.da_init_epoch:
                 self.optim_da.zero_grad()

                 with torch.no_grad():
                      v_d_s_fused_aligned_detach, v_p_s_out_detach, f_pooled_s_detach, score_s_detach = self.model(v_d_s, smiles_sequences_s, v_p_s_strings, mode="train")

                      v_d_t_fused_aligned_detach, v_p_t_out_detach, f_pooled_t_detach, score_t_detach = self.model(v_d_t, smiles_sequences_t, v_p_t_strings, mode="train")

                 softmax_output_s_detach = torch.nn.Softmax(dim=1)(score_s_detach)
                 softmax_output_t_detach = torch.nn.Softmax(dim=1)(score_t_detach)

                 h_s_detach = torch.bmm(f_pooled_s_detach.unsqueeze(2), softmax_output_s_detach.unsqueeze(1)).view(-1, self.cdan_h_dim)
                 h_t_detach = torch.bmm(f_pooled_t_detach.unsqueeze(2), softmax_output_t_detach.unsqueeze(1)).view(-1, self.cdan_h_dim)

                 if self.random_layer:
                     if isinstance(self.random_layer, nn.Linear):
                          h_s_detach = self.random_layer(h_s_detach)
                          h_t_detach = self.random_layer(h_t_detach)
                     elif isinstance(self.random_layer, RandomLayer):
                         h_s_detach = self.random_layer.forward([f_pooled_s_detach, softmax_output_s_detach])
                         h_t_detach = self.random_layer.forward([f_pooled_t_detach, softmax_output_t_detach])

                     else:
                         pass


                 adv_output_src_score_detach = self.domain_dmm(h_s_detach)
                 adv_output_tgt_score_detach = self.domain_dmm(h_t_detach)

                 if self.use_da_entropy:
                      entropy_src_detach = entropy_logits(score_s_detach)
                      entropy_tgt_detach = entropy_logits(score_t_detach)
                      src_weight_detach = entropy_src_detach / (torch.sum(entropy_src_detach) + 1e-6)
                      tgt_weight_detach = entropy_tgt_detach / (torch.sum(entropy_tgt_detach) + 1e-6)
                 else:
                     src_weight_detach = None
                     tgt_weight_detach = None

                 domain_label_src = torch.zeros(self.batch_size).long().to(self.device)
                 domain_label_tgt = torch.ones(self.batch_size).long().to(self.device)

                 n_src_da_detach, loss_cdan_src_detach = cross_entropy_logits(adv_output_src_score_detach, domain_label_src, src_weight_detach)
                 n_tgt_da_detach, loss_cdan_tgt_detach = cross_entropy_logits(adv_output_tgt_score_detach, domain_label_tgt, tgt_weight_detach)

                 loss_D = loss_cdan_src_detach + loss_cdan_tgt_detach

                 loss_D.backward()
                 self.optim_da.step()
                 da_loss_D_epoch += loss_D.item()


            self.optim.zero_grad()

            v_d_s_fused_aligned, v_p_s_out, f_pooled_s, score_s = self.model(v_d_s, smiles_sequences_s, v_p_s_strings, mode="train")
            softmax_output_s = torch.nn.Softmax(dim=1)(score_s)

            v_d_t_fused_aligned, v_p_t_out, f_pooled_t, score_t = self.model(v_d_t, smiles_sequences_t, v_p_t_strings, mode="train")
            softmax_output_t = torch.nn.Softmax(dim=1)(score_t)

            h_s = torch.bmm(f_pooled_s.unsqueeze(2), softmax_output_s.unsqueeze(1)).view(-1, self.cdan_h_dim)
            h_t = torch.bmm(f_pooled_t.unsqueeze(2), softmax_output_t.unsqueeze(1)).view(-1, self.cdan_h_dim)

            reverse_h_s = ReverseLayerF.apply(h_s, epoch_lamb_da)
            reverse_h_t = ReverseLayerF.apply(h_t, epoch_lamb_da)

            if self.random_layer:
                 if isinstance(self.random_layer, nn.Linear):
                     input_D_s = self.random_layer(reverse_h_s)
                     input_D_t = self.random_layer(reverse_h_t)
                 elif isinstance(self.random_layer, RandomLayer):
                     input_D_s = self.random_layer.forward([f_pooled_s, softmax_output_s])
                     input_D_t = self.random_layer.forward([f_pooled_t, softmax_output_t])
                 else:
                     input_D_s = reverse_h_s
                     input_D_t = reverse_h_t
            else:
                 input_D_s = reverse_h_s
                 input_D_t = reverse_h_t


            domain_label_source_reversed = torch.ones(self.batch_size).long().to(self.device)
            domain_label_target_reversed = torch.zeros(self.batch_size).long().to(self.device)

            if self.use_da_entropy:
                 entropy_src = entropy_logits(score_s)
                 entropy_tgt = entropy_logits(score_t)
                 src_weight = self._compute_entropy_weights(score_s)
                 tgt_weight = self._compute_entropy_weights(score_t)
                 src_weight = src_weight / (torch.sum(src_weight) + 1e-6)
                 tgt_weight = tgt_weight / (torch.sum(tgt_weight) + 1e-6)
            else:
                 src_weight = None
                 tgt_weight = None


            n_src_da_model, loss_cdan_src_model = cross_entropy_logits(self.domain_dmm(input_D_s), domain_label_source_reversed, src_weight)
            n_tgt_da_model, loss_cdan_tgt_model = cross_entropy_logits(self.domain_dmm(input_D_t), domain_label_target_reversed, tgt_weight)

            da_loss_model_objective = loss_cdan_src_model + loss_cdan_tgt_model


            if self.current_epoch >= self.da_init_epoch:
                total_loss = model_loss + epoch_lamb_da * da_loss_model_objective
                da_loss_value = da_loss_model_objective.item()
            else:
                total_loss = model_loss
                da_loss_value = 0.0

            total_loss.backward()
            self.optim.step()

            total_loss_epoch += total_loss.item()
            model_loss_epoch += model_loss.item()
            da_loss_epoch += da_loss_value

            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", total_loss.item(), step=self.step)
                if self.current_epoch >= self.da_init_epoch:
                    self.experiment.log_metric("train_step da loss (Model Objective)", da_loss_value, step=self.step)
                    self.experiment.log_metric("train_step da loss (Discriminator Objective)", loss_D.item() if self.current_epoch >= self.da_init_epoch else 0.0, step=self.step)


        total_loss_epoch /= num_batches
        model_loss_epoch /= num_batches
        da_loss_epoch /= num_batches
        da_loss_D_epoch /= num_batches

        if self.current_epoch < self.da_init_epoch:
            print(f'Training at Epoch {self.current_epoch} with model training loss {total_loss_epoch:.4f}')
        else:
            print(f'Training at Epoch {self.current_epoch} model training loss {model_loss_epoch:.4f}'
                  + f", DA loss (Model Obj) {da_loss_epoch:.4f}, DA loss (D Obj) {da_loss_D_epoch:.4f}"
                  + f", total training loss {total_loss_epoch:.4f}, DA lambda {epoch_lamb_da:.4f}")

        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da