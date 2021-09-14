import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evaluation import *
import matplotlib.colors as mcolors
import os

linestyle_tuple = [
# ('loosely dotted', (0, (1, 10))),
# ('dotted', (0, (1, 1))),
('solid','solid'), 
('dashed','dashed'), 
('dashdot','dashdot'), 
('dotted','dotted'),
# ('densely dotted', (0, (1, 2))), 
# ('loosely dashed', (0, (5, 10))),
# ('dashed', (0, (5, 5))),
('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
('densely dashdotted', (0, (3, 1, 1, 1))),
('densely dashed', (0, (5, 1))), ('loosely dashdotted', (0, (3, 10, 1, 10))),
('dashdotted', (0, (3, 5, 1, 5))),
('densely dashdotted', (0, (3, 1, 1, 1))), ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

class Config(mt.BaseExpConfs):
    def __init__(self):
        super().__init__()
        self.exp_flag = 'demo'
        self.snr = None
        self.device = 0

class RFFExperiments(mt.BaseExperiment):
    def __init__(self):
        mt.BaseExperiment.__init__(self, self)
        self.preprocessing()

    def main(self, *args, **kwargs):
        trainer_keys = self.trainers.keys()
        dataset_keys = self.datasets.keys()
        for trainer_key in trainer_keys:
            trainer = self.trainers[trainer_key](device=self.device)
            trainer.run(load_best=True, retrain=False, is_del_loger=True)
            self.input_idx = trainer.data_idx
            self.logs = {}
            self.logs['trainer'] = trainer_key
            self.logs['param num'] = mt.utils.params_count(trainer.models['C'])
            self.print_logs()
            for dataset_key in dataset_keys:
                self.logs = {}
                self.eval(trainer, trainer_key, dataset_key)
                self.ROC(trainer_key, dataset_key)
                self.logs['dataset'] = dataset_key
                self.print_logs()
            del trainer


    def eval(self, trainer, trainer_name, eval_dataset):
        trainer.models['C'].eval()
        device = trainer.device
        correct = 0.0
        test_loss = 0.0
        feature_list = []
        label_list = []

        with torch.no_grad():
            for data in self.dataloaders[eval_dataset]:
                x, y = data[trainer.data_idx], data[1]
                x, y = x.to(device), y.to(device)
                N = len(x)
                features = trainer.models['C'].features(x)
                feature_list.append(features)
                label_list.append(y)

                if 'close' in eval_dataset:
                    scores = trainer.models['C'].output(features)
                    test_loss += F.cross_entropy(scores, y, reduction='sum').item()
                    pred_y = torch.argmax(scores, dim=1)
                    correct += torch.sum(pred_y == y).item()

        features = torch.cat(feature_list, dim=0).cpu().detach()
        labels = torch.cat(label_list).cpu().detach()
        self.results[trainer_name][eval_dataset]['features'] = features
        self.results[trainer_name][eval_dataset]['labels'] = labels

        if 'close' in eval_dataset:
            acc = correct / len(self.datasets[eval_dataset])
            test_loss = test_loss/ len(self.datasets[eval_dataset])
            self.results[trainer_name][eval_dataset]['acc'] = acc
            self.results[trainer_name][eval_dataset]['test_loss'] = test_loss
            self.logs['Test Loss'] = test_loss
            self.logs['acc'] = acc
            self.logs['data'] = eval_dataset
            self.print_logs()
            self.logs = {}


    def ROC(self, trainer, dataset, is_save_dist=True):
        features = self.results[trainer][dataset]['features'].numpy()
        labels = self.results[trainer][dataset]['labels'].numpy()
        intra_dist, inter_dist = inter_intra_dist(features, labels)
        
        eer, roc_auc, thresh = get_auc_eer(intra_dist, inter_dist, plot_roc=False)
        if is_save_dist:
            self.results[trainer][dataset]['intra_dist'] = intra_dist
            self.results[trainer][dataset]['inter_dist'] = inter_dist
        self.results[trainer][dataset]['eer'] = eer
        self.results[trainer][dataset]['roc_auc'] = roc_auc
        self.results[trainer][dataset]['thresh'] = thresh
        self.logs['auc'] = roc_auc
        self.logs['eer'] = eer
        self.logs['thresh'] = thresh

    def dist_hist_plots(self, trainer_keys, dataset_keys):
        for trainer in trainer_keys:
            for dataset in dataset_keys:
                intra_dist = self.results[trainer][dataset]['intra_dist']
                inter_dist = self.results[trainer][dataset]['inter_dist']
                distance_hist_plot(
                    intra_dist, inter_dist, filename= self.exp_path + '/Final_{}_{}_{}_dist_hist.png'.format(self.flag, trainer, dataset))

    def roc_plots(self, trainer_keys, dataset_keys, name_dict, file_name='ROC.png'):
        plt.figure(figsize=(4,4), dpi=300)
        lw = 2
        filename= os.path.join(self.exp_path, file_name)
        colour_idxs = list(mcolors.TABLEAU_COLORS)
        linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
        ci = 0
        # colours = ['blue', 'green', 'red', 'black', 'yellow', 'orange']
        for trainer in trainer_keys:
            for dataset in dataset_keys:
                
                intra_dist = self.results[trainer][dataset]['intra_dist']
                inter_dist = self.results[trainer][dataset]['inter_dist']
                inter_label = np.ones_like(inter_dist)
                intra_label = np.zeros_like(intra_dist)
                y_test = np.append(inter_label, intra_label)
                y_score = np.append(inter_dist, intra_dist)
                fpr, tpr, threshold = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                thresh = interp1d(fpr, threshold)(eer)
                print(trainer, dataset, 'auc:{}'.format(roc_auc), 'eer:{}'.format(eer))
                if name_dict is None:
                    label_name = trainer
                else:
                    label_name = name_dict[trainer]
                plt.plot(
                    fpr, tpr, color=mcolors.TABLEAU_COLORS[colour_idxs[ci]], 
                    linestyle=linestyle_tuple[int(ci%7)][1],
                        lw=lw, label='AUC:{:.2f} EER:{:.2f} {}'.format(roc_auc, eer, label_name))
                ci += 1
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC {}'.format(dataset_keys[0]))
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def pr_plots(self, trainer_keys, dataset_keys, name_dict, file_name='PR.png'):
        plt.figure(figsize=(4,4), dpi=300)
        lw = 2
        filename= os.path.join(self.exp_path, file_name)
        colour_idxs = list(mcolors.TABLEAU_COLORS)
        linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
        ci = 0
        # colours = ['blue', 'green', 'red', 'black', 'yellow', 'orange']
        for trainer in trainer_keys:
            for dataset in dataset_keys:
                
                intra_dist = self.results[trainer][dataset]['intra_dist']
                inter_dist = self.results[trainer][dataset]['inter_dist']
                inter_label = np.ones_like(inter_dist)
                intra_label = np.zeros_like(intra_dist)
                y_test = np.append(inter_label, intra_label)
                y_score = np.append(inter_dist, intra_dist)
                precision, recall, threshold = precision_recall_curve(y_test, y_score)
                pr_auc = auc(recall, precision)
                # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                # thresh = interp1d(fpr, threshold)(eer)
                print(trainer, dataset, 'pr_auc:{}'.format(pr_auc))
                if name_dict is None:
                    label_name = trainer
                else:
                    label_name = name_dict[trainer]
                plt.plot(
                    recall, precision, color=mcolors.TABLEAU_COLORS[colour_idxs[ci]], 
                    linestyle=linestyle_tuple[int(ci%7)][1],
                        lw=lw, label='auc:{:.2f} {}'.format(pr_auc, label_name))
                ci += 1
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.title('ROC {}'.format(dataset_keys[0]))
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def snr_auc_plots(self, trainer_keys, dataset_keys, snr_list, name_dict=None, file_name='snr_auc.png'):
        plt.figure(figsize=(4,4), dpi=300)
        lw = 2
        filename= os.path.join(self.exp_path, file_name)
        colour_idxs = list(mcolors.TABLEAU_COLORS)
        linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
        # colours = ['blue', 'green', 'red', 'black', 'yellow', 'orange']
        best_models = {}
        for dataset in dataset_keys:
            ci = 0
            for trainer in trainer_keys:
                auc_list = []
                for snr in snr_list:
                    auc_list.append(self.results[trainer][dataset][snr]['roc_auc'])
                    if auc_list[-1] > best_models.setdefault(dataset, ['', 0.0, 0.0])[1]:
                        best_models[dataset] = [trainer, auc_list[-1], snr]
                if name_dict is None:
                    label_name = trainer
                else:
                    label_name = name_dict[trainer]
                plt.plot(
                    snr_list, auc_list, marker=markers[int(ci%7)], 
                    color=mcolors.TABLEAU_COLORS[colour_idxs[int(ci%len(colour_idxs))]], 
                    linestyle=linestyle_tuple[int(ci%7)][1],
                    lw=lw, label='{}'.format(label_name))
                # plt.scatter(snr_list, auc_list, color=mcolors.TABLEAU_COLORS[colour_idxs[int(ci%len(colour_idxs))]], label='{}'.format(trainer_type))
                # plt.scatter(param_list, auc_list, label='{}'.format(trainer_type))
                ci += 1
        print(best_models)
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel('SNR (dB)')
        plt.ylabel('ROC-AUC')
        # plt.title('SNR-AUC {}'.format(dataset_keys[0]))
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def param_auc_plots(self, trainer_dict, dataset_keys, name_dict=None, file_name=None):
        plt.figure(figsize=(4,4), dpi=300)
        lw = 2
        filename= os.path.join(self.exp_path, file_name)
        colour_idxs = list(mcolors.TABLEAU_COLORS)
        linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
        best_models = {}
        ci = 0
        for trainer_type, trainer_list in trainer_dict.items():
            param_list = []
            auc_list = []
            for trainer in trainer_list:
                for dataset in dataset_keys:
                    param_list.append((self.results[trainer]['params']-513*54)/1000000) 
                    auc_list.append(self.results[trainer][dataset]['roc_auc'])
                    if auc_list[-1] > best_models.setdefault(dataset, ['', 0.0, 0.0])[1]:
                        best_models[dataset] = [trainer, auc_list[-1], param_list[-1]]
            if name_dict is None:
                label_name = trainer_type
            else:
                label_name = name_dict[trainer_type]
            plt.scatter(param_list, auc_list, 
                color=mcolors.TABLEAU_COLORS[colour_idxs[int(ci%len(colour_idxs))]], marker=markers[int(ci%7)],
                label='{}'.format(label_name))
            # plt.scatter(param_list, auc_list, label='{}'.format(trainer_type))
            ci += 1
        print(best_models)
        plt.xlabel('# Params (M)')
        plt.ylabel('ROC-AUC')
        plt.ylim(0.95,1.005)
        # plt.ylim(0.990,1.001)
        # plt.title('Params-AUC {}'.format(dataset_keys[0]))
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


    def acc_plots(self, trainer_keys, dataset_keys):
        pass


if __name__ == "__main__":
    exp = Experiments()
    exp.run(is_rerun=True, is_del_loger=True)
    # exp.dist_hist_plots(exp.trainers.keys(), exp.datasets.keys())
    eval_trainer = [
        'NS+CNN+arcface',
        'NS+CNN+softmaxL',
        'NS+CNN+softmax',
        'CNN+softmax',
        'TS+CNN+softmax',
        'TS+YJB+softmax'
    ]
    exp.roc_plots(eval_trainer, ['open'], 'ROC_{}-open.png'.format(exp.flag))
    exp.roc_plots(eval_trainer, ['open-all'], 'ROC_{}-open-all.png'.format(exp.flag))
    exp.roc_plots(eval_trainer, ['open8-9'], 'ROC_{}-open8-9.png'.format(exp.flag))
    exp.roc_plots(eval_trainer, ['open10-11'], 'ROC_{}-open10-11.png'.format(exp.flag))
    exp.roc_plots(eval_trainer, ['open8-11'], 'ROC_{}-open8-11.png'.format(exp.flag))

    # print(exp.results)

    # ['CNN+softmax', 'NS+CNN+arcface(old)', 'NS+CNN+arcface', 'NS+CNN+softmax', 'NS+CNN+softmax(time)', 'TS+CNN+softmax', 'TS+YJB+softmax', 'CNS+CNN+softmax(old)', 'CNS+CNN+arcface(old)']