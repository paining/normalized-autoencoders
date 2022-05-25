"""
evaluate OOD detection performance through AUROC score

Example:
    python evaluate_cifar_ood.py --dataset FashionMNIST_OOD \
            --ood MNIST_OOD,ConstantGray_OOD \
            --resultdir results/fmnist_ood_vqvae/Z7K512/e300 \
            --ckpt model_epoch_280.pkl \
            --config Z7K512.yml \
            --device 1
"""
import os
import yaml
import argparse
import copy
import torch
import numpy as np
from torch.utils import data
from models import get_model, load_pretrained
from loaders import get_dataloader

from utils import roc_btw_arr, batch_run, search_params_intp, parse_unknown_args, parse_nested_args
from utils import roc_curve_arr, get_optimal


parser = argparse.ArgumentParser()
parser.add_argument('--resultdir', type=str, help='result dir. results/... or pretrained/...')
parser.add_argument('--config', type=str, help='config file name')
parser.add_argument('--ckpt', type=str, help='checkpoint file name to load. default', default=None)
parser.add_argument('--ood', type=str, help='list of OOD datasets, separated by comma')
parser.add_argument('--device', type=str, help='device')
parser.add_argument('--dataset', type=str, choices=['MNIST','MNIST_OOD', 'CIFAR10_OOD', 'ImageNet32', 'FashionMNIST_OOD',
                                                    'FashionMNISTpad_OOD'],
                    default='MNIST', help='inlier dataset dataset')
parser.add_argument('--aug', type=str, help='pre-defiend data augmentation', choices=[None, 'CIFAR10', 'CIFAR10-OE'])
parser.add_argument('--method', type=str, choices=[None, 'likelihood_regret', 'input_complexity', 'outlier_exposure'])
args, unknown = parser.parse_known_args()
d_cmd_cfg = parse_unknown_args(unknown)
d_cmd_cfg = parse_nested_args(d_cmd_cfg)
print(d_cmd_cfg)


# load config file
cfg = yaml.load(open(os.path.join(args.resultdir, args.config)), Loader=yaml.FullLoader)
result_dir = args.resultdir
if args.ckpt is not None:
    ckpt_file = os.path.join(result_dir, args.ckpt)
else:
    raise ValueError(f'ckpt file not specified')

print(f'loading from {ckpt_file}')
l_ood = [s.strip() for s in args.ood.split(',')]
device = f'cuda:{args.device}'

#print(f'loading from : {ckpt_file}')


def evaluate(m, in_dl, out_dl, device):
    """computes OOD detection score"""
    in_pred = batch_run(m, in_dl, device, method='predict')
    out_pred = batch_run(m, out_dl, device, method='predict')
    auc = roc_btw_arr(out_pred, in_pred)
    return auc


# load dataset
print('ood datasets')
print(l_ood)
if args.dataset in  {'MNIST','MNIST_OOD', 'FashionMNIST_OOD'}:
#if args.dataset in {'MNIST_OOD', 'FashionMNIST_OOD'}:
    size = 28
    channel = 1
else:
    size = 32
    channel = 3
data_dict = {'path': 'datasets',
             'size': size,
             'channel': channel,
             'batch_size': 64,
             'n_workers': 4,
             'split': 'evaluation',
             'path': 'datasets'}


data_dict_ = copy.copy(data_dict)
data_dict_['dataset'] = args.dataset
in_dl = get_dataloader(data_dict_)

l_ood_dl = []
for ood_name in l_ood:
    data_dict_ = copy.copy(data_dict)
    data_dict_['dataset'] = ood_name 
    dl = get_dataloader(data_dict_)
    l_ood_dl.append(dl)

model = get_model(cfg).to(device)
ckpt_data = torch.load(ckpt_file)
if 'model_state' in ckpt_data:
    model.load_state_dict(ckpt_data['model_state'])
else:
    model.load_state_dict(torch.load(ckpt_file))

model.eval()
model.to(device)

in_pred = batch_run(model, in_dl, device=device, no_grad=False)

l_ood_pred = []
for dl in l_ood_dl:
    out_pred = batch_run(model, dl, device=device, no_grad=False)
    l_ood_pred.append(out_pred)

l_ood_auc = []
for pred in l_ood_pred:
    l_ood_auc.append(roc_btw_arr(pred, in_pred))
####### SJY 2022-05-24
"""TPR,FPR,Threshold"""
l_ood_tpr = []
l_ood_fpr = []
l_ood_threshold = []
for pred in l_ood_pred:
    tpr , fpr, threshold = roc_curve_arr(pred,in_pred)
    l_ood_tpr.append(tpr)
    l_ood_fpr.append(fpr)
    l_ood_threshold.append(threshold)
l_ood_tpr = np.concatenate(np.array(l_ood_tpr))
l_ood_fpr = np.concatenate(np.array(l_ood_fpr))
l_ood_threshold = np.concatenate(np.array(l_ood_threshold))
########

print('OOD Detection Results in AUC')
for ds, auc in zip(l_ood, l_ood_auc):
    print(f'{ds}:{auc:.4f}')

######## SJY 2022-05-24
"""Optimal Trheshold"""
optimal_threshold = get_optimal(pred,in_pred)
print("Optimal Threshold :",optimal_threshold)
########

######## SJY 2022-05-24
"""ROC 커브 시각화"""
import matplotlib.pyplot as plt
plt.plot(fpr,tpr,color='darkorange',linewidth=2,label='ROC curve(area=%0.4f)'% l_ood_auc[0])
plt.plot([0,1],[0,1],color='navy',linewidth=2,linestyle='--')
plt.title('Receiver operating characteristic curve')
plt.xlim([-0.01,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('/home/jyshin/test/mlp_test/roc_1.png')
plt.close()
########

######## SJY 2022-05-24
"""pred 값 출력 테스트"""
print(torch.min(in_pred))
print(torch.max(in_pred))
print(torch.mean(in_pred))
#print(l_ood_pred[0].shape)
print(torch.min(l_ood_pred[0]))
print(torch.max(l_ood_pred[0]))
print(torch.mean(l_ood_pred[0]))
########

########## SJY 2022-05-24
"""FP 이미지 저장"""
from tqdm import tqdm
count = 0
tot = 0
for batch in tqdm(in_dl):
    x = batch[0]
    with torch.no_grad():
        pred_ = model.predict(x.cuda(device)).detach().cpu()
        tot += len(pred_)
        for idx in range(0,len(pred_)):
            if pred_[idx] > 0.035477214:
                count += 1
                plt.imshow(x[idx,0,:,:])
                plt.savefig('/home/jyshin/test/mlp_test/FPR'+str(count)+'.png')
                plt.close()
print(count)
print(tot)
###########

########### SJY 2022-05-24
"""FN 이미지 저장"""
count = 0
tot = 0
for dl in l_ood_dl:
    for batch in tqdm(dl):
        x = batch[0]
        with torch.no_grad():
            pred_ = model.predict(x.cuda(device)).detach().cpu()
            tot += len(pred_)
            for idx in range(0,len(pred_)):
                if pred_[idx] < 0.035477214:
                    count += 1
                    plt.imshow(x[idx,0,:,:])
                    plt.savefig('/home/jyshin/test/mlp_test/FN'+str(count)+'.png')
                    plt.close()
print(count)
print(tot)
###########