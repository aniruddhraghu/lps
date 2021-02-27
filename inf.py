# inf model
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import scipy.stats as stats
import json

from torch.distributions.beta import Beta
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
import torch.nn.functional as F

import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

import pickle
import sys

from models import *
from utils import *
from loss_utils import *

# These two functions are just defined as placeholders for now -- the paths need to be specified appropriately. They are in utils.py
df_reduced_final = get_df()
all_ecgs = get_ecg(df_reduced_final)

all_others = df_reduced_final[['creat_clean', 'age', 'g_proc', 'dem1', 'dem2', 'dem3', 'egfr']].values
all_bphr = df_reduced_final[['bpsys', 'bpdia', 'hr_proc']].values
all_y = df_reduced_final['death_6month_binary'].values

# SPLIT
positive_idxs = np.array([i for i in np.arange(len(all_y)) if all_y[i] == 1])
negative_idxs = np.array([i for i in np.arange(len(all_y)) if all_y[i] == 0])
frac_train = 0.6
frac_val = 0.2
frac_test = 1 - frac_val - frac_train

frac_val_end = frac_train + frac_val

SEED = int(sys.argv[1])
rng = np.random.RandomState(seed=SEED)

rng.shuffle(positive_idxs)
rng.shuffle(negative_idxs)

train_positive_idxs = positive_idxs[:int(frac_train*len(positive_idxs))]
val_positive_idxs = positive_idxs[int(frac_train*len(positive_idxs)): int(frac_val_end*len(positive_idxs))]
test_positive_idxs = positive_idxs[int(frac_val_end*len(positive_idxs)):]

train_negative_idxs = negative_idxs[:int(frac_train*len(negative_idxs))]
val_negative_idxs = negative_idxs[int(frac_train*len(negative_idxs)): int(frac_val_end*len(negative_idxs))]
test_negative_idxs = negative_idxs[int(frac_val_end*len(negative_idxs)):]

train_idxs = np.concatenate([train_positive_idxs, train_negative_idxs])
val_idxs = np.concatenate([val_positive_idxs, val_negative_idxs])
test_idxs = np.concatenate([test_positive_idxs, test_negative_idxs])

train_ecgs, train_bphr, train_other, train_y = all_ecgs[train_idxs], all_bphr[train_idxs], all_others[train_idxs],all_y[train_idxs]
train_tab = np.concatenate([train_bphr, train_other], axis=1)

val_ecgs, val_bphr, val_other, val_y = all_ecgs[val_idxs], all_bphr[val_idxs], all_others[val_idxs],all_y[val_idxs]
val_tab = np.concatenate([val_bphr, val_other], axis=1)

test_ecgs, test_bphr, test_other, test_y = all_ecgs[test_idxs], all_bphr[test_idxs], all_others[test_idxs],all_y[test_idxs]
test_tab = np.concatenate([test_bphr, test_other], axis=1)

# some normalisation here of age
age_mean_train = train_tab[:, 4].mean()
age_std_train = train_tab[:, 4].std()
train_tab[:,4] = (train_tab[:,4] - age_mean_train)/age_std_train
val_tab[:,4] = (val_tab[:,4] - age_mean_train)/age_std_train
test_tab[:,4] = (test_tab[:,4] - age_mean_train)/age_std_train

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df_reduced_final_train = df_reduced_final.iloc[train_idxs]
df_reduced_final_val = df_reduced_final.iloc[val_idxs]
df_reduced_final_test = df_reduced_final.iloc[test_idxs]

# load params 
model_params = np.load('forward-model-learned-seed%d/fm-params.npy' % SEED, allow_pickle=True)
all_learn_params = [i.to(device) for i in model_params]

(muRln0_dev, muRln1_dev, 
muCln0_dev, muCln1_dev, 
muTsln0_dev, muTsln1_dev, 
muTdln0_dev, muTdln1_dev, 
muCOln0_dev, muCOln1_dev,
sigRln0_dev,sigRln1_dev,
sigCln0_dev,sigCln1_dev,
sigTsln0_dev,sigTsln1_dev,
sigTdln0_dev,sigTdln1_dev,
sigCOln0_dev,sigCOln1_dev) = all_learn_params

for i in all_learn_params:
    i.requires_grad = False

def get_log_p_ypi(pi, y):
#     print(logit_pi.max(), logit_pi.min(), logit_pi.mean())
    bce_loss = nn.BCELoss(reduction='sum')
    retval = bce_loss(torch.squeeze(pi), y)
    return -1*retval

def get_log_ppi(pi):
    conc1 = torch.tensor([1.0]).to(device)
    conc2 = torch.tensor([1.0]).to(device)
    m = Beta(conc1, conc2)
    return torch.sum(m.log_prob(torch.clamp(pi, 0.05, 0.95)))

def get_loss(enc, dec_ecg, dec_demo, x_batch_ecg, x_batch_other, y_batch, full=True):
    z, pi = enc.forward(x_batch_ecg, x_batch_other)
    
    log_p_zpi, zpi_ld = get_log_p_zpi(z, pi, all_learn_params)
    log_p_xz, xz_ld =  get_log_p_xz(x_batch_ecg, x_batch_other, z, dec_ecg, dec_demo)
    log_p_ypi = get_log_p_ypi(pi, y_batch)
    log_ppi = get_log_ppi(pi)

    lossdict = {
        'ypi_logprob' : log_p_ypi.item(),
        'ppi_logprob' : log_ppi.item()
    }

    lossdict.update(zpi_ld)
    lossdict.update(xz_ld)

    if full:
        loss = -1*(log_ppi + log_p_zpi + log_p_xz + log_p_ypi)
    else:
        loss = -1*(log_ppi + log_p_zpi + log_p_ypi)
    
    lossdict['loss'] = loss.item()

    lossdict = {k: v/len(x_batch_ecg) for (k,v) in lossdict.items()}

    if torch.isnan(loss):
        print("got a nan")   

    return loss, lossdict

def get_preds(dl, enc, dec_ecg, dec_demo):
    z_preds = []
    pi_preds = []
    y_trues = []
    enc.eval()
    dec_ecg.eval()
    dec_demo.eval()
    for i, (xecg, xother, y) in enumerate(dl):
        y_trues.append(y.detach().numpy())
        xecg = xecg.to(device)
        xother = xother.to(device)

        z, pi = enc.forward(xecg, xother)
    
        z_preds.append(z.detach().cpu().numpy())

        pi_preds.append(pi.cpu().detach().numpy())
    
    return (z_preds, pi_preds, y_trues)

def evaluate(dl, enc, dec_ecg, dec_demo):
    enc.eval()
    dec_ecg.eval()
    dec_demo.eval()
    ld = {}
    total = 0

    for i, (xecg, xother, y) in enumerate(dl):
        xecg = xecg.to(device)
        xother = xother.to(device)
        y = y.to(device)
        _, lossdict = get_loss(enc, dec_ecg, dec_demo, xecg, xother, y)
        ld = update_lossdict(ld, lossdict, action='sum')
        total += len(y)
    ld = {i: v/total for i,v in ld.items()}
    (z_preds, pi_preds, y_trues) = get_preds(dl, enc, dec_ecg, dec_demo)
    pi_preds = np.squeeze(np.concatenate(pi_preds, axis=0))
    y_trues = np.squeeze(np.concatenate(y_trues, axis=0))
    try:
        ld['auc'] = roc_auc_score(y_trues, pi_preds)
        ld['prauc'] = average_precision_score(y_trues, pi_preds)
    except ValueError:
        ld['auc'] =  0
        ld['prauc'] = 0
    return ld

def train_model(train_dataloader, val_dataloader, test_dataloader, enc=None, dec_ecg=None, dec_demo=None):
    if enc is None:
        enc = resnet_inf('resnet18', BasicBlock, [2, 2, 2, 2], False, False, num_outputs=6).to(device)
    if dec_ecg is None:
        dec_ecg = DecoderECG(5).to(device)
    if dec_demo is None:
        dec_demo = DecoderDemo(5).to(device)
    dec_ecg.load_state_dict(torch.load('forward-model-learned-seed%d/dec-ecg.ckpt' % SEED))
    dec_demo.load_state_dict(torch.load('forward-model-learned-seed%d/dec-demo.ckpt' % SEED))
    dec_ecg.eval()
    dec_demo.eval()
    optim_list = [
                    {'params': enc.parameters(), 'lr' :1e-3},
                ]
    optimizer = torch.optim.Adam(optim_list)

    train_ld = {}
    val_ld = {}
    test_ld = {}
    alltestpreds = []
    
     # Train the model
    num_epochs = 200
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        enc.train()
        for i, (xecg, xother, y) in enumerate(train_dataloader):
            # do some reshaping
            xecg = xecg.to(device)
            xother = xother.to(device)
            y = y.to(device)
            
            if epoch < 5:
                loss, lossdict = get_loss(enc, dec_ecg, dec_demo, xecg, xother, y, full=False)
            else:
                loss, lossdict = get_loss(enc, dec_ecg, dec_demo, xecg, xother, y, full=True)
            optimizer.zero_grad()       
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1)
            optimizer.step()
            
            if (i) % 8 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, log p(y|pi): {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, lossdict['loss'], lossdict['ypi_logprob']))
            train_ld = update_lossdict(train_ld, lossdict)

        lossdict = evaluate(val_dataloader, enc, dec_ecg, dec_demo)
        val_ld = update_lossdict(val_ld, lossdict)
        lossdict = evaluate(test_dataloader, enc, dec_ecg, dec_demo)
        test_ld = update_lossdict(test_ld, lossdict)

        # grab preds
        tpred = get_preds(test_dataloader, enc, dec_ecg, dec_demo)
        alltestpreds.append(tpred)

    return enc, train_ld, val_ld, test_ld, alltestpreds

# Train
scale_x = np.array([[100, 100, 100, 1, 1, 1, 1, 1, 1, 1]])
train_dataloader = norm_and_dataloader(train_ecgs, train_tab/scale_x, train_y,bs=64)
val_dataloader = norm_and_dataloader(val_ecgs, val_tab/scale_x, val_y,bs=64, shuffle=False)
test_dataloader = norm_and_dataloader(test_ecgs, test_tab/scale_x, test_y,bs=64, shuffle=False)

enc, trainld, valld, testld, testpreds = train_model(train_dataloader, val_dataloader, test_dataloader)

for k,v in trainld.items():
    trainld[k] = [float(i) for i in v]

for k,v in valld.items():
    valld[k] = [float(i) for i in v]

for k,v in testld.items():
    testld[k] = [float(i) for i in v]

final_res = {
        'trainld' : trainld,
        'valld' : valld,
        'testld' : testld,
}

savefol = "results_inf_seed%d-2/" % SEED
os.makedirs(savefol, exist_ok=True)

json_fname = os.path.join(savefol, 'res.json')

with open(json_fname, 'w') as f:
    json.dump(final_res, f)

preds_savefol = "preds_inf_seed%d-2/" % SEED
os.makedirs(preds_savefol, exist_ok=True)

preds_fname = os.path.join(preds_savefol, 'preds.pkl')

# Save preds
pickle.dump(testpreds, open(preds_fname, 'wb'))