# vEM
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df_reduced_final_train = df_reduced_final.iloc[train_idxs]
df_reduced_final_val = df_reduced_final.iloc[val_idxs]
df_reduced_final_test = df_reduced_final.iloc[test_idxs]


idxs = pd.notnull(df_reduced_final_train['svr_proc'])
svr = df_reduced_final_train['svr_proc'][idxs]
outcomes = train_y[idxs]

# Convert the units of the SVR into Woods units (from dyn)
muRln0 = np.mean(np.log(svr[outcomes==0]*3/4000))
sigRln0 = np.std(np.log(svr[outcomes==0]*3/4000))
muRln1 = np.mean(np.log(svr[outcomes==1]*3/4000))
sigRln1 = np.std(np.log(svr[outcomes==1]*3/4000))

idxs = pd.notnull(df_reduced_final_train['svr_proc'])
svr = df_reduced_final_train['svr_proc'][idxs]*3/4000
outcomes = train_y[idxs]

# Diastole time: assume to be 2/3 of a cardiac cycle, and convert bpm to seconds.
td =( 40 / df_reduced_final_train['hr_proc'].values).astype(np.float32)

# Calculate the BPSYS/BPDIAS ratio from the diastole section of the 2 element Windkessel model used to calculate the time constant.
tmp = ((df_reduced_final_train['bpsys'].values + 1e-5)/(df_reduced_final_train['bpdia'].values + 1e-5)).astype(np.float32)

# Calculate RC using the above, and the Windkessel relationship. 
rc = np.clip(1 / td * np.log(tmp), 1e-5, 5)[idxs]
svc = np.clip(rc/svr, 0, 10)
muCln0 = np.mean(np.log(svc[outcomes==0]))
sigCln0 = np.std(np.log(svc[outcomes==0]))
muCln1 = np.mean(np.log(svc[outcomes==1]))
sigCln1 = np.std(np.log(svc[outcomes==1]))


idxs = pd.notnull(df_reduced_final_train['thermo_co_proc'])
c_op = df_reduced_final_train['thermo_co_proc'][idxs]
outcomes = train_y[idxs]
muCOln0 = np.mean(np.log(c_op[outcomes==0]))
sigCOln0 = np.std(np.log(c_op[outcomes==0]))
muCOln1 = np.mean(np.log(c_op[outcomes==1]))
sigCOln1 = np.std(np.log(c_op[outcomes==1]))

ts =( 20 / df_reduced_final_train['hr_proc'].values).astype(np.float32)
muTsln0 = np.mean(np.log(ts[train_y==0]))
sigTsln0 = np.std(np.log(ts[train_y==0]))
muTsln1 = np.mean(np.log(ts[train_y==1]))
sigTsln1 = np.std(np.log(ts[train_y==1]))

td =( 40 / df_reduced_final_train['hr_proc'].values).astype(np.float32)
muTdln0 = np.mean(np.log(td[train_y==0]))
sigTdln0 = np.std(np.log(td[train_y==0]))
muTdln1 = np.mean(np.log(td[train_y==1]))
sigTdln1 = np.std(np.log(td[train_y==1]))

muRln0_dev = torch.autograd.Variable(torch.tensor(muRln0)).to(device)
muRln1_dev = torch.autograd.Variable(torch.tensor(muRln1)).to(device)

muCln0_dev = torch.autograd.Variable(torch.tensor(muCln0)).to(device)
muCln1_dev = torch.autograd.Variable(torch.tensor(muCln1)).to(device)

muTsln0_dev = torch.autograd.Variable(torch.tensor(muTsln0)).to(device)
muTsln1_dev = torch.autograd.Variable(torch.tensor(muTsln1)).to(device)

muTdln0_dev = torch.autograd.Variable(torch.tensor(muTdln0)).to(device)
muTdln1_dev = torch.autograd.Variable(torch.tensor(muTdln1)).to(device)

muCOln0_dev = torch.autograd.Variable(torch.tensor(muCOln0)).to(device)
muCOln1_dev = torch.autograd.Variable(torch.tensor(muCOln1)).to(device)


sigRln0_dev = torch.autograd.Variable(torch.tensor(sigRln0)).to(device)
sigRln1_dev = torch.autograd.Variable(torch.tensor(sigRln1)).to(device)

sigCln0_dev = torch.autograd.Variable(torch.tensor(sigCln0)).to(device)
sigCln1_dev = torch.autograd.Variable(torch.tensor(sigCln1)).to(device)

sigTsln0_dev = torch.autograd.Variable(torch.tensor(sigTsln0)).to(device)
sigTsln1_dev = torch.autograd.Variable(torch.tensor(sigTsln1)).to(device)

sigTdln0_dev = torch.autograd.Variable(torch.tensor(sigTdln0)).to(device)
sigTdln1_dev = torch.autograd.Variable(torch.tensor(sigTdln1)).to(device)

sigCOln0_dev = torch.autograd.Variable(torch.tensor(sigCOln0)).to(device)
sigCOln1_dev = torch.autograd.Variable(torch.tensor(sigCOln1)).to(device)

all_learn_means = [muRln0_dev, muRln1_dev, 
                   muCln0_dev, muCln1_dev, 
                   muTsln0_dev, muTsln1_dev, 
                   muTdln0_dev, muTdln1_dev, 
                   muCOln0_dev, muCOln1_dev]

all_phi_mean_muprior = [muRln0, muRln1, 
                   muCln0, muCln1, 
                   muTsln0, muTsln1, 
                   muTdln0, muTdln1, 
                   muCOln0, muCOln1]

all_phi_mean_stdprior = [0.01, 0.01, 
                   0.01, 0.01, 
                   0.01, 0.01, 
                   0.01, 0.01, 
                   0.01, 0.01]

all_learn_stds = [sigRln0_dev,sigRln1_dev,
                  sigCln0_dev,sigCln1_dev,
                  sigTsln0_dev,sigTsln1_dev,
                  sigTdln0_dev,sigTdln1_dev,
                  sigCOln0_dev,sigCOln1_dev]

all_learn_params = all_learn_means
all_dist_params = all_learn_means + all_learn_stds

for i in all_learn_means:
    i.requires_grad = True

# Do not learn the std devs of the forward model (more stable).
for i in all_learn_stds:
    i.requires_grad = False

def log_p_phi_helper(val, mean, std):
    normobj = Normal(torch.tensor(mean).to(device), torch.tensor(std).to(device))
    return normobj.log_prob(val)

def get_log_p_phi():
    retval = 0.
    # means
    for param, meanval, stdval in zip(all_learn_means, all_phi_mean_muprior, all_phi_mean_stdprior):
        retval += log_p_phi_helper(param, meanval, stdval)
    
    return retval

def get_log_p_ypi(pi, y):
    bce_loss = nn.BCELoss(reduction='sum')
    retval = bce_loss(torch.squeeze(pi), y)
    return retval

def get_log_ppi(pi):
    conc1 = torch.tensor([1.0]).to(device)
    conc2 = torch.tensor([1.0]).to(device)
    m = Beta(conc1, conc2)
    return torch.sum(m.log_prob(torch.clamp(pi, 0.05, 0.95)))


def get_loss(enc, dec_ecg, dec_demo, x_batch_ecg, x_batch_other, y_batch, full=True):
    zmu, zstd, pi_alpha, pi_beta = enc.forward(x_batch_ecg, x_batch_other)

    z_R_obj = LogNormal(zmu[:,0],zstd[:,0])
    z_C_obj = LogNormal(zmu[:,1],zstd[:,1])
    z_Ts_obj = LogNormal(zmu[:,2],zstd[:,2])
    z_Td_obj = LogNormal(zmu[:,3],zstd[:,3])
    z_CO_obj = LogNormal(zmu[:,4],zstd[:,4])
    
    pi_obj = Beta(pi_alpha, pi_beta)   

    z_R_samp = z_R_obj.rsample()
    z_C_samp = z_C_obj.rsample()
    z_Ts_samp = z_Ts_obj.rsample()
    z_Td_samp = z_Td_obj.rsample()
    z_CO_samp = z_CO_obj.rsample()
    pi_samp = pi_obj.rsample()

    z_samp = torch.cat([z_R_samp.view(-1, 1), 
                        z_C_samp.view(-1, 1),
                        z_Ts_samp.view(-1, 1),
                        z_Td_samp.view(-1, 1),
                        z_CO_samp.view(-1,1)],
                       dim=1)
    
    log_p_phi = get_log_p_phi()
    log_p_zpi, zpi_ld = get_log_p_zpi(z_samp, pi_samp, all_dist_params)
    log_p_xz, xz_ld =  get_log_p_xz(x_batch_ecg, x_batch_other, z_samp, dec_ecg, dec_demo)
    log_p_ypi = get_log_p_ypi(pi_samp, y_batch)
    log_qzpi = get_log_qzpi(zmu, zstd, z_samp, pi_alpha, pi_beta, pi_samp)
    log_ppi = get_log_ppi(pi_samp)

    lossdict = {
        'phi_logprob' : log_p_phi.item(),
        'ypi_logprob' : log_p_ypi.item(),
        'qzpi_logprob' : log_qzpi.item(),
        'ppi_logprob' : log_ppi.item()
    }

    lossdict.update(zpi_ld)
    lossdict.update(xz_ld)

    if full:
        loss = -1*(log_ppi + log_p_phi + log_p_zpi + log_p_xz + log_p_ypi - log_qzpi)
    else:
        loss = -1*(log_ppi+ log_p_phi + log_p_zpi + log_p_ypi - log_qzpi)
    
    lossdict['loss'] = loss.item()

    lossdict = {k: v/len(x_batch_ecg) for (k,v) in lossdict.items()}

    if torch.isnan(loss):
        print("got a nan")

    return loss, lossdict


def get_preds(dl, enc, dec_ecg, dec_demo):
    z_preds_mu = []
    z_preds_std = []
    pi_preds_alpha = []
    pi_preds_beta = []
    y_trues = []
    x_trues_ecg = []
    x_trues_other = []
    x_hats_ecg = []
    x_hats_other = []
    enc.eval()
    dec_ecg.eval()
    dec_demo.eval()
    for i, (xecg, xother, y) in enumerate(dl):
        y_trues.append(y.detach().numpy())
        xecg = xecg.to(device)
        xother = xother.to(device)

        zmu, zstd, pi_alpha, pi_beta = enc.forward(xecg, xother)
    
        z_preds_mu.append(zmu.detach().cpu().numpy())
        z_preds_std.append(zstd.detach().cpu().numpy())
        z_mode = torch.exp(zmu - zstd**2)
    
        x_recon_bphr = pa_p_hr_model(z_mode)
        x_recon_other = torch.cat(x_recon_bphr, dim=1)
        x_recon_ecg = dec_ecg.forward(z_mode)
        x_recon_demo = dec_demo.forward(z_mode)
        x_recon_other = torch.cat([x_recon_other, x_recon_demo], dim=1)

        pi_preds_alpha.append(pi_alpha.cpu().detach().numpy())
        pi_preds_beta.append(pi_beta.cpu().detach().numpy())
    
        x_hats_ecg.append(x_recon_ecg.detach().cpu().numpy())
        x_hats_other.append(x_recon_other.detach().cpu().numpy())

        x_trues_ecg.append(xecg.detach().cpu().numpy())
        x_trues_other.append(xother.detach().cpu().numpy())

    return (z_preds_mu, z_preds_std, pi_preds_alpha, pi_preds_beta, x_hats_ecg, x_hats_other, y_trues, x_trues_ecg, x_trues_other)


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
    (z_preds_mu, z_preds_std, pi_preds_alpha, pi_preds_beta, 
     x_hats_ecg, x_hats_bphr, y_trues, x_trues_ecg, x_trues_bphr) = get_preds(dl, enc, dec_ecg, dec_demo)
    pi_preds_alpha = np.squeeze(np.concatenate(pi_preds_alpha, axis=0))
    pi_preds_beta = np.squeeze(np.concatenate(pi_preds_beta, axis=0))
#     pi_preds = (pi_preds_alpha / (pi_preds_alpha + pi_preds_beta))

    # Take the mode of the beta as the predicted pi, rather than the mean (commented out above).
    pi_preds = ((pi_preds_alpha - 1) / (pi_preds_alpha + pi_preds_beta - 2))
    y_trues = np.squeeze(np.concatenate(y_trues, axis=0))
    x_trues_bphr = np.squeeze(np.concatenate(x_trues_bphr, axis=0))
    x_hats_bphr = np.squeeze(np.concatenate(x_hats_bphr, axis=0))
    try:
        ld['r2_esp'] = r2_score(x_trues_bphr[:,0], x_hats_bphr[:,0])
        ld['r2_edp'] = r2_score(x_trues_bphr[:,1], x_hats_bphr[:,1])
        ld['r2_hr'] = r2_score(x_trues_bphr[:,2], x_hats_bphr[:,2])
        ld['auc'] = roc_auc_score(y_trues, pi_preds)
        ld['prauc'] = average_precision_score(y_trues, pi_preds)
    except ValueError:
        ld['r2_mp'] = -1e5
        ld['r2_esp'] = -1e5
        ld['r2_edp'] = -1e5
        ld['r2_hr'] = -1e5
        ld['auc'] =  0
        ld['prauc'] = 0
    return ld


def train_model(train_dataloader, val_dataloader, test_dataloader, enc=None, dec_ecg=None, dec_demo=None):
    if enc is None:
        enc = resnet_vi('resnet18', BasicBlock, [2, 2, 2, 2], False, False, num_outputs=6).to(device)
    if dec_ecg is None:
        dec_ecg = DecoderECG(5).to(device)
    if dec_demo is None:
        dec_demo = DecoderDemo(5).to(device)
    optim_list = [
                    {'params': enc.parameters(), 'lr' :1e-4},
                    {'params': dec_ecg.parameters(), 'lr' :1e-4}, 
                    {'params': dec_demo.parameters(), 'lr' :1e-4}, 
                ]
    other_params = [ {'params' : i, 'lr' : 1e-4} for i in all_learn_params]
    optimizer = torch.optim.Adam(optim_list + other_params)

    train_ld = {}
    val_ld = {}
    test_ld = {}
    alltestpreds = []
    
     # Train the model
    num_epochs = 200
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        enc.train()
        dec_ecg.train()
        dec_demo.train()
        for i, (xecg, xother, y) in enumerate(train_dataloader):
            # do some reshaping
            xecg = xecg.to(device)
            xother = xother.to(device)
            y = y.to(device)
            
            if epoch < 10:
                loss, lossdict = get_loss(enc, dec_ecg, dec_demo, xecg, xother, y, full=False)
            else:
                loss, lossdict = get_loss(enc, dec_ecg, dec_demo, xecg, xother, y, full=True)
            optimizer.zero_grad()       
            loss.backward()
            if epoch < 10:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1)
            else:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(dec_ecg.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(dec_demo.parameters(), 1)
            optimizer.step()
            
            if (i) % 8 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, log p(y|pi): {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, lossdict['loss'], lossdict['ypi_logprob']))
#                 print(lossdict)
            train_ld = update_lossdict(train_ld, lossdict)

        lossdict = evaluate(val_dataloader, enc, dec_ecg, dec_demo)
        val_ld = update_lossdict(val_ld, lossdict)

        lossdict = evaluate(test_dataloader, enc, dec_ecg, dec_demo)
        test_ld = update_lossdict(test_ld, lossdict)

        # grab preds
        (z_preds_mu, z_preds_std, pi_preds_alpha, pi_preds_beta, 
         x_hats_ecg, x_hats_bphr, y_trues, x_trues_ecg, x_trues_bphr) = get_preds(test_dataloader, enc, dec_ecg, dec_demo)
        tpred = [z_preds_mu, z_preds_std, pi_preds_alpha, pi_preds_beta, y_trues]
        alltestpreds.append(tpred)

    return enc, dec_ecg, dec_demo, train_ld, val_ld, test_ld, alltestpreds
    


# Train
scale_x = np.array([[100, 100, 100, 1, 1, 1, 1, 1, 1, 1]])
train_dataloader = norm_and_dataloader(train_ecgs, train_tab/scale_x, train_y,bs=64)
val_dataloader = norm_and_dataloader(val_ecgs, val_tab/scale_x, val_y,bs=64, shuffle=False)
test_dataloader = norm_and_dataloader(test_ecgs, test_tab/scale_x, test_y,bs=64, shuffle=False)

enc, _, _, trainld, valld, testld, testpreds = train_model(train_dataloader, val_dataloader, test_dataloader)

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

savefol = "results_vem_seed%d/" % SEED
os.makedirs(savefol, exist_ok=True)

json_fname = os.path.join(savefol, 'res.json')

with open(json_fname, 'w') as f:
    json.dump(final_res, f)

preds_savefol = "preds_vem_seed%d/" % SEED
os.makedirs(preds_savefol, exist_ok=True)

preds_fname = os.path.join(preds_savefol, 'preds.pkl')

# Save preds
pickle.dump(testpreds, open(preds_fname, 'wb'))