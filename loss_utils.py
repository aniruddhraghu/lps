# loss utils
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

def log_p_zpi_lognorm_helper(z, pi, mu0, std0, mu1, std1):
    denom_val = np.sqrt(2*np.pi)
    maxelem = torch.mean(torch.stack((-(torch.log(z)-mu1)**2 / (2*std1**2), 
                                      -(torch.log(z)-mu0)**2 / (2*std0**2)),
                                     dim=1),dim=1)
    
    p1 = 1/(denom_val * std1*z) * torch.exp(-(torch.log(z)-mu1)**2 / (2*std1**2) - maxelem)
    p0 = 1/(denom_val * std0*z) * torch.exp(-(torch.log(z)-mu0)**2 / (2*std0**2) - maxelem)
    
    logval = pi*p1 + (1-pi)*p0
    
    retval = torch.sum(torch.log(logval+1e-10) + maxelem)
    return retval

def get_log_p_zpi(z, pi, dist_params):
    (muRln0_dev, muRln1_dev, 
    muCln0_dev, muCln1_dev, 
    muTsln0_dev, muTsln1_dev, 
    muTdln0_dev, muTdln1_dev, 
    muCOln0_dev, muCOln1_dev,
    sigRln0_dev,sigRln1_dev,
    sigCln0_dev,sigCln1_dev,
    sigTsln0_dev,sigTsln1_dev,
    sigTdln0_dev,sigTdln1_dev,
    sigCOln0_dev,sigCOln1_dev) = dist_params
    r_prob = log_p_zpi_lognorm_helper(z[:, 0], pi, muRln0_dev, sigRln0_dev, muRln1_dev, sigRln1_dev)
    c_prob = log_p_zpi_lognorm_helper(z[:, 1], pi, muCln0_dev, sigCln0_dev, muCln1_dev, sigCln1_dev)
    ts_prob = log_p_zpi_lognorm_helper(z[:, 2], pi, muTsln0_dev, sigTsln0_dev, muTsln1_dev, sigTsln1_dev)
    td_prob = log_p_zpi_lognorm_helper(z[:, 3], pi, muTdln0_dev, sigTdln0_dev, muTdln1_dev, sigTdln1_dev)
    co_prob = log_p_zpi_lognorm_helper(z[:, 4], pi, muCOln0_dev, sigCOln0_dev, muCOln1_dev, sigCOln1_dev)
    val = r_prob + c_prob + ts_prob + td_prob + co_prob
    retdict = {
        'zpi_logprob' : val.item(),
        'r_logprob': r_prob.item(),
        'c_logprob': c_prob.item(),
        'ts_logprob': ts_prob.item(),
        'td_logprob': td_prob.item(),
        'co_logprob': co_prob.item(),
    }
    return val, retdict

# compute the systole phase of the 2 element WK model :
# https://isn.ucsd.edu/courses/beng221/problems/2012/BENG221_Project%20-%20Catanho%20Sinha%20Vijayan.pdf
def systole_torch(R, C, Ts, t, P0=85, I0=424.1):
    tau = R*C
    c1 = P0 + (I0 * Ts * R * np.pi * tau)/(Ts**2 + (np.pi * tau)**2)
    p = c1*torch.exp(-t/tau) + (-torch.exp(t/tau) * Ts * I0 * R * (C*np.pi*R*torch.cos(np.pi*t/Ts) - Ts*torch.sin(np.pi*t/Ts)))/(Ts**2 +(tau*np.pi)**2)
    return p


# compute the diastole phase:
# https://isn.ucsd.edu/courses/beng221/problems/2012/BENG221_Project%20-%20Catanho%20Sinha%20Vijayan.pdf
def diastole_torch(R, C, Pd, t) :
    tau = R * C
    p = Pd * torch.exp(-t/tau)
    return p


# This steps the 2 element Windkessel model forward for 10 cycles; take the BP diastolic and systolic in cycles 5-10 and average to get estimates.
# This helps the system reach steady state.
def pa_p_model(z):
    R = z[:,0].view(-1,1)
    C = z[:,1].view(-1,1)
    Ts = z[:,2].view(-1,1)
    Td = z[:,3].view(-1,1)
    CO = z[:,4].view(-1,1)

    I0 = (np.pi * CO * 1e3 * (Ts + Td)) / (120*Ts)
    
    p_de = 80
    p_des = []
    p_ses = []
    for i in range(10):
        p_se = systole_torch(R, C, Ts, Ts, p_de, I0=I0)
        p_se = torch.clamp(p_se, 0, 200)
        p_de = diastole_torch(R, C, p_se, Td)
        p_de = torch.clamp(p_de, 0, 200)
        if i > 3:
            p_des.append(p_de)
            p_ses.append(p_se)
    p_de = torch.mean(torch.stack(p_des, dim=0), dim=0)
    p_se = torch.mean(torch.stack(p_ses, dim=0), dim=0)

    p_se = p_se.view(-1,1)
    p_de = p_de.view(-1,1)

    yout = torch.cat([p_se, p_de], axis=1)
    return yout/100

def pa_p_hr_model(z):
    Ts = z[:,2].view(-1,1)
    Td = z[:,3].view(-1,1)
  
    x_hat_p = pa_p_model(z)
    x_hat_hr = (0.6 / (Ts + Td)).view(-1, 1)

    return x_hat_p, x_hat_hr

def get_log_p_xz(x_ecg, x_other, z, dec_ecg, dec_demo):
    # Four terms: Demo, BP, HR, and ECG
    
    # BP and HR
    x_hat_p, x_hat_hr = pa_p_hr_model(z)
    
    # compute logprob
    p_logprob = torch.sum(-1 * (x_hat_p - x_other[:,:2])**2 / (2*.1**2))
    hr_logprob = torch.sum(-1 * (x_hat_hr.squeeze() - x_other[:,2])**2 / (2*.1**2))
    
    # ECG term
    x_hat_ecg = torch.squeeze(dec_ecg.forward(z))
    # as stated in the decoder, downsample x
    x_target = torch.transpose(x_ecg, 1,2)  [:,0, :500] # TODO check this.
    
    ecg_logprob = torch.sum(-1 * (x_hat_ecg - x_target.squeeze())**2 / (2*5**2))
    
    # Demo term
    x_hat_demo = dec_demo.forward(z)
    demo_logprob = torch.sum(-1 * (x_hat_demo.squeeze() - x_other[:,3:])**2 / (2*.5**2))

    val = p_logprob + hr_logprob + ecg_logprob + demo_logprob
    
    retdict = {
        'xz_logprob' : val.item(),
        'p_logprob': p_logprob.item(),
        'hr_logprob': hr_logprob.item(),
        'ecg_logprob': ecg_logprob.item(),
        'demo_logprob': demo_logprob.item(),
    }    

    return val, retdict

def get_log_qzpi(zmu, zstd, zsamp, pi_alpha, pi_beta, pi_samp):
    
    qz_R_obj = LogNormal(zmu[:,0],zstd[:,0])
    qz_C_obj = LogNormal(zmu[:,1],zstd[:,1])
    qz_Ts_obj = LogNormal(zmu[:,2],zstd[:,2])
    qz_Td_obj = LogNormal(zmu[:,3],zstd[:,3])
    qz_CO_obj = LogNormal(zmu[:,4],zstd[:,4])

    qz_pi_obj = Beta(pi_alpha, pi_beta) 
    
    return torch.sum(qz_R_obj.log_prob(zsamp[:,0])) + \
            torch.sum(qz_C_obj.log_prob(zsamp[:,1])) + \
            torch.sum(qz_Ts_obj.log_prob(zsamp[:,2])) + \
            torch.sum(qz_Td_obj.log_prob(zsamp[:,3])) + \
            torch.sum(qz_CO_obj.log_prob(zsamp[:,4])) + \
            torch.sum(qz_pi_obj.log_prob(torch.clamp(pi_samp, 0.1, 0.9)))

