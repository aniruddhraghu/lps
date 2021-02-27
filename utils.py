# utils
import numcodecs
import numpy as np
import os
import torch
import json

import torch.utils.data as t_utils

import pandas as pd
from datetime import datetime
from dateutil import relativedelta

import pickle
import h5py
import ast

def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict

def norm_and_dataloader(X_ecg, X_bphr, Y, bs=32, shuffle=True):
    tensor_X_ecg = torch.Tensor(X_ecg.astype(np.float32))
    tensor_X_bphr = torch.Tensor(X_bphr.astype(np.float32))
    tensor_Y = torch.Tensor(Y)
    dataset = t_utils.TensorDataset(tensor_X_ecg, tensor_X_bphr, tensor_Y)
    dataloader = t_utils.DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=0)
    return dataloader

def get_df():
    # This is just a template function for now.
    # The dataframe should contain BP systolic/diastolic, heart rate, age, gender, creatinine, and demographic identifiers.
    # It should also contain some information about the path to the ECG for each record.
    df = pd.read_csv('/path/to/df.csv')
    df_reduced_final = df.dropna(subset=['bpsys', 'bpdia', 'hr_proc'])

    # Calculate eGFR using a standard formula. 
    egfr = []
    rel_df = df_reduced_final[['age','g_proc', 'dem1', 'dem2', 'dem3', 'creat_clean']]
    for i in rel_df.index:
        row = rel_df.loc[i]
        alpha = -0.329 if row['g_proc'] == 0.5 else -0.411
        k = 0.7 if row['g_proc'] == 0.5 else 0.9
        dem = 1.159 if row['dem2'] == 1 else 1
        gen = 1.018 if row['g_proc'] == .5 else 1
        Cr = np.exp(row['creat_clean'])
        val = min(Cr/k, 1) **alpha * max(Cr/k,1)**(-1.209) * 0.993**row['age'] # * gen * dem
        egfr.append(val)
    df_reduced_final['egfr'] = np.array(egfr)
    return df_reduced_final



ECG_REST_AMP_LEADS = {
    'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
    'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11,
}

def _decompress_data(data_compressed, dtype):
    codec = numcodecs.zstd.Zstd()
    data_decompressed = codec.decode(data_compressed)
    if dtype == 'str':
        data = data_decompressed.decode()
    else:
        data = np.frombuffer(data_decompressed, dtype)
    return data

def _resample_voltage(voltage, desired_samples):
    if len(voltage) == desired_samples:
        return voltage
    elif len(voltage) == 2500 and desired_samples == 5000:
        x = np.arange(2500)
        x_interp = np.linspace(0, 2500, 5000)
        return np.interp(x_interp, x, voltage)
    elif len(voltage) == 5000 and desired_samples == 2500:
        return voltage[::2]
    else:
        raise ValueError('Voltage length {len(voltage)} is not desired {desired_samples} and re-sampling method is unknown.')

ecg_shape = (2500,12)

def make_voltage(fn, leads=ECG_REST_AMP_LEADS, population_normalize=2000.0):
    # This function assumes that ECGs are stored in HDF5 files and then can be read and processed per-lead.
    # It will need to be changed depending on the specific data format.
    tensor = np.zeros(ecg_shape, dtype=np.float32)
    with h5py.File(fn, "r") as hd5:
        for cm, idx in leads.items():
            voltage = _decompress_data(data_compressed=hd5[cm][()], dtype=hd5[cm].attrs['dtype'])
            voltage = _resample_voltage(voltage, ecg_shape[0])
            tensor[:, idx] = voltage
        tensor /= population_normalize
    return tensor

def get_ecg(df_reduced_final):
    ecg_base = '/path/to/ecg/'
    all_ecgs = []
    for row in df_reduced_final['ecg_path']:
        ecg_fname = row.replace(':', '_')
        fn = os.path.split(ecg_fname)[-1]
        ecg_fpath = os.path.join(ecg_base, fn)
        proc_ecg = make_voltage(ecg_fpath)
        all_ecgs.append(proc_ecg)
    all_ecgs = np.array(all_ecgs)
    return all_ecgs