# -*- coding: utf-8 -*-
# @Author  : Jing
# @FileName: preprocess.py
# @IDE: PyCharm
"""滤波+基线校准+56通道+下采样8"""
import warnings
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.signal import resample, savgol_filter

from sklearn.preprocessing import scale

from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, find_events, pick_types, Epochs

warnings.filterwarnings('ignore')


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_csv(x, y, features, split=False):
    n_samples, n_features = x.shape
    y = np.reshape(y, (n_samples, 1))
    xy = np.concatenate((x, y), axis=1)
    xy = pd.DataFrame(xy, index=range(n_samples), columns=features+['label'])
    if split:
        xy0 = xy[xy['label'] == 0]
        xy1 = xy[xy['label'] == 1]
        return xy0, xy1
    else:
        return xy


def create_mne_raw_object(fname):
    """Create a mne epoch instance"""
    l_freq, h_freq = 0.5, 20
    method = 'iir'
    df = pd.read_csv(fname)
    df = df.drop(columns=['Time', 'EOG'])
    ch_names = list(df.columns)
    ch_types = ['eeg']*56+['stim']
    sfreq = 200
    montage = read_montage('standard_1020', ch_names)
    ch_dic = dict(zip(ch_names[0:56], list(range(56))))
    info = create_info(ch_names, sfreq, ch_types, montage)
    data = df.values
    raw = RawArray(data.T, info, verbose=False)
    picks = pick_types(raw.info, eeg=True)
    raw.filter(l_freq, h_freq, picks=picks, method=method, verbose=False)
    del data
    data = raw.get_data()
    del raw
    data[0:56] = savgol_filter(data[0:56], 31, 3)
    raw = RawArray(data, info, verbose=False)
    events = find_events(raw, ['FeedBackEvent'], verbose=False)
    raw = raw.drop_channels(['FeedBackEvent'])
    return raw, events, ch_dic


def process(save_path, subjects, label, data_path, verbose=False):
    ch = list(range(56))
    sample_tmin = 200
    sample_tmax = 600
    resample_n = int((sample_tmax-sample_tmin)*0.02)
    for index in tqdm(range(len(subjects))):
        data = pd.DataFrame(columns=['var_'+str(i) for i in range(resample_n*len(ch))]+['bias'],
                            index=range(60))
        sub_label = label[index*340:index*340+340]
        features = list(data.columns)
        new_save_path = save_path
        make_dir(new_save_path)
        for i in range(1, 6):
            fname = data_path+'Data_S'+subjects[index]+'_Sess0'+str(i)+'.csv'
            raw, events, ch_dic = create_mne_raw_object(fname)
            picks = pick_types(raw.info, meg=False, eeg=True)
            epochs = Epochs(raw, events, tmin=-0.2, tmax=1.3, baseline=(None, 0), picks=picks,
                            preload=True, verbose=verbose)
            tmp = epochs.get_data()[:, ch, int(sample_tmin//5)+40:int(sample_tmax//5)+40]
            if resample_n is not None:
                tmp = resample(tmp, resample_n, axis=2)
            tmp = scale(np.reshape(tmp, (tmp.shape[0], tmp.shape[1]*tmp.shape[2])), axis=0)
            bias = np.ones([tmp.shape[0], 1])
            tmp = np.concatenate((tmp, bias), axis=1)
            if i == 1:
                tmp = pd.DataFrame(tmp, columns=features, index=range(60))
                data = tmp
            elif i == 5:
                tmp = pd.DataFrame(tmp, columns=features, index=range(100))
                data = pd.concat([data, tmp], ignore_index=True)
            else:
                tmp = pd.DataFrame(tmp, columns=features, index=range(60))
                data = pd.concat([data, tmp], ignore_index=True)
        data = data.values
        xy = make_csv(data, sub_label, features)
        save_file = new_save_path+'S'+subjects[index]+'.csv'
        xy.to_csv(save_file, index=False)


def main():
    ex_type = 'test'
    train_subjects = ['02', '06', '07', '11', '12', '13', '14', '16',
                      '17', '18', '20', '21', '22', '23', '24', '26']
    test_subjects = ['01', '03', '04', '05', '08', '09', '10', '15', '19', '25']
    if ex_type == 'train':
        label_file = '../input/TrainLabels.csv'
        subjects = train_subjects
        label = pd.read_csv(label_file)['Prediction'].values
    else:
        label_file = '../input/true_labels.csv'
        subjects = test_subjects
        label = pd.read_csv(label_file, header=None).values
    raw_data_path = '../input/'
    sub_raw_data_path = raw_data_path + ex_type + '/'
    output_path = '../output/'
    make_dir(output_path)
    processed_data_path = output_path+'data/'
    make_dir(processed_data_path)
    process(processed_data_path, subjects, label, sub_raw_data_path)


if __name__ == '__main__':
    main()

