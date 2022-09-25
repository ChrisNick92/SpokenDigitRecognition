import glob
from turtle import forward
from typing import OrderedDict
from unicodedata import decimal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# The data should be placed on the working directory path

# Implementation of data parser


def data_parser():
    """
    A function that reads the wav files in digits file
    and returns three lists, 
    1: Containing the wav file names
    2: The number of the speaker
    3: The digit
    """
    digits_list = list(glob.glob("digits/*.wav"))
    wavs, speakers, digits = [], [], []
    for wav_file in digits_list:
        wavs.append(librosa.load(wav_file)[0])
        splitted = wav_file.split(".")[0]
        p = -2 if ("0"<=splitted[-2]<="9") else -1
        speakers.append(splitted[p:])
        digits.append(splitted[:p].split("/")[1])

    return wavs, speakers, digits

def vis_wave_form(signals, speakers, digits,
    w = 14, h = 10):
    """
    Visualize the waveforms of four samples
    """
    if len(speakers) != 4:
        raise ValueError(f"The number of samples should be equal to {4}...not {len(speakers)}")
    else:
        sns.set_style("dark")
        fig, ax = plt.subplots(figsize = (w,h), nrows =2, ncols = 2)
        for i,signal in enumerate(signals):
            r,c = divmod(i,2)
            librosa.display.waveshow(signal, ax = ax[r,c])
            ax[r,c].set_title(f"The Waveform of Speaker {speakers[i]} for digit {digits[i]}",
                fontsize = 10)
        plt.show()

def extract_mfcss(wavs, n_mfcc = 13, sr = 22050):
    win_length = int(0.025*sr)
    step = int(0.01*sr)

    mfcss = [librosa.feature.mfcc(
        y = wav, sr = sr, n_fft = win_length, hop_length = win_length - step,
        n_mfcc=n_mfcc
    ).T for wav in tqdm(wavs, desc="Extracting mfccs...")]

    delta = [librosa.feature.delta(
        data.T,order=1
    ).T for data in tqdm(mfcss, desc= "Extracting deltas...")]

    deltadeltas = [librosa.feature.delta(
        data.T, order=2
    ).T for data in tqdm(mfcss, desc = "Extracting delta-deltas...")]

    return mfcss, delta, deltadeltas


############## Parse free digits dataset ##############

def concatenate(frames, dframes, ddframes):
    return [
        np.concatenate((frame, dframe, ddframe), axis=1)
        for frame, dframe, ddframe in tqdm(zip(frames, dframes, ddframes), desc='Concatenating...')
    ]

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob.glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("/")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=13, Fs=22050, with_deltas=True):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    # window = int(0.025*Fs)
    # step = int(0.01*Fs)
    frames = [
        np.float64(librosa.feature.mfcc(
            y= wav, sr = Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T)

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    if with_deltas:
        #calculates also delta and delta-delta mfcc...
        d_frames = [np.float64(librosa.feature.delta(frame.T, order=1).T)
                    for frame in tqdm(frames, desc='calculating deltas...')]
        dd_frames = [np.float64(librosa.feature.delta(frame.T, order=2).T)
                     for frame in tqdm(frames, desc='calculating delta-deltas...')]
        return concatenate(frames, d_frames, dd_frames)

    return frames


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4", "5", "6", "8", "9", "10"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale


def parser(directory, n_mfcc=13, with_deltas = True):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs, with_deltas=with_deltas)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

#########################################################################


import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_pad_zeros(batch):
    xs = [x[0] for x in batch]
    ys = [x[1] for x in batch]
    x_padded = pad_sequence(xs, batch_first=True)

    return x_padded, ys

class custom_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx], dtype = torch.float32), torch.tensor(self.y[idx], dtype =torch.int64)
    

class lstm(nn.Module):
    def __init__(self, input_size = 39, hidden_size = 128,
                num_layers = 3, out_classes = 10):
        super(lstm, self).__init__()
    
        self.main_body = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)

        self.classifier = nn.Sequential(nn.Linear(hidden_size, 64),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                        nn.Linear(64, 10))
    def forward(self,x):
        x = self.main_body(x)
        return self.classifier(x)
        

