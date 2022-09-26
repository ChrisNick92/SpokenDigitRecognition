from audioop import bias
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
from torch.optim import Adam
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
    
    return x_padded, torch.tensor(ys, dtype=torch.int64)

class custom_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx], dtype = torch.float32), self.y[idx]
    

class lstm(nn.Module):
    def __init__(self, input_size = 39, hidden_size = 128,
                num_layers = 1, out_classes = 10, bidirectional = False):
        super(lstm, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.main_body = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True,
                            bidirectional = bidirectional)
        if self.bidirectional:
            self.classifier = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, out_classes))
        else:
            self.classifier = nn.Sequential(nn.Linear(hidden_size, out_classes))
            
    def forward(self,x):
        _,(x,_) = self.main_body(x)
        if self.bidirectional:
            x1 = x[0]
            x2 = x[1]
            return self.classifier(torch.cat((x1, x2), dim=1))
        else:
            return self.classifier(torch.squeeze(x))
        
def training_loop(model, train_loader, val_loader, epochs,
                  lr, loss_fn, regularization=None,
                  reg_lambda=None, mod_epochs=20, early_stopping = False,
                  patience = None, verbose = None, title = None):
    optim = Adam(model.parameters(), lr=lr)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_loss_list = []
    val_loss_list = []
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    counter_epochs = 0

    if early_stopping:
        ear_stopping = EarlyStopping(patience= patience, verbose=verbose)

    for epoch in range(epochs):
        counter_epochs+=1
        model.train()
        train_loss, val_loss = 0.0, 0.0
        for X,y in train_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            train_loss += loss.item()

            # Regulirization
            if regularization == 'L2':
                l_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + reg_lambda * l_norm
            elif regularization == 'L1':
                l_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + reg_lambda * l_norm

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X, y = val_batch[0].to(device), val_batch[1].to(device)
                preds = model(X)
                val_loss += loss_fn(preds, y).item()
        train_loss /= num_train_batches
        val_loss /= num_val_batches
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch + 1) % mod_epochs == 0:
            print(
                f"Epoch: {epoch + 1}/{epochs}{5 * ' '}Training Loss: {train_loss:.4f}{5 * ' '}Validation Loss: {val_loss:.4f}")

        if early_stopping:
            ear_stopping(val_loss, model)
            if ear_stopping.early_stop:
                print("Early stopping")
                break

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("dark")
    ax.plot(range(1, counter_epochs + 1), train_loss_list, label='Train Loss')
    ax.plot(range(1, counter_epochs + 1), val_loss_list, label='Val Loss')
    ax.set_title("Train - Val Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    plt.legend()
    plt.show()

    if early_stopping:
        model.load_state_dict(torch.load("checkpoint.pt"))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter % 5 == 0:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def test_loop(model, test_dloader, device='cpu'):
    predictions_list = np.array([], dtype=np.int64)
    targets_list = np.array([], dtype=np.int64)
    model.eval()

    for val_sample in test_dloader:
        X = val_sample[0].to(device)
        y = val_sample[1].cpu().numpy()
        targets_list = np.concatenate((targets_list, y))

        with torch.no_grad():
            preds = model(X)
            predictions_list = np.concatenate((predictions_list,
                                               torch.argmax(preds, dim=-1).cpu().numpy()))
    return predictions_list, targets_list