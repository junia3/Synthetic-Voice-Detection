import os
from torch.utils.data import Dataset
from collections import defaultdict
import torch
import numpy as np
from spafe.features import lfcc, bfcc, cqcc, mfcc, msrcc, ngcc, pncc, psrcc, rplp
from spafe.utils.preprocessing import SlidingWindow
import soundfile as sf

class LADataset(Dataset):
    '''
        Dataset class for ASVspoof 2019 dataset (LA Only)
        ---
        Input :
            split : dataset split ('train', 'eval', 'dev')
            txtpath : text file path (protocol file)
            datadir : audio file path (.flac file)
            transforms : feature extration method
            num_features : number of cepstral coefficients
            n_fft : fourier transform number(in time domain) for FFT
            w_len : Hamming window length for short time fourier transform
            norm : normalization
            expand : expand only first cepstral coefficient
        
        ---
        Output(getitem):
            feature : pre-processed feature
            label : ground truths
    '''

    # (0) Define meta information for training dataset
    speaker_map = {
            'LA_0079': 0, 'LA_0080': 1, 'LA_0081': 2, 'LA_0082': 3, 'LA_0083': 4, 'LA_0084': 5, 'LA_0085': 6,
            'LA_0086': 7, 'LA_0087': 8,
            'LA_0088': 9, 'LA_0089': 10, 'LA_0090': 11, 'LA_0091': 12, 'LA_0092': 13, 'LA_0093': 14, 'LA_0094': 15,
            'LA_0095': 16, 'LA_0096': 17,
            'LA_0097': 18, 'LA_0098': 19
    }
    tag_map = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
           "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18, "A19": 19}
    label_map = {"spoof": 1, "bonafide": 0}

    def __init__(self, split='train', txtpath=None, datadir=None, transforms=None,
                 f_len=750, padding='repeat', num_features=20, n_fft=512, w_len=0.02, norm=None, expand=False):

        self.transforms = transforms
        # Default setting
        if self.transforms is None:
            self.transforms = 'mfcc'
        
        # Available feature extractions
        assert self.transforms in ['lspec', 'lfcc', 'bspec', 'bfcc', 'cqspec', 'cqcc', 'mspec',
                                   'imfcc', 'mfcc', 'msrcc', 'ngcc', 'pncc', 'psrcc', 'plp', 'rplp'], "Not available feature extraction method"

        # Feature extraction parameters
        self.split = split
        self.num_features = num_features
        self.n_fft = n_fft
        self.w_len = w_len
        self.f_len = f_len
        self.padding = padding
        self.norm = norm
        self.expand = expand
        
        if ("cc" not in self.transforms and "lp" not in self.transforms) and self.expand:
            print("[Warning] : Expanding feature only works on cepstral coefficient methods")

        assert self.norm in ['mvn', 'ms', 'vn', 'mn', None], "Unavailable normalization method"

        # (1) Bring annotation from txtfile
        if txtpath is None:
            if split == "train":
                txtpath = os.path.abspath(f"./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.trl.txt")
            else:
                txtpath = os.path.abspath(f"./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.trl.txt")

        assert os.path.isfile(txtpath), "Please check annotation file"

        # (2) Bring dataset directory
        if datadir is None:
            datadir = f"./LA/ASVspoof2019_LA_{split}"
        assert os.path.isdir(datadir), "Please check ASVspoof dataset"

        # (3) List-up proper dataset
        self.dataset = defaultdict(list)
        with open(txtpath, "r") as file:
            for line in file:
                info = line.split()
                speaker, filename, tag, label = info[0], info[1], info[-2], info[-1]
                self.dataset["path"].append(os.path.join(datadir, f"flac/{filename}.flac"))
                self.dataset["label"].append(self.label_map[label])
                if split=="train":
                    self.dataset["speaker"].append(self.speaker_map[speaker])
                    self.dataset["tag"].append(self.tag_map[tag])

    # Padeding method with zero
    def _zeropad(self, feature):
        frame, _ = feature.shape
        # Only padding when current length is smaller than feautre length 
        assert self.f_len > frame
        pad_len = self.f_len - frame
        return np.pad(feature, ((0, pad_len), (0, 0)), 'constant', constant_values=0)

    # Padding method with repeat
    def _reppad(self, feature):
        frame, _ = feature.shape
        # Only padding when current length is smaller than feature length
        assert self.f_len > frame
        repeat_num = int(np.ceil(self.f_len/frame))
        return np.repeat(feature, repeat_num, axis=0)[:self.f_len, :]

    # Random cut method
    def _rancut(self, feature):
        frame, _ = feature.shape
        # Only cut when current length is larger than feature length
        assert self.f_len < frame
        s_point = np.random.randint(frame-self.f_len)
        return feature[s_point:s_point+self.f_len, :]

    def __len__(self):
        return len(self.dataset["path"])

    def __getitem__(self, idx):
        # Load audio and labels from dictionary
        sample_path = self.dataset["path"][idx]
        sample_label = self.dataset["label"][idx]

        if self.split == "train":
            sample_speaker = self.dataset["speaker"][idx]
            sample_tag = self.dataset["tag"][idx]
            label = {"label": sample_label, "tag": sample_tag, "speaker": sample_speaker}
        else:
            label = {"label": sample_label}
        
        audio, rate = sf.read(sample_path)
        # Transform audio with feature extraction method
        # (0) Linear spectogram
        if self.transforms == 'lspec':
            feature, _ = lfcc.linear_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                                 window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                                 nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2)
        
        # (1) Linear frequency cepstral coefficient
        elif self.transforms == "lfcc":
            feature = lfcc.lfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=self.num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)
        
        # (2) Bark spectogram
        elif self.transforms == 'bspec':
            feature, _ = bfcc.bark_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                               window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                               nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2)
    
        # (3) Bark frequency cepstral coefficient
        elif self.transforms == 'bfcc':
            feature = bfcc.bfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=self.num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        # (4) Constant Q-transform spetogram
        elif self.transforms == 'cqspec':
            feature = cqcc.cqt_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                           window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                           nfft=self.n_fft, low_freq=0, high_freq=rate/2).T
            feature = np.absolute(feature)

        # (5) Constant Q-transform Cepstral Coeﬃcients
        elif self.transforms == 'cqcc':
            feature = cqcc.cqcc(audio, fs=rate,
                                pre_emph=1, num_ceps=self.num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)
        
        # (6) Mel spectogram
        elif self.transforms == 'mspec':
            feature, _ = mfcc.mel_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                              window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                              nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2)

        # (7) Inverse Mel frequency cepstral coefficient
        elif self.transforms == 'imfcc':
            feature = mfcc.imfcc(audio, fs=rate,
                                 pre_emph=1, num_ceps=self.num_features,
                                 pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                 nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        # (8) Mel frequency cepstral coefficient
        elif self.transforms == 'mfcc':
            feature = mfcc.mfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=self.num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)
        
        # (9) Magnitude based Spectral Root Cepstral Coefficients
        elif self.transforms == 'msrcc':
            feature = msrcc.msrcc(audio, fs=rate,
                                  pre_emph=1, num_ceps=self.num_features,
                                  pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                  nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        # (10) Normalized Gammachirp Cepstral Coefficients
        elif self.transforms == 'ngcc':
            feature  = ngcc.ngcc(audio, fs=rate,
                                 num_ceps=self.num_features, pre_emph=1,
                                 pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                 nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        elif self.transforms == 'pncc':
            feature  = pncc.pncc(audio, fs=rate,
                                 num_ceps=self.num_features, pre_emph=1,
                                 pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                 nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        elif self.transforms == 'psrcc':
            feature = psrcc.psrcc(audio, fs=rate,
                                  num_ceps=self.num_features, pre_emph=1,
                                  pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                  nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        elif self.transforms == 'plp':
            feature = rplp.plp(audio, fs=rate,
                               order=self.num_features, pre_emph=1,
                               pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                               nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)

        elif self.transforms == 'rplp':
            feature = rplp.rplp(audio, fs=rate,
                                order=self.num_features, pre_emph=1,
                                pre_emph_coeff=0.97, window=SlidingWindow(self.w_len, self.w_len/2, "hamming"),
                                nfilts=128, nfft=self.n_fft, low_freq=0, high_freq=rate/2, normalize=self.norm)
        
        if feature.shape[0] > self.f_len:
            feature = self._rancut(feature)
        
        if feature.shape[0] < self.f_len:
            if self.padding == 'zero':
                feature = self._zeropad(feature)
            elif self.padding == 'repeat':
                feature = self._reppad(feature)
            else:
                raise ValueError('Zero padding or Repeat padding is available')

        if ("cc" in self.transforms or "lp" in self.transforms) and self.expand:
            feature = np.repeat(np.expand_dims(feature[:, 0], axis=1), 20, axis=1)

        return torch.FloatTensor(feature).unsqueeze(0), label

def collate_fn(batch):
    inputs = torch.stack([sample[0] for sample in batch], dim=0)
    # Make targets as batch of dictionary
    targets = {}
    targets["label"] = torch.LongTensor([sample[1]["label"] for sample in batch])
    if len(list(batch[0][1].keys())) > 1:
        targets["tag"] = torch.LongTensor([sample[1]["tag"] for sample in batch])
        targets["speaker"] = torch.LongTensor([sample[1]["speaker"] for sample in batch])

    # Return features, targets and original lengths before padding
    return inputs, targets