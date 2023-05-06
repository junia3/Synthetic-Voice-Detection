import torch
import numpy as np
from spafe.features import lfcc, bfcc, cqcc, mfcc, msrcc, ngcc, pncc, psrcc, rplp
from spafe.utils.preprocessing import SlidingWindow
import soundfile as sf

# Code from dataset.py
def zeropad(feature, f_len):
    frame, _ = feature.shape
    # Only padding when current length is smaller than feautre length 
    assert f_len > frame
    pad_len = f_len - frame
    return np.pad(feature, ((0, pad_len), (0, 0)), 'constant', constant_values=0)

# Padding method with repeat
def reppad(feature, f_len):
    frame, _ = feature.shape
    # Only padding when current length is smaller than feature length
    assert f_len > frame
    repeat_num = int(np.ceil(f_len/frame))
    return np.repeat(feature, repeat_num, axis=0)[:f_len, :]

# Random cut method
def rancut(feature, f_len):
    frame, _ = feature.shape
    # Only cut when current length is larger than feature length
    assert f_len < frame
    s_point = np.random.randint(frame-f_len)
    return feature[s_point:s_point+f_len, :]

# Split feature in same dimension/+interval
def sample_feature(audio, rate, transforms="lfcc"):
    w_len = 0.02
    num_features = 20
    n_fft = 512
    norm = None
    f_len = 300
    interv = int(150*(rate/100))
    padding = 'repeat'
    length = len(audio)
    # start pointer and end pointer for feature
    # the interval between two pointer should be f_len
    s_pointer, e_pointer = 0, int(rate*(f_len/100))
    audios = []
    while e_pointer < length:
        audios += [audio[s_pointer:e_pointer]]
        s_pointer += interv
        e_pointer += interv
    
    if s_pointer < length:
        audios += [audio[s_pointer:]]

    features = []
    for audio in audios:
        # Transform audio with feature extraction method
        # (0) Linear spectogram
        if transforms == 'lspec':
            feature, _ = lfcc.linear_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                                 window=SlidingWindow(w_len, w_len/2, "hamming"),
                                                 nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)
        
        # (1) Linear frequency cepstral coefficient
        elif transforms == "lfcc":
            feature = lfcc.lfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
        
        # (2) Bark spectogram
        elif transforms == 'bspec':
            feature, _ = bfcc.bark_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                               window=SlidingWindow(w_len, w_len/2, "hamming"),
                                               nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)
    
        # (3) Bark frequency cepstral coefficient
        elif transforms == 'bfcc':
            feature = bfcc.bfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        # (4) Constant Q-transform spetogram
        elif transforms == 'cqspec':
            feature = cqcc.cqt_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                           window=SlidingWindow(w_len, w_len/2, "hamming"),
                                           nfft=n_fft, low_freq=0, high_freq=rate/2).T
            feature = np.absolute(feature)
    
        # (5) Constant Q-transform Cepstral Coe?cients
        elif transforms == 'cqcc':
            feature = cqcc.cqcc(audio, fs=rate,
                                pre_emph=1, num_ceps=num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
        
        # (6) Mel spectogram
        elif transforms == 'mspec':
            feature, _ = mfcc.mel_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                              window=SlidingWindow(w_len, w_len/2, "hamming"),
                                              nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)
    
        # (7) Inverse Mel frequency cepstral coefficient
        elif transforms == 'imfcc':
            feature = mfcc.imfcc(audio, fs=rate,
                                 pre_emph=1, num_ceps=num_features,
                                 pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                 nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        # (8) Mel frequency cepstral coefficient
        elif transforms == 'mfcc':
            feature = mfcc.mfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
        
        # (9) Magnitude based Spectral Root Cepstral Coefficients
        elif transforms == 'msrcc':
            feature = msrcc.msrcc(audio, fs=rate,
                                  pre_emph=1, num_ceps=num_features,
                                  pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                  nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        # (10) Normalized Gammachirp Cepstral Coefficients
        elif transforms == 'ngcc':
            feature  = ngcc.ngcc(audio, fs=rate,
                                 num_ceps=num_features, pre_emph=1,
                                 pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                 nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        elif transforms == 'pncc':
            feature  = pncc.pncc(audio, fs=rate,
                                 num_ceps=num_features, pre_emph=1,
                                 pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                 nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        elif transforms == 'psrcc':
            feature = psrcc.psrcc(audio, fs=rate,
                                  num_ceps=num_features, pre_emph=1,
                                  pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                  nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        elif transforms == 'plp':
            feature = rplp.plp(audio, fs=rate,
                               order=num_features, pre_emph=1,
                               pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                               nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
        elif transforms == 'rplp':
            feature = rplp.rplp(audio, fs=rate,
                                order=num_features, pre_emph=1,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
        
        if feature.shape[0] > f_len:
            feature = rancut(feature, f_len)
        
        if feature.shape[0] < f_len:
            if padding == 'zero':
                feature = zeropad(feature, f_len)
            elif padding == 'repeat':
                feature = reppad(feature, f_len)
            else:
                raise ValueError('Zero padding or Repeat padding is available')
        
        features += [torch.FloatTensor(feature)]

    return torch.FloatTensor(torch.stack(features, dim=0))

def test_sample(audio, rate, model_pth):
    feature = sample_feature(audio, rate, "lfcc")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature = feature.unsqueeze(1).to(device)
    model = torch.load(model_pth, map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(feature)
    
    return output