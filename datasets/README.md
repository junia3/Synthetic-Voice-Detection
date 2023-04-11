# Feature extraction for audio file
### Prepare audio sample

```python
from spafe.features import lfcc, bfcc, cqcc, mfcc, msrcc, ngcc, pncc, psrcc, rplp
from spafe.utils.preprocessing import SlidingWindow
import soundfile as sf
import numpy as np
from spafe.utils.vis import show_features
from spafe.utils.vis import show_spectrogram

filepath= 'sample.wav'
```

---

### Feature extraction functions with spafe module(example)

```python
def feature_func(sample_path, option, n_fft=512, w_len=0.02, num_features=20, norm="mvn"):
    audio, rate = sf.read(sample_path)
    length = len(audio)
    # Transform audio with feature extraction method
    # (0) Linear spectogram
    if option == 'lspec':
        feature, _ = lfcc.linear_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                                window=SlidingWindow(w_len, w_len/2, "hamming"),
                                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)
    
    # (1) Linear frequency cepstral coefficient
    elif option == "lfcc":
        feature = lfcc.lfcc(audio, fs=rate,
                            pre_emph=1, num_ceps=num_features,
                            pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
    # (2) Bark spectogram
    elif option == 'bspec':
        feature, _ = bfcc.bark_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                            window=SlidingWindow(w_len, w_len/2, "hamming"),
                                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)

    # (3) Bark frequency cepstral coefficient
    elif option == 'bfcc':
        feature = bfcc.bfcc(audio, fs=rate,
                            pre_emph=1, num_ceps=num_features,
                            pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    # (4) Constant Q-transform spetogram
    elif option == 'cqspec':
        feature = cqcc.cqt_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                        window=SlidingWindow(w_len, w_len/2, "hamming"),
                                        nfft=n_fft, low_freq=0, high_freq=rate/2).T
        feature = np.absolute(feature)

    # (5) Constant Q-transform Cepstral Coeﬃcients
    elif option == 'cqcc':
        feature = cqcc.cqcc(audio, fs=rate,
                            pre_emph=1, num_ceps=num_features,
                            pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                            nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
    # (6) Mel spectogram
    elif option == 'mspec':
        feature, _ = mfcc.mel_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                            window=SlidingWindow(w_len, w_len/2, "hamming"),
                                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)

    # (7) Inverse Mel frequency cepstral coefficient
    elif option == 'imfcc':
        feature = mfcc.imfcc(audio, fs=rate,
                                pre_emph=1, num_ceps=num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    # (8) Mel frequency cepstral coefficient
    elif option == 'mfcc':
        feature = mfcc.mfcc(audio, fs=rate,
                            pre_emph=1, num_ceps=num_features,
                            pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
    # (9) Magnitude based Spectral Root Cepstral Coefficients
    elif option == 'msrcc':
        feature = msrcc.msrcc(audio, fs=rate,
                                pre_emph=1, num_ceps=num_features,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    # (10) Normalized Gammachirp Cepstral Coefficients
    elif option == 'ngcc':
        feature  = ngcc.ngcc(audio, fs=rate,
                                num_ceps=num_features, pre_emph=1,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    elif option == 'pncc':
        feature  = pncc.pncc(audio, fs=rate,
                                num_ceps=num_features, pre_emph=1,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    elif option == 'psrcc':
        feature = psrcc.psrcc(audio, fs=rate,
                                num_ceps=num_features, pre_emph=1,
                                pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    elif option == 'plp':
        feature = rplp.plp(audio, fs=rate,
                            order=num_features, pre_emph=1,
                            pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

    elif option == 'rplp':
        feature = rplp.rplp(audio, fs=rate,
                            order=num_features, pre_emph=1,
                            pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                            nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)
    
    return feature, length
```

---

### 1-1. Linear spectrogram

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231042758-0162a90d-2c26-4178-ba7e-63028dd009ec.png" width="1000">
</p>

### 1-2. Linear Frequency Cepstral Coefficient(LFCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231042911-11332867-cbb4-4d5e-8b52-02f58c1bea92.png" width="1000">
</p>

### 2-1. Mel spectogram

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043122-24c2edad-9f04-453c-82f8-ac8ea067e686.png" width="1000">
</p>

### 2-2. Mel Frequency Cepstral Coefficient(MFCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043166-58009f9b-6a22-46e1-b37d-552dfa2b8c47.png" width="1000">
</p>

### 2-3. Inverse Mel Frequency Cepstral Coefficient(IMFCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043346-2fc08438-b8da-4af1-a467-ea12fec5dd84.png" width="1000">
</p>

### 3-1. Bark spectrogram

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043447-33a9f871-b32b-4c7b-83b3-5971b352cda4.png" width="1000">
</p>

### 3-2. Bark Frequency Cepstral Coefficient

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043388-a69e0a27-6849-440d-8ce6-894f8a3b731d.png" width="1000">
</p>

### 4-1. Constant Q-Transform spectrogram

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043674-7354a9a0-ecd7-4281-87a9-ae321a8a437e.png" width="1000">
</p>

### 4-2. Constant Q-Transform Cepstral Coefficient(CQCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043634-259bb003-1037-4415-810b-b41dd5937eea.png" width="1000">
</p>

### 5. Magnitude based Spectral Root Cepstral Coefficient(MSRCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043717-db31fbeb-4342-422a-8c0a-27b982c5b3a9.png" width="1000">
</p>

### 6. Normalized Gammachirp Cepstral Coefficient(NGCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043753-27c0a811-80c2-4548-91e3-cba245338bdd.png" width="1000">
</p>

### 7. Power-Normalized Cepstral Coefficient(PNCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043792-78bb0e52-50ff-4d7b-95ad-dc0b4ccda26c.png" width="1000">
</p>

### 8. Phase based Spectral Root Cepstral Coefficient(PSRCC)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043843-c8b107bb-ea2a-49bb-8dd5-f086a088f8e5.png" width="1000">
</p>

### 9-1. Perceptual linear predictions(PLP)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043905-704e3eb6-deea-467b-9eba-e2e43b0637f4.png" width="1000">
</p>

### 9-2. Rasta Perceptual linear predictions(RPLP)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043932-809f3de5-8e09-4698-8da3-39d376c0b53f.png" width="1000">
</p>
