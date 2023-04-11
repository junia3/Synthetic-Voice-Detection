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
audio, rate = sf.read(filepath)
length = len(audio)

# Configuration
w_len = 0.02
n_fft = 2048
num_features = 20
```

---

### 1-1. Linear spectrogram

```python
feature, _ = lfcc.linear_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                    window=SlidingWindow(w_len, w_len/2, "hamming"),
                                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)
                                    
show_spectrogram(feature.T, 16000, xmin=0, xmax=length/16000, ymin=0, ymax=(16000/2)/1000,
                 dbf=80.0, xlabel="Time (s)", ylabel="Frequency (kHz)", title="Linear spectrogram (dB)", cmap="jet")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231042758-0162a90d-2c26-4178-ba7e-63028dd009ec.png" width="800">
</p>

### 1-2. Linear Frequency Cepstral Coefficient(LFCC)

```python
feature = lfcc.lfcc(audio, fs=rate,
                    pre_emph=1, num_ceps=num_features,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Linear Frequency Cepstral Coefﬁcients", "LFCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231042911-11332867-cbb4-4d5e-8b52-02f58c1bea92.png" width="800">
</p>

### 2-1. Mel spectogram

```python
feature, _ = mfcc.mel_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)
                                
show_spectrogram(feature.T, 16000, xmin=0, xmax=length/16000, ymin=0, ymax=(16000/2)/1000,
                 dbf=80.0, xlabel="Time (s)", ylabel="Frequency (kHz)", title="Mel spetogram (dB)", cmap="jet")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043122-24c2edad-9f04-453c-82f8-ac8ea067e686.png" width="800">
</p>

### 2-2. Mel Frequency Cepstral Coefficient(MFCC)

```python
feature = mfcc.mfcc(audio, fs=rate,
                    pre_emph=1, num_ceps=num_features,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Mel Frequency Cepstral Coefﬁcients", "MFCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043166-58009f9b-6a22-46e1-b37d-552dfa2b8c47.png" width="800">
</p>

### 2-3. Inverse Mel Frequency Cepstral Coefficient(IMFCC)

```python
feature = mfcc.imfcc(audio, fs=rate,
                    pre_emph=1, num_ceps=num_features,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Inverse Mel Frequency Cepstral Coefﬁcients", "IMFCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043346-2fc08438-b8da-4af1-a467-ea12fec5dd84.png" width="800">
</p>

### 3-1. Bark spectrogram

```python
feature, _ = bfcc.bark_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                    window=SlidingWindow(w_len, w_len/2, "hamming"),
                                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2)

show_spectrogram(feature.T, 16000, xmin=0, xmax=length/16000, ymin=0, ymax=(16000/2)/1000,
                 dbf=80.0, xlabel="Time (s)", ylabel="Frequency (kHz)", title="Bark spectrogram (dB)", cmap="jet")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043447-33a9f871-b32b-4c7b-83b3-5971b352cda4.png" width="800">
</p>

### 3-2. Bark Frequency Cepstral Coefficient

```python
feature = bfcc.bfcc(audio, fs=rate,
                    pre_emph=1, num_ceps=num_features,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Bark Frequency Cepstral Coefﬁcients", "BFCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043388-a69e0a27-6849-440d-8ce6-894f8a3b731d.png" width="800">
</p>

### 4-1. Constant Q-Transform spectrogram

```python
feature = cqcc.cqt_spectrogram(audio, fs=rate, pre_emph=0, pre_emph_coeff=0.97,
                                window=SlidingWindow(w_len, w_len/2, "hamming"),
                                nfft=n_fft, low_freq=0, high_freq=rate/2).T
feature = np.absolute(feature)

show_spectrogram(feature.T, 16000, xmin=0, xmax=length/16000, ymin=0, ymax=(16000/2)/1000,
                 dbf=80.0, xlabel="Time (s)", ylabel="Frequency (kHz)", title="Constant Q-transform spetogram (dB)", cmap="jet")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043674-7354a9a0-ecd7-4281-87a9-ae321a8a437e.png" width="800">
</p>

### 4-2. Constant Q-Transform Cepstral Coefficient(CQCC)

```python
feature = cqcc.cqcc(audio, fs=rate,
                    pre_emph=1, num_ceps=num_features,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Constant Q-transform Cepstral Coefﬁcients", "CQCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043634-259bb003-1037-4415-810b-b41dd5937eea.png" width="800">
</p>

### 5. Magnitude based Spectral Root Cepstral Coefficient(MSRCC)

```python
feature = msrcc.msrcc(audio, fs=rate,
                    pre_emph=1, num_ceps=num_features,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Magnitude based Spectral Root Cepstral Coefficients", "MSRCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043717-db31fbeb-4342-422a-8c0a-27b982c5b3a9.png" width="800">
</p>

### 6. Normalized Gammachirp Cepstral Coefficient(NGCC)

```python
feature  = ngcc.ngcc(audio, fs=rate,
                    num_ceps=num_features, pre_emph=1,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Normalized Gammachirp Cepstral Coefficients", "NGCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043753-27c0a811-80c2-4548-91e3-cba245338bdd.png" width="800">
</p>

### 7. Power-Normalized Cepstral Coefficient(PNCC)

```python
feature  = pncc.pncc(audio, fs=rate,
                    num_ceps=num_features, pre_emph=1,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Power-Normalized Cepstral Coefficients", "PNCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043792-78bb0e52-50ff-4d7b-95ad-dc0b4ccda26c.png" width="800">
</p>

### 8. Phase based Spectral Root Cepstral Coefficient(PSRCC)

```python
feature = psrcc.psrcc(audio, fs=rate,
                    num_ceps=num_features, pre_emph=1,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Phase based Spectral Root Cepstral Coefficients", "PSRCC Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043843-c8b107bb-ea2a-49bb-8dd5-f086a088f8e5.png" width="800">
</p>

### 9-1. Perceptual linear predictions(PLP)

```python
feature = rplp.plp(audio, fs=rate,
                    order=num_features, pre_emph=1,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Perceptual linear prediction coefficents", "PLP Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043905-704e3eb6-deea-467b-9eba-e2e43b0637f4.png" width="800">
</p>

### 9-2. Rasta Perceptual linear predictions(RPLP)

```python
feature = rplp.rplp(audio, fs=rate,
                    order=num_features, pre_emph=1,
                    pre_emph_coeff=0.97, window=SlidingWindow(w_len, w_len/2, "hamming"),
                    nfilts=128, nfft=n_fft, low_freq=0, high_freq=rate/2, normalize=norm)

show_features(feature, "Rasta Perceptual linear prediction coefficents", "RPLP Index", "Frame Index")
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/231043932-809f3de5-8e09-4698-8da3-39d376c0b53f.png" width="800">
</p>
