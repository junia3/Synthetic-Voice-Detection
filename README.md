<p align="center">
    <img src="https://github.com/junia3/Synthetic-Voice-Detection/assets/79881119/4950c6ce-9a0f-4b48-ade5-c7dafb53d60e" width="1000">
</p>
<p align="center"; font-size=24px><b>Spoofing detection project in 2nd YAICON</b></p>

---

# Members

<p align="center"> <b>
</br> &nbsp; 🤗 박준영, YAI 9th
</br> &nbsp; 👤 변지혁, YAI 8th
</br> &nbsp; 👤 주다윤, YAI 10th
</br> &nbsp; 👤 김강현, YAI 10th
</b></p> 

---

<p align="center"><img src="https://user-images.githubusercontent.com/79881119/236765082-9a073601-d863-4900-b3ad-6c176ee39cc1.gif" width="400"></p>

# Requirements setting

### Cloning repository
```bash
git clone https://github.com/junia3/Synthetic-Speech-Detection.git
cd Synthetic-Speech-Detection
```

### Create conda environment with following commands
```bash
conda create -n ssd python=3.8
conda activate ssd
pip install ipykernel # Optional
python -m ipykernel install --user --name ssd --display-name ssd # Optional
```
Actually, you do not need to add environment kernel for jupyter notebook(optional)

### Install python requirements

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install spafe matplotlib soundfile tqdm torchsummary
```
---

# Download dataset
This project basically use ASVspoof-2019 dataset, which can be downloaded on [this page](https://datashare.ed.ac.uk/handle/10283/3336).
Or you can just download LA.zip file with following command(recommended).

```bash
curl -o LA.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y
unzip LA.zip -d datasets
```

Then, your repository will have following structure.
```bash
Synthetic-Speech-Detection
├── datasets
│   ├── dataset.py
│   ├── LA
│   │    ├── ASVspoof2019_LA_asv_protocols
│   │    ├── ASVspoof2019_LA_asv_scores
│   │    ├── ...
``` 

---

# Dataset
Detailed implementation is written on [dataset page](./datasets).

---

# Train baseline
Check file ['baseline.ipynb'](./baseline.ipynb).

Hello and welcome to the README for this project!

If you are planning to use the baseline model for training, we strongly recommend that you train for more than 50 epochs. This is because in the initial training steps, the validation loss may be unstable and fluctuate quite a bit. However, with more epochs, the validation loss tends to stabilize and converge to a more meaningful value.

So, to ensure that you get the best results from your training, we suggest training the baseline model for at least 50 epochs. Of course, you may need to adjust this number depending on the specifics of your project and dataset.

Thank you for using our code, and we wish you the best of luck with your training!

---

# Evaluation metric
<p align="center">
    <img src="https://github.com/junia3/Synthetic-Speech-Detection/assets/79881119/15a3f70b-a848-4231-8bc2-a45281ddcbc4" width="800"> 
</p>

### Equal Error Rate(EER) Metric
> Audio spoofing is the act of attempting to deceive a system that relies on audio input, such as a speaker recognition system, by playing a recording of a legitimate user's voice instead of speaking in real-time. Detecting audio spoofing is important to prevent unauthorized access to sensitive information and protect against fraud. One way to evaluate the performance of an audio spoofing detection system is to use the EER metric. The EER is the point at which the false acceptance rate (FAR) and the false rejection rate (FRR) are equal.

> The FAR is the proportion of spoof attacks that are incorrectly accepted as genuine, while the FRR is the proportion of genuine attempts that are incorrectly rejected as spoof attacks. Ideally, a spoofing detection system should have low values for both FAR and FRR.

> To calculate the EER, a system is tested on a dataset of both genuine and spoofed audio samples, and the FAR and FRR are calculated at different thresholds. The threshold represents the level of confidence required for a system to classify a sample as either genuine or spoofed. The EER is the point where the FAR and FRR intersect on a Receiver Operating Characteristic (ROC) curve.

> In summary, the EER metric is a useful way to evaluate the performance of an audio spoofing detection system. It takes into account both false acceptance and false rejection rates, and provides a single value that represents the level of performance achieved by the system.

### tandem Detection Cost Function (t-DCF)
> The t-DCF is a commonly used metric for evaluating the performance of audio spoofing detection systems, especially in the context of speaker verification systems. The t-DCF metric takes into account both the detection accuracy of the system as well as the potential financial cost associated with false acceptance and false rejection errors.

> The t-DCF is calculated as the weighted sum of two costs: the false alarm cost (Cfa) and the missed detection cost (Cmiss). The false alarm cost represents the cost associated with incorrectly accepting a spoof attack as genuine, while the missed detection cost represents the cost associated with incorrectly rejecting a genuine attempt as a spoof attack.

> The t-DCF metric is computed using a scoring function that assigns a score to each audio sample based on the likelihood that it is a genuine or a spoofed sample. The t-DCF metric is then calculated using a set of parameters that define the costs of false acceptance and false rejection errors, as well as the prior probabilities of genuine and spoofed samples in the dataset.

> The t-DCF metric is especially useful for evaluating the performance of audio spoofing detection systems in real-world scenarios where the cost of false alarms and missed detections can be high. For example, in a speaker verification system used for financial transactions, a false alarm could result in unauthorized access to an account, while a missed detection could result in a legitimate user being denied access to their own account.

> In summary, the t-DCF metric is a widely used evaluation metric in the field of audio spoofing detection, which takes into account the financial costs associated with false acceptance and false rejection errors.

---

# Front-end demo web

### Requirements

Basically you need some requirements to run this program(for augmentation)
```bash
pip install soundfile
pip install audiomentations
```

### Implemented with pytorch Flask 
You should match model specification(pre-trained model) with service setting.

- Default setting(transform = LFCC, feature length = $750$) 
```bash
python service.py
```

- Free setting(transform or feature length)
```bash
python service.py --feature 500 --transform cqcc
```

### Download pretrained model(to be updated)
Be sure you have to 'download' best_model.pt(pretrained model) and locate it on the same directory as ```service.py```   
The front-end web is running on [your local URL](http://127.0.0.1:5000/).

|Model|EER(%)|t-DCF|Acc(%)|Link|Model|EER(%)|t-DCF|Acc(%)|Link|
|---|---|---|---|---|---|---|---|---|---|
|RN18/500/LFCC|6.74370|0.18492|67.7219|[download](https://drive.google.com/file/d/14dbuA_i70sSgEcjZwglyxTLZr0f8NiSm/view?usp=share_link)|RN18/750/LFCC|6.19861|0.13363|75.7644|[download](https://drive.google.com/file/d/1AvSWsj3aaPlo9V0fDQOTb6RJpqW8cqBl/view?usp=share_link)|
|RN34/500/LFCC|7.24569|0.18298|81.2626|[download](https://drive.google.com/file/d/12LMzvkdQA6P3TgaYwYRGWCKArwA1J0mQ/view?usp=share_link)|RN34/750/LFCC|8.40194|0.17825|60.7552|[download](https://drive.google.com/file/d/1qpV1WhHUKx6Lq73LWIRCz8Kr1QsDNMQn/view?usp=share_link)|

### Upload audio sample

Press button 'Choose an audio file' and upload audio data

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236757128-d89c4001-864c-48b6-8564-e6a10f0cee15.png" width="700">
</p>
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236757249-a2ba17ec-a23f-436c-8bd5-2ecbe4b67e1a.png" width="700">
</p>

If you uploaded audio file properly, click 'Upload and play'.
Apply augmentation on your data. If want to run it with default setting, just set two values $0$. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236757504-e8359cb6-52f2-42b4-bbd6-6f41225cdc4d.png" width="700">    
</p>

Apply augmentation and wait for data pre-processing. Then click 'Try Me!' button. You can use our service for free in your own device.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236757663-53f99025-e88f-4a8c-89cd-0eebdc5cfea8.png" width="700">
</p>

After inference is over, the result is presented on the page! It only takes few seconds for 10 sec sample even in **CPU environment**!

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236757854-0eac19d2-6422-4e82-b3c2-4ba95ea568d2.png" width="700">    
</p>

And with "show result" button, you can check the specification details on model prediction!!

<p align="center">
    <img src="https://github.com/junia3/Synthetic-Voice-Detection/assets/79881119/1e18f9e3-09c7-4b6e-bfcb-b6809f630f02" width="400">
    <img src="https://github.com/junia3/Synthetic-Voice-Detection/assets/79881119/c96b03fe-d79d-42b4-8d60-c114bfe9f725" width="400">
</p>

As the probability of "Prediction" approaches zero, the probability of it being a synthesized voice is low. In other words, on the graph, the blue color (values less than 0.5) represents the location predicted to be a real voice, while the red color (values greater than or equal to 0.5) represents the location predicted to be a spoof voice.

P.S. We have a small gift for anyone who runs the web demo application on their own device! Give it a try and discover what it is.
