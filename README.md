![header](https://capsule-render.vercel.app/api?type=waving&color=0:FF7F50,50:008B8B,100:8A2BE2&height=250&section=header&text=Synthetic%20Speech%20Detection&fontSize=45&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
<!-- 
<p align="center"><a href="#">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:F9D976,100:F39F86&height=250&section=header&text="Synthetic speech detection" &fontSize=40&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40" alt="header" />
</a></p>
 -->
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236666746-2bfa307e-c6de-4f30-9e42-2d1d21466799.png" width="500">
</p>
<p align="center"; font-size=24px><b>This project implemented with YAI members for 2nd YAICON</b></p>

---

# Members

<p align="center"> <b>
</br> &nbsp; 🤗 박준영, YAI 9th
</br> &nbsp; 👤 변지혁, YAI 8th
</br> &nbsp; 👤 주다윤, YAI 10th
</br> &nbsp; 👤 김강현, YAI 10th
</b></p> 

---

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
Check file ['baseline.ipynb'](./baseline.ipynb)

---

# Evaluation metric

---

# Front-end demo web

### Requirements

Basically you need some requirements to run this program(for augmentation)
```bash
pip install soundfile
pip install audiomentations
```

### Implemented with pytorch Flask 
```bash
python service.py
```

### Download pretrained model(to be updated)
Be sure you have to 'download' best_model.pt(pretrained model) and locate it on the same directory as ```service.py```   
The front-end web is running on [your local](http://127.0.0.1:5000/)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236666905-c6a90b49-4a84-4a08-b1de-42a4ffece525.png" width="400">    
</p>


### Upload audio sample

Press button 'Choose an audio file' and upload audio data

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236666882-efb92bc4-5200-4ca9-a3e7-a9951aeb2242.png" width="350">
    <img src="https://user-images.githubusercontent.com/79881119/236666930-7686fc8b-165d-4d7b-ba89-92bb89512e7e.png" width="400">
</p>

If you uploaded audio file properly, click 'Upload and play'.
Apply augmentation on your data. If want to run it with default setting, just set two values $0$. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236666979-74664dca-bd95-4844-8a7a-0cf4b9af33a5.png" width="400">    
</p>

Apply augmentation and wait for data pre-processing. Then click 'Try Me!' button. You can use our service for free in your own device.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236667053-bc5fc6b3-a6b1-40b2-93b5-fb51cb9a9858.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/236667257-7cf5fb6d-39f3-4f35-91fa-e811f9b6777d.png" width="400">  
</p>

After inference is over, the result is presented on the page! It only takes few seconds for 10 sec sample even in CPU environment!

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/236667484-5657d4d0-4eac-4614-9f85-6cb243885a4c.png" width="400">    
</p>

"P.S. We have a small gift for anyone who runs the web demo application on their own device! Give it a try and discover what it is."

![footer](https://capsule-render.vercel.app/api?type=waving&color=0:FF7F50,50:008B8B,100:8A2BE2&height=150&section=footer&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
