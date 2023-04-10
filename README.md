![header](https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=250&section=header&text=Synthetic%20Speech%20Detection&fontSize=45&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
<!-- 
<p align="center"><a href="#">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:F9D976,100:F39F86&height=250&section=header&text="Synthetic speech detection" &fontSize=40&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40" alt="header" />
</a></p>
 -->

<p align="center"; font-size=24px><b>This project implemented with YAI members for YAICON 2nd Season</b></p>

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
Check file 'baseline.ipynb'

![footer](https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=150&section=footer&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
