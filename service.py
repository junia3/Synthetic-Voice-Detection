from flask import Flask, request, render_template, jsonify
import soundfile as sf
import torch
from inference_sample import test_sample
from audiomentations import *
import numpy as np
import string
import random
import os
from glob import glob

# Run augmentation code
def voice_augment(audio_path, pitch, n_scale):
    # Apply the augmentation settings to your image processing code here
    audio, rate = sf.read(audio_path)
    # Pitch shift the audio
    augment = Compose([
        PitchShift(min_semitones=pitch, max_semitones=pitch, p=1),
    ])
    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=audio, sample_rate=rate)
    noise, _ = sf.read("noise.wav")
    if noise.ndim > 1:
        noise = noise[:, 0]
    
    # Make sure the noise is at least as long as the audio
    if len(noise) < len(audio):
        n_repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, n_repeats)[:len(audio)]
    else:
        noise = noise[:len(audio)]

    # Calculate the scaling factor for the noise based on the desired intensity
    rms_audio = np.sqrt(np.mean(np.square(augmented_samples)))
    rms_noise = np.sqrt(np.mean(np.square(noise)))
    scale = (rms_audio / rms_noise) * n_scale
    
    # Add the noise to the audio at the desired level of intensity
    augmented_samples = augmented_samples + (noise * scale)
    return augmented_samples, rate

# Run estimation code
def run_inference(audio_path):
    audio, rate = sf.read(audio_path)
    score = test_sample(audio, rate, "best_model.pt")[1]
    _, prediction = torch.max(torch.softmax(score, dim=1), dim=1)
    ensemble = prediction.float().mean()
    if ensemble > 0.5:
        infer = "This is a fake voice 😰"
    else:
        infer = "This is a real voice 🤗"
    return infer

def create_randomname(length):
    letters_set = string.ascii_letters
    return ''.join(random.choice(letters_set) for i in range(length))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    audio_path = request.form['audio-path']
    inference_result = run_inference(audio_path=audio_path)
    return jsonify({"result" : inference_result})

@app.route('/apply-augmentation', methods=['POST'])
def apply_augmentation():
    # Remove previous samples
    for file in glob("static/audio/*.wav"):
        os.remove(file)
    
    # Get audio file
    audio_file = request.files['audio-file']
    audio_file.save('static/audio/og_audio.wav')
    pitch = float(request.form['pitch'])
    noise = float(request.form['noise'])
    audio, rate = voice_augment("static/audio/og_audio.wav", pitch, noise)
    key=create_randomname(6)
    aug_path = f"static/audio/aug_{key}_audio.wav"
    sf.write(aug_path, audio, rate)
    return jsonify({"result" : aug_path})

if __name__ == "__main__":
    app.run(debug=True)