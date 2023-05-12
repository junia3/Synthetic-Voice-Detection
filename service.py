import argparse
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
import matplotlib.pyplot as plt

def create_randomname(length):
    letters_set = string.ascii_letters
    return ''.join(random.choice(letters_set) for i in range(length))

def plot_waveform_w_pred(audio_path, prediction):
    # Load the audio file
    audio_data, sample_rate = sf.read(audio_path)

    # If the audio has multiple channels, we'll just use the first channel for now
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Calculate the time array
    duration = len(audio_data) / sample_rate
    time_array = np.linspace(0, duration, len(audio_data))

    # Normalize the audio data to the range [0, 1]
    norm_audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    # Plot waveform
    time_predarray = np.linspace(0, duration, len(prediction))
    
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(time_array, norm_audio_data, color='#333333', alpha=0.2)
    ax1.set_xlabel("Time (s)", fontsize=14, labelpad=10)
    ax1.set_ylabel("Amplitude", fontsize=14, labelpad=10)
    ax2 = ax1.twinx()
    
    # Assign colors to the line segments
    mask = np.ma.masked_less(prediction, 0.5)
    ax2.plot(time_predarray, prediction, color='#2D33F7', linewidth=2)
    ax2.plot(time_predarray, mask, color='#fd3412', linewidth=2)
    ax2.set_ylabel("Prediction", fontsize=14, labelpad=10)
    ax2.set_ylim(0, 1.5)
    
    # Remove previous samples
    for file in glob("static/graph/*.png"):
        os.remove(file)
    
    # Save graph with random name
    key=create_randomname(6)
    plt.savefig(f"static/graph/result_{key}.png")
    return key

# Run augmentation code
def voice_augment(audio_path, pitch, n_scale):
    # Apply the augmentation settings to your image processing code here
    audio, rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
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
    score = test_sample(audio, rate, "best_model.pt", args.feature, args.transform)[1]
    _, prediction = torch.max(torch.softmax(score, dim=1), dim=1)
    ensemble = prediction.float().mean()
    if ensemble > 0.5:
        infer = "This is a fake voice 😰"
    else:
        infer = "This is a real voice 🤗"
        
    # Plot the waveform with the prediction
    key = plot_waveform_w_pred(audio_path, torch.softmax(score, dim=1)[:, 1].cpu().numpy())
    return infer, key

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    audio_path = request.form['audio-path']
    inference_result, key = run_inference(audio_path=audio_path)
    graph_path = f"static/graph/result_{key}.png"
    return jsonify({"result" : inference_result, "graph" : graph_path})

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature', type=int, help="feature length", default=500)
    parser.add_argument("--transform", type=str, help="feature extraction method", default="lfcc")
    args = parser.parse_args()
    app.run(debug=True)
