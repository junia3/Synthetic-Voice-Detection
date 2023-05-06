from flask import Flask, request, render_template, jsonify
import soundfile as sf
import torch
from inference_sample import test_sample

# Rung estimation code
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

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    audio_file = request.files['audio-file']
    audio_file.save('uploaded_audio.wav')
    inference_result = run_inference(audio_path="uploaded_audio.wav")
    return jsonify({"result" : inference_result})

if __name__ == "__main__":
    app.run(debug=True)
