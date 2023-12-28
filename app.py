from flask import Flask, render_template, request, redirect, url_for, session
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import streamlit as st

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secret key for session security

# Load TTS pipeline
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# Load speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Initialize session state
if 'tts_result' not in st.session_state:
    st.session_state.tts_result = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/account')
def account():
    return render_template('account.html')

@app.route('/tts', methods=['GET', 'POST'])
def text_to_speech():
    if request.method == 'POST':
        text_to_synthesize = request.form.get('text_to_synthesize')

        # Synthesize text to audio with speaker embedding
        speech = synthesizer(text_to_synthesize, forward_params={"speaker_embeddings": speaker_embedding})

        # Save audio locally
        audio_path = "static/speech.wav"
        sf.write(audio_path, speech["audio"], samplerate=speech["sampling_rate"])

        # Update session state with TTS result
        st.session_state.tts_result = {
            'text_to_synthesize': text_to_synthesize,
            'audio_path': audio_path
        }

        return redirect(url_for('text_to_speech'))

    return render_template('tts.html', tts_result=st.session_state.tts_result)

if __name__ == '__main__':
    app.run(debug=True)
