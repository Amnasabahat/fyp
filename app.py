import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model
try:
    model = load_model('model_best.hdf5')  # Replace with your actual path
    st.write("Model loaded successfully.")
except Exception as e:
    st.write("Error loading model:", str(e))

# Title for the web app
st.title('Cat Sound Classifier')

# Sidebar for user input
st.sidebar.header("Settings")

# Upload audio file
audio_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Function to extract features from the audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return np.expand_dims(log_spectrogram, axis=0)

# Classify the uploaded audio file
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # Extract features
    features = extract_features(audio_file)
    
    # Make predictions using the loaded model
    predictions = model.predict(features)
    
    # Display the predictions
    st.write("Predictions:", predictions)
    
    # Assuming you have the label mapping for the classes
    labels = ['cat_meow', 'cat_purr', 'cat_hiss']  # Replace with your actual labels
    predicted_class = labels[np.argmax(predictions)]
    
    st.write(f"Predicted class: {predicted_class}")
    
    # Visualize the spectrogram
    st.write("Spectrogram of the audio:")
    st.image(log_spectrogram[0], use_column_width=True)

# Instructions for users
st.markdown("""
    Upload a cat sound file (.wav or .mp3), and the model will classify it.
    The classifier will predict whether the sound is a 'Meow', 'Purr', or 'Hiss'.
""")
