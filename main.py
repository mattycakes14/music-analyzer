from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import tempfile
import librosa
import numpy as np

app = FastAPI()

@app.get("/health")
def read_root():
    return {"Status": "OK"}

@app.post("/analyze-bpm-librosa")
async def analyze_bpm(file: UploadFile = File(...)):
    
    # Create a temporary file to save the uploaded audio
    # librosa needs a file path to load effectively
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Load the audio file
        # y is raw audio waveform, sr is the sampling rate
        y, sr = librosa.load(temp_path)

        # 1. Separate Harmonic and Percussive components
        # This isolates drums/transients from melody/vocals
        # This is often more effective than raw low-pass filtering for beat tracking
        _, y_percussive = librosa.effects.hpss(y)

        # 2. Run beat tracker on the PERCUSSIVE component
        # We also loosen the 'tightness' to allow it to deviate from 120 more easily
        tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, tightness=100)

        # Handle return type (librosa returns a numpy array or scalar depending on version)
        bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo

        return {
            "filename": file.filename,
            "bpm": round(float(bpm), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/analyze-bpm-essentia")
async def analyze_bpm_essentia(file: UploadFile = File(...)):
    return {"Status": "OK"}