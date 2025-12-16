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

@app.post("/analyze-key")
async def analyze_key(file: UploadFile = File(...)):
    # Create a temporary file to save the uploaded audio
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        y, sr = librosa.load(temp_path)

        # 1. Extract Chroma Features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # 2. Key Detection Logic (Krumhansl-Schmuckler)
        # Standard profiles for Major and Minor keys
        # C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        chroma_mean = np.mean(chroma, axis=1)
        
        key_correlations = []
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Calculate correlation for all 12 major and 12 minor keys
        for i in range(12):
            # Roll the profile to match the current root note
            major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
            minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
            key_correlations.append((major_corr, f"{pitches[i]} Major"))
            key_correlations.append((minor_corr, f"{pitches[i]} Minor"))

        # Sort by correlation coefficient (descending)
        key_correlations.sort(key=lambda x: x[0], reverse=True)
        best_key = key_correlations[0][1]

        # 3. Convert to Camelot Wheel
        # Standard Camelot Wheel Mapping
        camelot_mapping = {
            # B Major / Ab Minor (5 Sharps / 7 Flats) -> 1
            'B Major': '1B', 'G# Minor': '1A', 'Ab Minor': '1A',
            
            # F# Major / Eb Minor (6 Sharps / 6 Flats) -> 2
            'F# Major': '2B', 'Gb Major': '2B', 'D# Minor': '2A', 'Eb Minor': '2A',
            
            # Db Major / Bb Minor (5 Flats / 7 Sharps) -> 3
            'C# Major': '3B', 'Db Major': '3B', 'A# Minor': '3A', 'Bb Minor': '3A',
            
            # Ab Major / F Minor (4 Flats) -> 4
            'G# Major': '4B', 'Ab Major': '4B', 'F Minor': '4A',
            
            # Eb Major / C Minor (3 Flats) -> 5
            'D# Major': '5B', 'Eb Major': '5B', 'C Minor': '5A',
            
            # Bb Major / G Minor (2 Flats) -> 6
            'A# Major': '6B', 'Bb Major': '6B', 'G Minor': '6A',
            
            # F Major / D Minor (1 Flat) -> 7
            'F Major': '7B', 'D Minor': '7A',
            
            # C Major / A Minor (No Sharps/Flats) -> 8
            'C Major': '8B', 'A Minor': '8A',
            
            # G Major / E Minor (1 Sharp) -> 9
            'G Major': '9B', 'E Minor': '9A',
            
            # D Major / B Minor (2 Sharps) -> 10
            'D Major': '10B', 'B Minor': '10A',
            
            # A Major / F# Minor (3 Sharps) -> 11
            'A Major': '11B', 'F# Minor': '11A', 'Gb Minor': '11A',
            
            # E Major / C# Minor (4 Sharps) -> 12
            'E Major': '12B', 'C# Minor': '12A', 'Db Minor': '12A'
        }
        
        camelot_code = camelot_mapping.get(best_key, "Unknown")

        return {
            "filename": file.filename,
            "detected_key": best_key,
            "camelot_code": camelot_code
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
