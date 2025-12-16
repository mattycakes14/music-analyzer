from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import tempfile
import librosa
import numpy as np
import scipy.ndimage

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

@app.post("/analyze-structure")
async def analyze_structure(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Load audio (downsample to 22050 for speed)
        y, sr = librosa.load(temp_path, sr=22050)

        # 1. Calculate RMS Energy (Loudness) and Spectral Contrast (Bass vs Treble)
        # Hop length of 512 gives us roughly 43 analysis frames per second
        hop_length = 512
        
        # Calculate Root Mean Square energy (overall volume)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Calculate Spectral Centroid (brightness - helps detect drops)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Calculate Low-Frequency Energy (Bass)
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        fft_freqs = librosa.fft_frequencies(sr=sr)
        # Sum energy below 200Hz
        bass_mask = fft_freqs < 200
        bass_energy = np.sum(S[bass_mask, :], axis=0)

        # Smooth the signals to remove jitter (like short drum hits)
        # We use a median filter over ~1 second (43 frames)
        smoothing_window = 43 
        bass_smooth = scipy.ndimage.median_filter(bass_energy, size=smoothing_window)
        rms_smooth = scipy.ndimage.median_filter(rms, size=smoothing_window)
        
        # Normalize signals to 0-1 range for easier thresholding
        bass_norm = librosa.util.normalize(bass_smooth)
        rms_norm = librosa.util.normalize(rms_smooth)

        # 2. Detect "Drop" and "Breakdown" events
        # Logic: 
        # - High Bass + High RMS = Drop / Main Chorus
        # - Low Bass + Medium/High RMS = Intro / Breakdown / Build-up
        
        # Convert frames to time
        times = librosa.frames_to_time(np.arange(len(bass_norm)), sr=sr, hop_length=hop_length)
        
        # Create segments (simplified for response)
        # We classify every 1 second chunk
        duration = librosa.get_duration(y=y, sr=sr)
        segments = []
        
        # Sample every 5 seconds to reduce output size
        sample_rate_sec = 5
        for t in range(0, int(duration), sample_rate_sec):
            # Find frame index for this time
            idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
            if idx >= len(bass_norm): break
            
            b_val = bass_norm[idx]
            r_val = rms_norm[idx]
            
            label = "Unknown"
            if b_val > 0.6:
                label = "High Energy (Drop/Chorus)"
            elif b_val < 0.3 and r_val > 0.3:
                label = "Breakdown/Build"
            elif b_val < 0.2 and r_val < 0.2:
                label = "Quiet/Intro/Outro"
            else:
                label = "Verse/Mid Energy"
                
            segments.append({
                "time": t,
                "label": label,
                "bass_level": round(float(b_val), 2),
                "energy_level": round(float(r_val), 2)
            })

        # 3. Find structural boundaries (Mix Points)
        # Look for sudden large changes in bass energy
        # Calculate derivative (rate of change)
        bass_diff = np.diff(bass_norm)
        # Find peaks in the derivative (sudden changes)
        change_points_frames = np.where(np.abs(bass_diff) > 0.3)[0] # Threshold 0.3 implies 30% jump
        change_points_times = librosa.frames_to_time(change_points_frames, sr=sr, hop_length=hop_length)
        
        # Filter changes that are too close together (keep only one every 10s)
        filtered_changes = []
        last_change = -100
        for cp in change_points_times:
            if cp - last_change > 10:
                filtered_changes.append(round(float(cp), 2))
                last_change = cp

        return {
            "filename": file.filename,
            "duration_sec": round(duration, 2),
            "mix_points_bass_change": filtered_changes,
            "segments_5s_interval": segments
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
