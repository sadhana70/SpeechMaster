import pyaudio
import numpy as np
import librosa
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def take_mic_input(interval=5):
    '''Take audio input from the microphone for `interval` seconds and extract MFCC features.'''
    audio = pyaudio.PyAudio()
    print("mic on")

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * interval)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    print("mic off")
    
    # Save the recorded audio to a WAV file
    with wave.open('temp.wav', "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
   
    X, sample_rate = librosa.load('temp.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)

    return mfccs


def is_consistent_loudness(audio_mfccs: np.ndarray) -> bool: 
    frame_energy = np.sqrt(np.mean(np.square(audio_mfccs), axis=0))
    loudness_threshold = 0.1
    energy_std = np.std(frame_energy)
    is_consistent_loudness = energy_std < (loudness_threshold * np.mean(frame_energy))
    return is_consistent_loudness


def is_consistent_pitch(audio_pitch: np.ndarray) -> bool:
    pitch_variation = np.std(audio_pitch)
    pitch_threshold = 10.0
    return pitch_variation < pitch_threshold

