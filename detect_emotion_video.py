import cv2
import numpy as np
import imutils
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import os
import math
import sys
from threading import Timer
import shutil
import time
import keras
import pyaudio
import numpy as np
import librosa
import wave
import pandas as pd
from keras.models import model_from_json

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def take_mic_input(interval=5):
    """
    Take audio input from the microphone for `interval` seconds and extract MFCC features.

    Parameters:
        interval (int, optional): Recording duration in seconds. Default is 5.

    Returns:
        np.ndarray: Extracted MFCC features.
    """
    audio = pyaudio.PyAudio()
    print("mic on")

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

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

    X, sample_rate = librosa.load('temp.wav', res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

    return mfccs


def is_consistent_loudness(audio_mfccs: np.ndarray) -> bool:
    """
    Check if audio loudness is consistent.

    Parameters:
        audio_mfccs (np.ndarray): Audio MFCC features.

    Returns:
        bool: True if loudness is consistent, False otherwise.
    """
    frame_energy = np.sqrt(np.mean(np.square(audio_mfccs), axis=0))
    loudness_threshold = 0.1
    energy_std = np.std(frame_energy)
    is_consistent_loudness = energy_std < (loudness_threshold * np.mean(frame_energy))
    return is_consistent_loudness


def is_consistent_pitch(audio_pitch: np.ndarray) -> bool:
    """
    Check if audio pitch is consistent.

    Parameters:
        audio_pitch (np.ndarray): Audio pitch data.

    Returns:
        bool: True if pitch is consistent, False otherwise.
    """
    pitch_variation = np.std(audio_pitch)
    pitch_threshold = 10.0
    return pitch_variation < pitch_threshold


def load_and_compile_voice_model() -> keras.models.Model:
    """
    Load and compile a pre-trained emotion detection model.

    Returns:
        keras.models.Model: The loaded and compiled model.
    """
    print("loading voice model")
    opt = keras.optimizers.RMSprop(learning_rate=0.00001)

    with open(r'C:\Users\Acer\codes\SpeechMaster\face\saved_models\model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(r"C:\Users\Acer\codes\SpeechMaster\face\saved_models\Emotion_Voice_Detection_Model.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return loaded_model


def predict_emotion_from_audio(audio_data: np.ndarray, loaded_model: keras.models.Model) -> str:
    """
    Predict emotion from audio data using a loaded and compiled model.

    Args:
        audio_data (np.ndarray): Audio data for prediction.
        loaded_model (keras.models.Model): Loaded and compiled emotion detection model.

    Returns:
        str: The predicted emotion label.
    """
    labels = [
        "angry",
        "calm",
        "fearful",
        "happy",
        "sad",
        "angry",
        "calm",
        "fearful",
        "happy",
        "sad"
    ]

    print("Making inference")

    df = pd.DataFrame(data=audio_data)
    df = df.stack().to_frame().T

    twodim = np.expand_dims(df, axis=2)

    pred = loaded_model.predict(twodim, batch_size=32, verbose=1)
    pred = pred.argmax(axis=1)
    pred = pred.astype(int).flatten()[0]

    return labels[pred]


def detect_and_predict_face(frame, faceNet, face_emotion_model, threshold):
    global detections
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            locs.append((startX, startY, endX, endY))
            preds.append(face_emotion_model.predict(face)[0].tolist())

    return (locs, preds)

audio_model = load_and_compile_voice_model()

MASK_MODEL_PATH = r'face\model\emotion_model.h5'
FACE_MODEL_PATH = r'face\face_detector'
THRESHOLD = 0.5

print("[INFO] loading face detector model...")
prototxtPath = r'face/face_detector/deploy.prototxt'
weightsPath = r'face\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading emotion detector model...")
face_emotion_model = load_model(MASK_MODEL_PATH)

print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

labels = ["happy", "neutral", "sad"]

counter = 0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    (locs, preds) = detect_and_predict_face(frame, faceNet, face_emotion_model, THRESHOLD)
    if counter % 20 == 0:
        audio = take_mic_input()
        audio_emotion = predict_emotion_from_audio(audio, audio_model)
        pitch = is_consistent_pitch(audio)
        loudness = is_consistent_loudness(audio)
        print("is consistent pitch", pitch)
        print("is consistent loudness", loudness)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        label = str(labels[np.argmax(pred)])
        if label == "happy":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 50), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 200, 50), 2)
            cv2.putText(original_frame, f"Voice: {audio_emotion}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(original_frame, f"Is consistent pitch: {pitch}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(original_frame, f"Is consistent loudness: {loudness}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        elif label == "neutral":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            cv2.putText(original_frame, f"Voice: {audio_emotion}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(original_frame, f"Is consistent pitch: {pitch}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(original_frame, f"Is consistent loudness: {loudness}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        elif label == "sad":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 50, 200), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 50, 200), 2)
            cv2.putText(original_frame, f"Voice: {audio_emotion}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(original_frame, f"Is consistent pitch: {pitch}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(original_frame, f"Is consistent loudness: {loudness}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)


    frame = cv2.resize(original_frame, (860, 490))
    cv2.imshow("Facial Expression", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    time.sleep(0.25)
    counter += 1

cv2.destroyAllWindows()
vs.stop()
