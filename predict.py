import keras
from keras.models import model_from_json
import numpy as np
import pandas as pd

def load_and_compile_voice_model() -> keras.models.Model:
    """
    Load and compile a pre-trained emotion detection model.

    Returns:
        loaded_model (keras.models.Model): The loaded and compiled model.
    """
    print("loading voice model")
    opt = keras.optimizers.RMSprop(learning_rate=0.00001)

    with open('saved_models/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return loaded_model

def predict_emotion_from_audio(audio_data: np.ndarray, loaded_model:keras.models.Model) -> str:
    """
    Predict emotion from audio data using a loaded and compiled model.

    Args:
        audio_data (np.ndarray): Audio data for prediction.

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