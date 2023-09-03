from predict import *
from utils import *

audio = take_mic_input()
print(is_consistent_pitch(audio))
print(is_consistent_loudness(audio))
model = load_and_compile_voice_model()
print(predict_emotion_from_audio(audio, model))
