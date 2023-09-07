import torch
import hydra
from pipelines.pipeline import InferencePipeline
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import openai
import os
import jellyfish
from google.cloud import speech
import datetime
from num2words import num2words
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video():
    # Read the uploaded video file from the request
    video_file = request.files['video_file']

    if video_file:
        # Save the uploaded video file temporarily
        video_filename = 'uploaded_video.mp4'
        video_file.save(video_filename)

        # Read other input parameters from the request JSON
        data = request.form.to_dict()
        config_filename = './configs/CMUMOSEAS_V_ES_WER44.5.ini'
        landmarks_filename = data.get('landmarks_filename', None) 
        gpu_idx = int(data.get('gpu_idx', -1)) 
        detector = data.get('detector', 'mediapipe')  
        audioFilename = "audio.mp3" 
        extract_audio_from_video(video_filename, audioFilename)
        ref = speech_to_text_openai(audioFilename)
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() and gpu_idx >= 0 else "cpu")
        tiempo = datetime.datetime.now()
        hyp = InferencePipeline(config_filename, device=device, detector=detector, face_track=True)(video_filename, landmarks_filename)
        tiempo2 = datetime.datetime.now()
        tiempoFinal=tiempo2 - tiempo;
        lever = levenshtein_distance(hyp, ref)
        cos = cosine_similarity(hyp, ref)
        seq = SequenceMatcher_py(hyp, ref)
        jel2, jel3, jel5 = jellyfish_distance(hyp, ref)
        isAltered = False
        alarms = 0
        if lever < 30:
            alarms += 1
        if cos < 0.1:
            alarms += 1
        if seq < 0.3:
            alarms += 1
        if jel2 >= 15:
            alarms += 1
        if jel3 < 0.5:
            alarms += 1
        if jel5 >= 30:
            alarms += 1
        if(alarms>3):
            isAltered = True
        # Prepare the results
        results = {
            "veredict": "Altered video" if isAltered else "No alterations",
            "results":{
                "levenshtein_distance":lever,
                "cosine_similarity":cos,
                "SequenceMatcher_python":seq,
                "jellyfish_distance_damerau_levenshtein_distance":jel2,
                "jellyfish_distance_jaro_distance":jel3,
                "jellyfish_distance_hamming_distance":jel5,
            },
            "message": "Video processing completed.",
            "total_time":str(tiempoFinal),
            "original_phrase":ref,
            "lip_reading":hyp
        }

        # Clean up by removing the temporarily saved video file
        os.remove(video_filename)
        os.remove(audioFilename)

        return jsonify(results)
    else:
        return jsonify({"error": "No video file provided in the request."}), 400

def speech_to_text_google(filename):
    client=speech.SpeechClient.from_service_account_file("key.json")
    with open(filename, "rb") as audio_file:
        content=audio_file.read()
    audio=speech.RecognitionAudio(content=content)
    config=speech.RecognitionConfig(
        #sample_rate_hertz=44100,
        enable_automatic_punctuation=True,  
        language_code="es-ES",
    )
    response=client.recognize(config=config, audio=audio)
    print(response,"hola")
    number=""
    for result in response.results:
        number=result.alternatives[0].transcript
    
    number=number_to_text(number)
    return number

def speech_to_text_openai(filename):
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    
    transcribed_text = transcript['text']
    transcribed_text_with_numbers_as_text = number_to_text(transcribed_text)
    return transcribed_text_with_numbers_as_text

def extract_audio_from_video(video_file, output_audio_file):
    video_clip = VideoFileClip(video_file)
    audio = video_clip.audio
    audio.write_audiofile(output_audio_file)

def number_to_text(number_str):
    number_text = []

    # Mapping of digits to their text representation in Spanish
    digit_mapping = {
        "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
        "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve"
    }

    # Split the input string by spaces to preserve spaces
    parts = number_str.split()

    for part in parts:
        # Process each part
        if part.isdigit():
            # If it's a digit, convert it to text
            text_representation = ''.join([digit_mapping[digit] for digit in part])
            number_text.append(text_representation)
        else:
            # If it's not a digit, keep it as is
            number_text.append(part)

    # Join the converted parts with spaces to form the complete textual representation
    text = ' '.join(number_text)

    return text


def levenshtein_distance(hyp, ref):
    distance = Levenshtein.distance(hyp, ref)
    similarity_percentage = 100 * (1 - distance / max(len(hyp), len(ref)))
    return similarity_percentage

def cosine_similarity(hyp, ref):
    vectorizer = CountVectorizer()
    vectorizer.fit([hyp, ref])
    vector = vectorizer.transform([hyp, ref]).toarray()
    similarity = cosine_similarity_sklearn(vector)
    return similarity[0][1]


def SequenceMatcher_py(a, b):
    return SequenceMatcher(None, a, b).ratio()

def jellyfish_distance(hyp, ref):
    jel2 = jellyfish.damerau_levenshtein_distance(hyp, ref)
    jel3 = jellyfish.jaro_distance(hyp, ref)
    jel5 = jellyfish.hamming_distance(hyp, ref)
    return jel2, jel3, jel5

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)