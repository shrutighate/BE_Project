import os

import cv2
import numpy as np
import openai
import whisper
import youtube_dl
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import assemblyai as aai
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip

from textSummarization import summarize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi as ytt
from pytube import YouTube

from videoSummarization import extract_keyframes

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/text_summarization')
def text_summarization():
    return render_template('text_summarization.html')


@app.route('/video_summarization')
def video_summarization():
    return render_template('video_summarization.html')


@app.route('/v2v_summarization')
def v2v_summarization():
    return render_template('v2v_summarization.html')


@app.route('/text_summary', methods=['POST'])
def text_summary():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarize(text)
        return render_template('summary.html', summary=summary)
    return render_template('text_summarization.html', summary="no summary")


model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def extract_video_id(url: str):
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]

    return None


def summarizer(script):
    input_ids = tokenizer("summarize: " + script, return_tensors="pt", max_length=512, truncation=True).input_ids

    outputs = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)

    summary_text = tokenizer.decode(outputs[0])
    return (summary_text)


@app.route('/video_summary', methods=['GET', 'POST'])
def get_summary():
    if request.method == 'POST':
        url = request.form['video_link']
        video_id = extract_video_id(url)
        data = ytt.get_transcript(video_id, languages=['de', 'en'])

        scripts = []
        for text in data:
            for key, value in text.items():
                if (key == 'text'):
                    scripts.append(value)
        transcript = " ".join(scripts)
        summary = summarizer(transcript)
        summary11 = summarize(transcript)
        return render_template("summary1.html", summary=summary, summary11=summary11)
        # return (summary)
    else:
        return "ERROR"


def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:

        text = recognizer.recognize_bing(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return None


def download_video(video_url, output_path):
    ydl_opts = {'outtmpl': output_path, 'verbose': True}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Error opening video file"

    frames = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break


        frames.append(frame)

    cap.release()

    return frames, None


@app.route('/v2v_summary', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        video_url = request.form['video_url']


        try:
            yt = YouTube(video_url)
            stream = yt.streams.first()
            video_path = stream.download()
            video_filename = os.path.basename(video_path)
            output_video_path = 'uploads\summary1.mpeg'
            num_sentences = 3
            # summary_path = generate_summary(video_path, output_video_path, num_sentences)
            summary_path = process_video(video_path)
            # return jsonify({'summary_path': summary_path})
            return render_template('summary3.html', summary=summary_path)

        except Exception as e:
            return jsonify({'error': str(e)})


def generate_summary(video_path, output_video_path, num_sentences=3):
    video = cv2.VideoCapture("1.mp4")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshold = 20.

    # writer = cv2.VideoWriter('final.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
    writer = cv2.VideoWriter('final.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))
    ret, frame1 = video.read()
    prev_frame = frame1

    a = 0
    b = 0
    c = 0

    while True:
        ret, frame = video.read()
        if ret is True:
            if (((np.sum(np.absolute(frame - prev_frame)) / np.size(frame)) > threshold)):
                writer.write(frame)
                prev_frame = frame
                a += 1
            else:
                prev_frame = frame
                b += 1

            cv2.imshow('frame', frame)
            c += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Total frames: ", c)
    print("Unique frames: ", a)
    print("Common frames: ", b)
    video.release()
    writer.release()
    cv2.destroyAllWindows()



UPLOADS_DIR = 'uploads'
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)


@app.route('/audio_summarization')
def audio_summary():
    return render_template('audio_summarization.html')


@app.route('/audio_summary', methods=['POST'])
def summarize_audio():
    if 'audioFile' not in request.files:
        print("No file uploaded - request.files:", request.files)
        return "No file uploaded"

    audio_file = request.files['audioFile']


    if audio_file.filename == '':
        print("No selected file - audio_file.filename:", audio_file.filename)
        return "No selected file"

    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)
    print("File Path: ", audio_path)
    print("File Name: ", audio_file)

    transcriber = transcribe_audio(audio_path)
    summary = summarize(transcriber)
    return render_template("summary2.html", summary=summary)
    # return summary


def transcribe_audio(audio_path):
    model = whisper.load_model('base')
    result = model.transcribe(audio_path)
    return result["text"]


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
