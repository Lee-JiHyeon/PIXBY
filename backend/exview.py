from django.shortcuts import render, redirect
from google.cloud import texttospeech
from google.cloud import storage
import ttos
import stot
import os

# Create your views here.
def ttsTest(request):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']="C:/Users/hie69/Desktop/tts/mypjt/tts-and-stt-308605-8c165d27d258.json"
    print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    # storage_client = storage.Client()

    # # Make an authenticated API request
    # buckets = list(storage_client.list_buckets())
    # print(buckets)
    return render(request, 'texttospeech/index.html')


def tts(request):
    # print("hello")
    # client = texttospeech.TextToSpeechClient()

    # # request에서 받은 문자 넣기
    # request_text = "Dog"
    # synthesis_input = texttospeech.SynthesisInput(text=request_text)

    # # 설정에서 정한 언어로 바꾸기
    # selected_language="en-US"
    # voice = texttospeech.VoiceSelectionParams(
    #     language_code=selected_language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    # )
    # audio_config = texttospeech.AudioConfig(
    #     audio_encoding=texttospeech.AudioEncoding.MP3
    # )
    # response = client.synthesize_speech(
    #     input=synthesis_input, voice=voice, audio_config=audio_config
    # )

    # with open("output.mp3", "wb") as out:
    #     out.write(response.audio_content)
    print('ttos')
    words="cat mom"
    language="en-US"
    audio = ttos.TextToSpeech(words, language)
    print(audio)
    return redirect('texttospeech:ttsTest')

def stt(request):
    language="en-US"
    # stot(audio, language)
    stot.SpeechToText(language)
    return redirect('texttospeech:ttsTest')