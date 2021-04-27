from google.cloud import texttospeech
import os
import base64

def Text_To_Speech(words, language):
    client = texttospeech.TextToSpeechClient()

    # request에서 받은 문자 넣기
    synthesis_input = texttospeech.SynthesisInput(text=words)

    # 설정에서 정한 언어로 바꾸기
    selected_language=language
    voice = texttospeech.VoiceSelectionParams(
        language_code=selected_language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open("output.mp3", "wb") as out:
        out.write(response.audio_content) 