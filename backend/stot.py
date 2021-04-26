from google.cloud import speech
import io
import os


os.environ['GOOGLE_APPLICATION_CREDENTIALS']="C:/Users/hie69/Desktop/tts/mypjt/tts-and-stt-308605-8c165d27d258.json"
def SpeechToText(language):
    
    # audio json으로된것 decoding해서 mp3로 변환?
    # print(audio)

    client = speech.SpeechClient()
    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        '.',
        'output.mp3')

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language)
    

    response = client.recognize(config=config, audio=audio)
    print(response)
    # for result in response.results:
    #     print('Transcript: {}'.format(result.alternatives[0].transcript))
    return