#여기 있는 파일은 전 부 문장생성에 필요한 함수와 변수를 정의한 파일
#특히나 경로설정을 Path사용해서 절대경로로 사용함 이유는 장고내에서는 기존의 파이썬 경로가 적용이 안돼서
#총 3개의 함수가 정의되어 있고 extract_features함수에서 반환된 값을
#문장 생성에 필요한 generate_desc 함수에 인풋으로 사용함

from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


MODEL_DIR = Path(__file__).resolve().parent


max_length = 32
tokenizer = load(open(Path(__file__).resolve().parent / "tokenizer.p", "rb"))
model = load_model(Path(__file__).resolve().parent /'models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")


# def extract_features(img, model):
def extract_features(filename, model):
    try:
        print('-------------')
        print(filename)
        print('-------------')
        photo = Image.open(filename)
    except Exception:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    finally:
        print(photo)
        image = photo.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text
