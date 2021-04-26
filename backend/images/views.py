from accounts.models import Kid
from django.shortcuts import get_object_or_404
from pathlib import Path
import matplotlib.pyplot as plt

from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import PhotoSerializer
from .serializers import WordSerializer
from rest_framework import status
from silhouettes.serializers import SilhouetteCharacterSerializer
from silhouettes.models import Silhouette_character

from .models import Word
from .models import Photo
# from django.core.files.storage import default_storage
from .captioning import (
    generate_desc,
    extract_features,
    xception_model,
    model,
    tokenizer,
    max_length,
)
from . import detect_simple
import base64

#프론트에서 넘어온 사진파일을 1차적으로 세이브시켜서 디비에 저장함
#저장된 이미지 정보에 경로값을 이용해서 문장학습의 인풋으로 사용함 
#프론트에서 사진을 바로 활용하기가 어려워 일단은 저장을 하고 저장된 이미지를 활용하는게 관건
#따라서 해당 요청에 대한 시간이 2~5초 정도 소요됨 문장에 대한 생성을 해야해서
@api_view(['POST', 'GET'])
def upload(request):
    if request.method == "POST":
        serializer = PhotoSerializer(data=request.data)
        # file = request.FILES["photo"]
        # file.name = "tmp.jpg"
        # # 폴더가 있으면 그냥 넘기고 없으면 생성하자
        # default_storage.save(file.name, file)

 
        if serializer.is_valid(raise_exception=True):
            # img = Path(__file__).resolve().parent.parent / 'media' / 'tmp' /'tmp.jpg' 
            # img = "https://ssafy-unma.s3.ap-northeast-2.amazonaws.com/ssafy2.jpg"
            # img = Path(__file__).resolve().parent.parent / 'media' / str(img) 
            obj = serializer.save()
            img = Path(__file__).resolve().parent.parent / 'media' / str(obj.photo)
            # detect_simple.main(img)
            user_img_features = extract_features(img, xception_model)
            description = generate_desc(model, tokenizer, user_img_features, max_length)
            print("\n\n")
            print(img)
            im = get_object_or_404(Photo, uuid = obj.uuid)
            im.captioning = description
            im.save()
            
            # print(im.photo)
            print(img)
            # print(serializer.data)
            context={
                'sentense': description,
                # 'imgUrl': img,
            }
            # return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(context, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 


def remove_backslash(text):
    text = fr"{text}"
    e_text = text.encode('unicode_escape')
    r_text = str(e_text).replace('\\\\','/')
    result = r_text[2:-1].replace('//','/')
    return result


def image_save(base64_string,kidId):
    # 바이트 타입으로 변경
    imgdata = bytes(base64_string, 'utf-8')

    # 바이트를 디코딩해서 이미지로 변환
    with open(f'media/input{kidId}.png', 'wb') as f:
        f.write(base64.decodebytes(imgdata))
    return


@api_view(['POST'])
def yolo(request):
     if request.method == "POST":
        photo_data = dict()
        photo_data['photo'] = request.data['photo']
        serializer = WordSerializer(data=photo_data)
        # kidId == string type인데 잘 찾음
        kidId = request.data['kid']
        if serializer.is_valid(raise_exception=True):
            obj = serializer.save()
            # 이미지 저장
            image_save(obj.photo, kidId)
            # 이미지 경로 저장
            img = str(Path(__file__).resolve().parent.parent / 'media' / f'input{kidId}.png')
            # 백슬래쉬를 슬래쉬로 변경
            changed_url = remove_backslash(img)
            # print(changed_url)
            # 캡셔닝
            context = detect_simple.main(changed_url, kidId)
            # 캡셔닝 결과값 경로 저장
            result_url = Path(__file__).resolve().parent.parent / 'media' / f'output{kidId}.png'
            img_url=str(result_url)
            # 캡셔닝 결과 이미지 인코딩
            with open(img_url, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            context['img'] = encoded_string
            
            
            kid = get_object_or_404(Kid, pk=kidId)
            wordlist = context['classlist']
            wordlist = list(set(wordlist))
            templist = []
            for word in wordlist:
                character = kid.silhouette_character_set.filter(character_eng_name=word)
                serializer2 = SilhouetteCharacterSerializer(character, many=True)
                templist.append(serializer2.data)
            # print(templist)
            
            context['templist']=templist

            for word in wordlist:
                character = get_object_or_404(Silhouette_character, kid=kid, character_eng_name=word)
                character.checked = True
                character.save()
                # print(character)
 
            # return Response(serializer.data, status=status.HTTP_201_CREATED)
        
            return Response(context, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
    




# def result(request, image_id):
#     img = Photo.objects.get(pk=image_id)
#     img = img.photo
#     img = Path(__file__).resolve().parent.parent / 'media' / str(img) 

#     user_img_features = extract_features(img, xception_model)
#     description = generate_desc(model, tokenizer, user_img_features, max_length)
#     print("\n\n")
#     print(description)

#     context = {
#         'caption' : description,
#         'img' : Photo.objects.get(pk=image_id),
#     }
#     return render(request, 'images/result.html', context)




# @api_view(['POST', 'DELETE'])
# def upload(request):
#     if request.method == 'POST':
#         serializer = PhotoSerializer(data = request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.status.HTTP_400_BAD_REQUEST)