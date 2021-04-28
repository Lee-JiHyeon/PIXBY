from django.shortcuts import render, redirect, get_object_or_404
from .dataList import books_dict, character_dict
from .models import Silhouette_book, Silhouette_character
from .serializers import SilhouetteBookSerializer, SilhouetteCharacterSerializer
from accounts.models import Kid
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
import json
import ttos
# from rest_framework.permissions import IsAuthenticated
# Create your views here.

@api_view(['GET','POST'])
def test():
    data = json.loads(request.body.decode(encoding='utf-8'))
    serializer = ReviewCreateSerializer(data=data['form_data'])
    Response(serializer.data, status.HTTP_201_CREATED)
    pass


@api_view(['POST'])
def createBooks(request):
    # 기존의 배경 경로를 집어넣는다.
    data = json.loads(request.body.decode(encoding='utf-8'))

    kid = get_object_or_404(Kid, pk=data['kidpk'])

    for books in books_dict:
        silhouette_book = Silhouette_book()
        silhouette_book.kid = kid
        silhouette_book.book_name = books
        silhouette_book.pic_dir = books_dict[books]
        silhouette_book.save()

    return Response({"실루엣북 생성" : True})


@api_view(['POST'])
def createCharacters(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    for characters in character_dict:

        book = get_object_or_404(Silhouette_book, book_name=characters, kid=kid)
        temps = character_dict[characters]
        for temp in temps:
            char = Silhouette_character()
            char.kid = kid
            char.silhouette_book = book
            char.character_eng_name = temp
            char.character_kor_name = temps[temp][0]
            char.character_jpn_name = temps[temp][1]
            char.character_chn_name = temps[temp][2]
            char.open_pic_dir = temps[temp][3]
            char.close_pic_dir = temps[temp][4]
            char.save()
    # 이런식으로 짜면 될듯
    return Response({"캐릭터 생성" : True})


@api_view(['POST'])
def findCharacters(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    # kid = get_object_or_404(Kid, pk=data['kidpk'])
    character = get_object_or_404(Silhouette_character, character_eng_name=data['character_eng_name'], kid=data['kidpk'])   
    character.checked = True
    character.save()

    serializer = SilhouetteCharacterSerializer(character)
    return Response(serializer.data)


@api_view(['POST'])
def characterList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    character_list = kid.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(character_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def transportList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    transport = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    transport_list = transport.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(transport_list, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def roadList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    road = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    road_list = road.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(road_list, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def livingroomList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    livingroom = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    livingroom_list = livingroom.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(livingroom_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def animalList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    animal = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    animal_list = animal.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(animal_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def bathList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    bath = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    bath_list = bath.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(bath_list, many=True)
    return Response(serializer.data)



@api_view(['POST'])
def kitchenList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    kitchen = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    kitchen_list = kitchen.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(kitchen_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def foodList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    food = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    food_list = food.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(food_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def myroomList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    myroom = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    myroom_list = myroom.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(myroom_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def skiList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    ski = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    ski_list = ski.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(ski_list, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def parkList(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    park = get_object_or_404(Silhouette_book, kid=kid, book_name=data['bookname'])
    park_list = park.silhouette_character_set.all()
    serializer = SilhouetteCharacterSerializer(park_list, many=True)
    return Response(serializer.data)


language_list = {
    'character_eng_name' : 'en-US',
    'character_kor_name' : 'ko-KR',
    'character_jpn_name' : 'ja-JP',
    'character_chn_name' : 'zh-CN',
}

@api_view(['POST'])
def wordReading(request):
    data = json.loads(request.body.decode(encoding='utf-8'))  # { 키 language 밸류 언어선택, 키 word 밸류 word name }
    set_language = data['language']
    words=data['word']

    language=language_list[set_language]
    context = ttos.TextToSpeech(words, language)
    return Response(context)


@api_view(['POST'])
def characterRating(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])

    total = 81
    all_list = kid.silhouette_character_set.all()
    cnt = 0
    for i in all_list:
        if i.checked == True:
            cnt += 1
    rate = (cnt*100) // total

    context = {
        "rate" : rate,
    }
    return Response(context)