from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view

from .models import EnvSettings, Kid, StudyRecord
from .serializers import (CustomUserSerializer, KidSerializer, EnvSettingsSerializer,
                        GetLanguageSerializer, )
from silhouettes.models import Silhouette_book, Silhouette_character
from django.contrib.auth import get_user_model
import json


from silhouettes.dataList import books_dict, character_dict

# Create your views here.



@api_view(['POST'])
def signup(request):
    serializer = CustomUserSerializer(data=request.data)
    if get_user_model().objects.filter(username=request.data.get('username')).exists():
        return Response({'fail': '이미 아이디가 존재합니다.'})
    
    if request.data.get('password') != request.data.get('passwordvalidation'):
        error_message = {'fail': '비밀번호가 맞지않습니다.'}
        return Response(data=error_message)

    if serializer.is_valid(raise_exception=True):
        user = serializer.save()
        user.set_password(request.data.get('password'))
        user.save()
        env = EnvSettings()
        env.user = user
        env.save()
        return Response(data=serializer.data)


@csrf_exempt
def login(request):
    # 프론트딴에서 로직 짜야할듯
    # 토큰은 줄수있음.
    pass


def logout(request):
    # 프론트딴에서 작업
    pass


@api_view(['POST'])
def effectMusicChange(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    user = get_object_or_404(get_user_model(), username=data['username'])
    env = get_object_or_404(EnvSettings, user=user)
    if env.effect_music == True:
        env.effect_music = False
    elif env.effect_music == False:
        env.effect_music = True
    env.save() 

    return Response({'효과음' : env.effect_music})


@api_view(['POST'])
def backgroundMusicChange(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    user = get_object_or_404(get_user_model(), username=data['username'])
    env = get_object_or_404(EnvSettings, user=user)
    if env.background_music == True:
        env.background_music = False
    elif env.background_music == False:
        env.background_music = True
    env.save()
    
    return Response({'배경음' : env.background_music})


@api_view(['POST'])
def allowData(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])
    character_list = kid.silhouette_character_set.all()
    for character in character_list:
        character.checked = True
        character.save()
    # kid.
    return Response({'데이터 잠금 해제':True})


@api_view(['POST'])
def createKid(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    user = get_object_or_404(get_user_model(), username=data['username'])
    data['user'] = user.pk    
    serializer = KidSerializer(data=data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()

        kid = get_object_or_404(Kid, pk=serializer.data['pk'])
        for books in books_dict:
            silhouette_book = Silhouette_book()
            silhouette_book.kid = kid
            silhouette_book.book_name = books
            silhouette_book.pic_dir = books_dict[books]
            silhouette_book.save()   

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

        return Response(serializer.data) 


@api_view(['POST'])
def getKid(request):
    
    data = json.loads(request.body.decode(encoding='utf-8'))
    user = get_object_or_404(get_user_model(), username=data['username'])
    kid_list = user.kid_set.all()
    serializer = KidSerializer(kid_list, many=True)
    return Response(serializer.data)

    # if request.method == 'POST':
    #     # data = json.loads(request.body.decode(encoding='utf-8'))
    #     serializer = KidSerializer(data=request.data)   
    #     if serializer.is_valid(raise_exception=True):
               
    #         return Response(serializer.data)


@api_view(['POST'])
def changelanguage(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])
    kid.language = data['language']
    kid.save()
    # kid.
    return Response({'언어설정': data['language']})



@api_view(['POST'])
def getsettings(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    user = get_object_or_404(get_user_model(), username=data['username'])
    env = get_object_or_404(EnvSettings, user=user)
    serializer = EnvSettingsSerializer(env)
    return Response(serializer.data)


@api_view(['POST'])
def getlanguage(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])
    serializer = GetLanguageSerializer(kid)
    return Response(serializer.data)


# 공부시간 기록은 일단 보류