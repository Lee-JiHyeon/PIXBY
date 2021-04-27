from django.shortcuts import render, redirect,get_object_or_404
from .models import Daily_quiz, Final_quiz, Final_quiz_content
from silhouettes.models import Silhouette_character
from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
import json
import random
from silhouettes.serializers import SilhouetteCharacterSerializer
from .serializers import DailyQuizSerializer
from silhouettes.models import Silhouette_character
from accounts.models import Kid

# Create your views here.
def test():
    return


@api_view(['GET','POST'])
def showQuiz(request):
    # kid = get_object_or_404(Kid, pk=kid_pk)
    if request.method == 'GET': 
        word_list = Silhouette_character.objects.all()
        randint = random.sample(range(0,80), 3)       
        character_list = []
        character_list.append(word_list[randint[0]])
        character_list.append(word_list[randint[1]])
        character_list.append(word_list[randint[2]])
        random.shuffle(character_list)
        character_list.append(word_list[randint[0]])
        serializer = SilhouetteCharacterSerializer(character_list, many=True)
        return Response(serializer.data)
    
    elif request.method == "POST":
        data = json.loads(request.body.decode(encoding='utf-8'))
        kidi = get_object_or_404(Kid, pk=data['kidpk'])
        dailyquizs = Daily_quiz()
        dailyquizs.kid = kidi
        dailyquizs.content = data['content']
        dailyquizs.test_type = data['testType']
        dailyquizs.try_time = data['testTime']
        dailyquizs.pic_dir = data['picDir']
        dailyquizs.save()
        return Response({"ans" : "저장완료"})
    


@api_view(['GET','POST'])
def dragQuiz(request):
    kid = get_object_or_404(Kid, pk=kid_pk)
    word_list = Silhouette_character.objects.filter(checked=True)
    if len(word_list) > 1:
        random_word = random.sample(word_list, 2)
        context = {
            'answerDir' : random_word[0].open_pic_dir,
            'answerName' : random_word[0].character_name,
            'fakeDir1' :  random_word[1].open_pic_dir,      
        }
        return render(request, '', context)
    else:
        pass
    pass


@api_view(['GET','POST'])
def speakQuiz(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])
    word_list = Silhouette_character.objects.filter(checked=True)
    if len(word_list) > 1:
        random_word = random.sample(word_list, 2)
        context = {
            'answerDir' : random_word[0].open_pic_dir,
            'answerName' : random_word[0].character_name,
            'fakeDir1' :  random_word[1].open_pic_dir,      
        }
        return render(request, '', context)
    else:
        pass
    pass



@api_view(['POST'])
def quizResult(request):
    data = json.loads(request.body.decode(encoding='utf-8'))
    kid = get_object_or_404(Kid, pk=data['kidpk'])
    quiz_list = Daily_quiz.objects.filter(kid=kid).order_by('-created_at')
    serializer = DailyQuizSerializer(quiz_list, many=True)
    return Response(serializer.data)