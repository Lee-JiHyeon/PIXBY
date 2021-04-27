from django.urls import path
from . import views


app_name = 'quizs'
urlpatterns = [

    path('', views.test, name='test'),
    path('showQuiz/', views.showQuiz, name='showQuiz'),
    path('dragQuiz/', views.dragQuiz, name='dragQuiz'),
    path('speakQuiz/', views.speakQuiz, name='speakQuiz'),
    path('quizResult/', views.quizResult, name='quizResult'),

]