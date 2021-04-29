from django.urls import path
from . import views



app_name = 'silhouettes'
urlpatterns = [

    path('', views.test, name='test'),
    path('createBooks/', views.createBooks, name='createBooks'),
    path('createtransports/', views.createCharacters, name='createCharacters'),
    path('findCharacters/', views.findCharacters, name='findCharacters'),
    path('characterList/', views.characterList, name='characterList'),
    path('transportList/', views.transportList, name='transportList'),
    path('roadList/', views.roadList, name='roadList'),
    path('livingroomList/', views.livingroomList, name='livingroomList'),
    path('animalList/', views.animalList, name='animalList'),
    path('bathList/', views.bathList, name='bathList'),
    path('kitchenList/', views.kitchenList, name='kitchenList'),
    path('foodList/', views.foodList, name='foodList'),
    path('myroomList/', views.myroomList, name='myroomList'),
    path('skiList/', views.skiList, name='skiList'),
    path('parkList/', views.parkList, name='parkList'),
    path('wordReading/', views.wordReading, name='wordReading'),
    path('characterRating/', views.characterRating, name='characterRating'),

]