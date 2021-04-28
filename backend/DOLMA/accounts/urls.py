from django.urls import path
from . import views


app_name = 'accounts'

urlpatterns = [

    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name="logout"),
    path('effectMusic/', views.effectMusicChange, name="effectMusicChange"),
    path('backgroundMusic/', views.backgroundMusicChange, name="backgroundMusicChange"),
    path('createKid/', views.createKid, name="createKid"),
    path('getKid/', views.getKid, name="getKid"),
    path('allowData/', views.allowData, name="allowData"),
    path('changelanguage/', views.changelanguage, name="changelanguage"),
    path('getsettings/', views.getsettings, name="getsettings"),
    path('getlanguage/', views.getlanguage, name="getlanguage"),
]