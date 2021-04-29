from django.urls import path
from . import views

app_name="images"
urlpatterns = [
    path('', views.upload, name="upload"),
    path('yolo/', views.yolo, name="yolo"),
    # path('result/<int:image_id>/', views.result, name="result"),
]
