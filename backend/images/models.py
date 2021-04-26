from django.db import models
from django.conf import settings
from accounts.models import Kid
import uuid

class Photo(models.Model):
    captioning = models.TextField(default="해당사진에 대한 문장 생성에 실패하였습니다.")
    uuid = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False,
    )
    created_at = models.DateTimeField(auto_now_add=True) 
    photo = models.FileField()


class Word(models.Model):
    uuid = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False,
    )
    created_at = models.DateTimeField(auto_now_add=True) 
    photo = models.JSONField()

 