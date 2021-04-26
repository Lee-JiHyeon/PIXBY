from django.db import models
from django.conf import settings
from accounts.models import Kid
# Create your models here.

class Silhouette_book(models.Model):
    kid = models.ForeignKey(Kid, on_delete=models.CASCADE)
    book_name = models.CharField(max_length=50)
    pic_dir = models.CharField(max_length=100)

    def __str__(self):
        return self.book_name


class Silhouette_character(models.Model):
    kid = models.ForeignKey(Kid, on_delete=models.CASCADE)
    silhouette_book = models.ForeignKey(Silhouette_book, on_delete=models.CASCADE)
    character_eng_name = models.CharField(max_length=50)
    character_kor_name = models.CharField(max_length=50)
    character_jpn_name = models.CharField(max_length=50)
    character_chn_name = models.CharField(max_length=50)
    open_pic_dir = models.CharField(max_length=128)
    close_pic_dir = models.CharField(max_length=128)
    checked = models.BooleanField(default=False)

    def __str__(self):
        return self.character_eng_name