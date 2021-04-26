from django.db import models
from django.contrib.auth.models import AbstractUser


# # Create your models here.
class User(AbstractUser):

    def __str__(self):
        return self.username
    
    pass


class EnvSettings(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    effect_music = models.BooleanField(default=True)
    background_music = models.BooleanField(default=True)

    def __str__(self):
        return self.user.username


class Kid(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=20)
    gender = models.IntegerField() # 1이면 남자 0면 여자
    language = models.CharField(max_length=50, default='character_kor_name') # 언어파트  ' 0 영어 1 한국어 2 일본어 3 중국어 '
    collection_open = models.BooleanField(default=False)
    quiz_difficulty = models.IntegerField(default=0)
    graduate = models.BooleanField(default=False)
    allow_data = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class StudyRecord(models.Model):
    kid = models.ForeignKey(Kid, on_delete=models.CASCADE)
    start_time = models.TimeField()
    end_time = models.TimeField()
    date = models.DateField()
    total_time = models.TimeField()

    def __str__(self):
        return self.kid