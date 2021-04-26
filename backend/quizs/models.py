from django.db import models
from accounts.models import Kid
# Create your models here.

class Daily_quiz(models.Model):
    kid = models.ForeignKey(Kid, on_delete=models.CASCADE)
    content = models.CharField(max_length=50)
    test_type = models.CharField(max_length=100)
    try_time = models.IntegerField(default=1)
    pic_dir = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return self.content


class Final_quiz(models.Model):
    kid = models.ForeignKey(Kid, on_delete=models.CASCADE)
    quiz = models.CharField(max_length=200)
    time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.quiz


class Final_quiz_content(models.Model):
    final_quiz = models.ForeignKey(Final_quiz, on_delete=models.CASCADE)
    content = models.CharField(max_length=200)
    test_type = models.IntegerField()
    correct = models.BooleanField()

    def __str__(self):
        return self.content