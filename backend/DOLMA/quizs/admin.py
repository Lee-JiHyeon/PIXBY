from django.contrib import admin
from .models import Final_quiz, Final_quiz_content, Daily_quiz

# Register your models here.

admin.site.register(Final_quiz)
admin.site.register(Final_quiz_content)
admin.site.register(Daily_quiz)