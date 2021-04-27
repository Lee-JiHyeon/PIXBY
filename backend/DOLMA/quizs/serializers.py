from rest_framework.serializers import ModelSerializer
from .models import Daily_quiz, Final_quiz, Final_quiz_content


class DailyQuizSerializer(ModelSerializer):

    class Meta:
        model = Daily_quiz
        fields = '__all__'
        read_only_fields = ['kid']


