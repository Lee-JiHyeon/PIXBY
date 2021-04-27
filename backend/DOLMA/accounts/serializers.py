from rest_framework.serializers import ModelSerializer
from .models import Kid, EnvSettings, StudyRecord
from django.contrib.auth import get_user_model

User = get_user_model()

class CustomUserSerializer(ModelSerializer):

    class Meta:
        model = User
        fields = ('pk', 'username', 'password', 'email')
        read_only_fields = ['pk']
        extra_kwargs = {
            'password' : {'write_only': True}
        }


class KidSerializer(ModelSerializer):

    class Meta:
        model = Kid
        fields = ('pk', 'user', 'name', 'gender')
        read_only_fields = ['pk']


class EnvSettingsSerializer(ModelSerializer):

    class Meta:
        model = EnvSettings
        fields = '__all__'
        read_only_fields = ['user']



class GetLanguageSerializer(ModelSerializer):

    class Meta:
        model = Kid
        fields = ['language']