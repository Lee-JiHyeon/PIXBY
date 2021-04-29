from rest_framework import serializers
from .models import Photo, Word

class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = ('__all__')


class WordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Word
        fields = ('__all__')