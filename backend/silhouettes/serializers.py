from rest_framework.serializers import ModelSerializer
from .models import Silhouette_book, Silhouette_character


class SilhouetteBookSerializer(ModelSerializer):

    class Meta:
        model = Silhouette_book
        fields = ('book_name', 'pic_dir')



class SilhouetteCharacterSerializer(ModelSerializer):

    class Meta:
        model = Silhouette_character
        fields = ('character_eng_name', 'character_kor_name','character_jpn_name','character_chn_name', 'open_pic_dir', 'close_pic_dir', 'checked')