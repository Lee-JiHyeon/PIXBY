from django import forms
from django.apps import AppConfig
from .models import Kid


class AccountsConfig(AppConfig):
    name = 'accounts'



class KidCreationForm(forms.ModelForm):
    name = forms.CharField(
        max_length=20,
        label='Name'
        widget=forms.TextInput(
            attrs={
                'placeholder': '제목을 입력해주세요.',
            },
            required=True
        )
    )

    gender = forms.CharField(
        max_length=20,
        label='gender'
        widget=forms.ChoiceField(
            ('남자','여자'),
            required=True           
        )
    )


    class Meta:
        model = Kid
        fields = '__all__' 
