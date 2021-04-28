"""
Django settings for unma project.

Generated by 'django-admin startproject' using Django 3.1.5.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

from pathlib import Path

#db를 불러오기위한 import
# import my_settings

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '-m4xe#w^@lx=velw@o1xc7ee7$gysor0-m+6uf5uuvnp@*ye7_'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'corsheaders',
    'rest_framework',

    # 추가한 앱들
    'accounts',
    'silhouettes',
    'quizs',

    'images',
    # 'imagekit',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'unma.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'unma.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# {
#     'default' : {
#         'ENGINE': 'django.db.backends.mysql',   
#         'NAME': 'django_unma',               
#         'USER': 'root',                         
#         'PASSWORD': 'ssafy',                
#         'HOST': 'localhost',                   
#         'PORT': '3306',                        
#     }
# }

# 기존 database링크
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }


# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'ko-kr'

TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

STATIC_URL = '/static/'

STATIC_ROOT = '/unma/assets/'

# CORS
CORS_ORIGIN_ALLOW_ALL = True
# CORS_ALLOWED_ORIGINS = [
#     'http://localhost:19000',
#     'http://127.0.0.1:19000',
# ]
# STATICFILES_DIRS = [
#     os.path.join(BASE_DIR, 'static'),
# ]

AUTH_USER_MODEL = 'accounts.User'



# 미디어 파일을 관리할 루트 media 디렉터리
# MEDIA_ROOT = BASE_DIR /'media'
# 각 media file에 대한 URL prefix
# media file (as a upload file)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# MEDIA_ROOT = os.path.join(BASE_DIR, 'media')


# JWT
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    )
}

from datetime import timedelta
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=1),
}

#AWS S3

# AWS_ACCESS_KEY_ID = '비밀~'
# AWS_SECRET_ACCESS_KEY = '비밀~'
# AWS_REGION = 'ap-northeast-2'
# AWS_STORAGE_BUCKET_NAME = '비밀~'
# AWS_S3_CUSTOM_DOMAIN = '%s.s3.%s.amazonaws.com' % (AWS_STORAGE_BUCKET_NAME,AWS_REGION)

# DATA_UPLOAD_MAX_MEMORY_SIZE = 1024000000 # value in bytes 1GB here
# FILE_UPLOAD_MAX_MEMORY_SIZE = 1024000000

# DEFAULT_FILE_STORAGE = 'unma.storages.S3DefaultStorage'
# STATICFILES_STORAGE = 'unma.storages.S3StaticStorage'