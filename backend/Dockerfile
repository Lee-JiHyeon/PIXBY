FROM python:3.8.8

ENV PYTHONUNBUFFERED 0
RUN apt-get -y update 
RUN apt-get install -y libgl1-mesa-glx
RUN pip3 install django

# 해당 디렉토리 만들고
WORKDIR /usr/src/app
# 다 복사하고 
COPY . .
# DOLMA 볼더 안에 들어가서 
WORKDIR ./DOLMA

# RUN chmod 755 start
# 필수 패키지 설치하고
RUN pip install --upgrade pip 
RUN pip install -r ai-requirements.txt 
# 실행
EXPOSE 8000
# ENTRYPOINT ["./start"]
# RUN ["python", "manage.py", "migrate"]
CMD ["python3", "manage.py", "runserver", "0:8000"]