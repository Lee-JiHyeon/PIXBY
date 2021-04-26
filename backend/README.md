## 사진인식 인공지능

1. yolo v4 (사물인식 API)

   You Only Look Once의 약어로 사물의 위치와 해당 사물의 class정보를 반환하는 API입니다. 이를 활용하여 유저가 업로드한 사진에서 사전에 정의한 테마에 맞는 사물들을 인식해 컬랙션 북에 추가합니다. 

   yolo는 tensorflow프로임워크와 openCV, darknet 모델을 활용하여 이미지 정보를 추출합니다. 

   현재는 mscoco에서 제공해주는 클래스들을 활용하여 사물을 분류하며 openCV를 활용하여 해당 사물의 주변에 바운딩박스를 그려서 사물의 위치정보역시 표현해 줍니다.

   

   *yolo 참조 깃허브

   [hunglc007/tensorflow-yolov4-tflite: YOLOv4, YOLOv4-tiny, YOLOv3, YOLOv3-tiny Implemented in Tensorflow 2.0, Android. Convert YOLO v4 .weights tensorflow, tensorrt and tflite (github.com)](https://github.com/hunglc007/tensorflow-yolov4-tflite)

   *yolo 공식 홈페이지

   [YOLO: Real-Time Object Detection (pjreddie.com)](https://pjreddie.com/darknet/yolo/)

​	



1. CNN + LSTM (이미지의 특징을 기반으로 문장 생성) 

   Flicker8k_Dataset 데이터를 활용하여 이미지에 대한 캡셔닝 모델을 만들었습니다. 기존에 이미지 인식에서 사용하는 이미지 특징을 추출하는 기능을 사용하였습니다. 다만 추출된 이미지의 특징을 마지막에 분류하는 것이 아닌 LSTM모델에 input으로 사용합니다. LSTM에서는 이 들어오는 값을 기반으로 단어들을 생성해내고 이 단어들의 모아서 문장을 만듭니다.

    Keras 프레임워크를 활용하였습니다. 이미지의 특징을 추출하는 모델은 기존에 pretrained된 Xception모델을 활용하였습니다. 

   

   *참고 자료

   [Python based Project - Learn to Build Image Caption Generator with CNN & LSTM - DataFlair (data-flair.training)](https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/)

   

   [시각적주의가있는 이미지 캡션  | TensorFlow Core](https://www.tensorflow.org/tutorials/text/image_captioning)

   



