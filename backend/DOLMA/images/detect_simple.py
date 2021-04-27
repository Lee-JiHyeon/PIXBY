import tensorflow as tf
from .core import utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from pathlib import Path
from django.core.files.storage import default_storage

# Path(__file__).resolve().parent / 'ssafy2.jpg'

MODEL_PATH = str(Path(__file__).resolve().parent / 'checkpoints' / 'yolov4-416')
# MODEL_PATH = './checkpoints/yolov4-416'

print('---------------------------------------')
print(MODEL_PATH)
print('---------------------------------------')

IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

class_list = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# def main():
def main(img_path, kidId):
    # img_path = Path(__file__).resolve().parent / str(img_path)
    # img_path = str(img_path)
    # print(img_path)
    img = cv2.imread(img_path)
    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_input = img_input / 255.
    img_input = img_input[np.newaxis, ...].astype(np.float32)
    img_input = tf.constant(img_input)

    pred_bbox = infer(img_input)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD
    )
    # 단어를 못찾아도 50개를 채워서 찾음
    # 캡셔닝해서 찾은 단어 갯수로 클래스들을 슬라이싱함
    
    
    # 숫자로 이루어진 클래스를 이름으로 변경
    
    # print(scores)
    # for i in range(n):
        # if scores[n]

    # 캡셔닝 이미지로 score 0.6이상만 저장
    # nbox = boxes.numpy()
    # print(nbox)
    # nbox[0] = np.delete(nbox[0], 0, 0)
    # print(nbox)
    # nscore = scores.numpy()
    # nclass = classes.numpy()
    # N= valid_detections.numpy()[0]
    # i = 0
    # while i < N:
    #     if nscore[0][i] < 0.6:
    #         N -= 1
    #         print(nscore[0][i])
    #         # print(nbox[0][i])
    #         print(nbox[0])
    #         np.delete(nbox[0], i, axis=0)
    #         print(nbox[0])
    #         # print(nscore[0][i])
    #         np.delete(nscore[0], i, axis=0)
    #         # print(nclass[0][i])
    #         np.delete(nclass[0], i, axis=0)
    #         pass
    #     else:
    #         i+=1 
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # pred_bbox = [nbox, nscore, nclass, [N]]
    result = utils.draw_bbox(img, pred_bbox)

    n = valid_detections.numpy()[0]
    valid_classes = classes.numpy()[0][:n]
    # print(nclass)
    # print(n, N)
    # print(valid_classes)
    clst = []
    for v in valid_classes:
        clst.append(class_list[int(v)])

    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'media/output{kidId}.png', result)
    
    # 프론트에 넘겨줄 값, 가공할 값 리턴
    context = {
        'classlist': clst,
        'box': boxes.numpy()[0][:n],
    }
    return context
    # default_storage.save(f'yolo-{img_path}', result)

# if __name__ == '__main__':
#     img_path = './data/kite.jpg'
#     main(img_path)
