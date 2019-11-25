# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import time

import numpy as np
import web
from PIL import Image

web.config.debug = True

from config import *
from apphelper.image import union_rbox, adjust_box_to_origin

import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3  ## GPU最大占用量
config.gpu_options.allow_growth = True  ##GPU是否可动态增加
K.set_session(tf.Session(config=config))
K.get_session().run(tf.global_variables_initializer())

scale, maxScale = IMGSIZE[0], 2048
from text.keras_detect import text_detect

from text.opencv_dnn_detect import angle_detect

from crnn.keys import alphabetChinese, alphabetEnglish

from crnn.network_torch import CRNN

if chineseModel:
    alphabet = alphabetChinese
    if LSTMFLAG:
        ocrModel = ocrModelTorchLstm
    else:
        ocrModel = ocrModelTorchDense

else:
    ocrModel = ocrModelTorchEng
    alphabet = alphabetEnglish
    LSTMFLAG = True

nclass = len(alphabet) + 1
crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
if os.path.exists(ocrModel):
    crnn.load_weights(ocrModel)
else:
    print("download model or tranform model with tools!")

ocr = crnn.predict_job

from main import TextOcrModel

model = TextOcrModel(ocr, text_detect, angle_detect)

billList = ['通用OCR', '火车票', '身份证']

if __name__ == '__main__':
    with tf.Session() as sess:
        # t = time.time()
        img = Image.open('test/img.jpeg').convert('RGB')
        img = np.array(img)
        print(img.shape)
        # detectAngle = True
        detectAngle = False
        result, angle = model.model(img,
                                    scale=scale,
                                    maxScale=maxScale,
                                    detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                    MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔，用于文本行的合并
                                    MIN_V_OVERLAPS=0.6,
                                    MIN_SIZE_SIM=0.6,
                                    TEXT_PROPOSALS_MIN_SCORE=0.1,
                                    TEXT_PROPOSALS_NMS_THRESH=0.3,
                                    TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                    LINE_MIN_SCORE=0.1,
                                    leftAdjustAlph=0.01,  ##对检测的文本行进行向左延伸
                                    rightAdjustAlph=0.01,  ##对检测的文本行进行向右延伸
                                    )

        result = union_rbox(result, 0.2)
        print(result)
        exit()
        # K.clear_session()
    # res = [{'text': x['text'],
    #         'name': str(i),
    #         'box': {'cx': x['cx'],
    #                 'cy': x['cy'],
    #                 'w': x['w'],
    #                 'h': x['h'],
    #                 'angle': x['degree']
    #
    #                 }
    #         } for i, x in enumerate(result)]
    # res = adjust_box_to_origin(img, angle, res)  ##修正box
