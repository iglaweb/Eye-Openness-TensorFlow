import datetime
import glob
import os
from pathlib import Path

import cv2
import dlib
import numpy
import tensorflow as tf
from imutils import face_utils
from pandas import np

assert tf.__version__.startswith('2')

"""
Use this to run interference
Helpful links
https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=WFHKkb7gcJei
"""


def get_timestamp_ms():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

TFLITE_FLOAT_MODEL = './models_output/mobilenetv2_eyes_model_float.tflite'

# it runs much slower than float version on CPU
# https://github.com/tensorflow/tensorflow/issues/21698#issuecomment-414764709
TFLITE_QUANT_MODEL = './models_output/mobilenetv2_eyes_model_quant.tflite'
VIDEO_FILE = '/Users/igla/Downloads/Memorable Monologue- Talking in the Third Person.mp4'
TEST_DIR = './out_close_eye/'
dataset_labels = ['closed', 'opened']

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')

# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe('caffe/deploy.prototxt', 'caffe/weights.caffemodel')

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_FLOAT_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", interpreter.get_output_details()[0]['name'])
print("shape:", interpreter.get_output_details()[0]['shape'])
print("type:", interpreter.get_output_details()[0]['dtype'])

floating_model = input_details[0]['dtype'] == np.float32
print('Floating model is: ' + str(floating_model))

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

time_elapsed = 0
exec_cnt = 0


def make_interference(image_frame):
    """
        Return True if opened class detected, otherwise False
    """
    # Acquire frame and resize to expected shape [1xHxWx3]
    # add N dim
    image_frame = cv2.resize(image_frame, (width, height), cv2.INTER_AREA)
    cv2.imshow('TEST', image_frame)

    if floating_model:
        # Normalize to [0, 1]
        image_frame = image_frame / 255.0
        images_data = np.expand_dims(image_frame, 0).astype(np.float32)  # or [img_data]
    else:  # 0.00390625 * q
        images_data = np.expand_dims(image_frame, 0).astype(np.uint8)  # or [img_data]

    start = get_timestamp_ms()

    # Inference on input data normalized to [0, 1]
    # input_img = np.expand_dims(data, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    global time_elapsed
    global exec_cnt
    diff = get_timestamp_ms() - start
    time_elapsed = time_elapsed + diff
    exec_cnt = exec_cnt + 1
    print(f'Elapsed time: {diff} ms')
    print(f'Avg time: {time_elapsed / exec_cnt}')

    predict_label = np.argmax(output_data)
    score = 100 * output_data[0][predict_label]
    if floating_model is False:
        score = score * 0.00390625

    # print(np.argmax(output()[0]))
    print("Predicted value for [0, 1] normalization. Label index: {}, confidence: {:2.0f}%"
          .format(predict_label, score))

    print(output_data)
    results = np.squeeze(output_data)
    top_k = results.argsort()[-2:][::-1]

    for i in top_k:
        print(dataset_labels[i], results[i])

    # max_index_col = np.argmax(results, axis=0)
    return True if top_k[0] == 1 else False


def detect_face(image):
    # accessing the image.shape tuple and taking the elements
    (h, w) = image.shape[:2]  # get our blob which is our input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # input the blob into the model and get back the detections
    face_model.setInput(blob)
    detections = face_model.forward()
    # Iterate over all of the faces detected and extract their start and end points
    count = 0
    rect_list = []
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            print('Face confidence: ' + str(confidence))
            rect_list.append((startX, startY, endX, endY))
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            count = count + 1  # save the modified image to the Output folder
    return rect_list


def detect_eye_left(frame_crop, shape):
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    # cv2.drawContours(frame_crop, [leftEyeHull], -1, (0, 255, 0), 1)
    # cv2.drawContours(frame_crop, [rightEyeHull], -1, (0, 255, 0), 1)

    # enlarge eye regions
    (l, t, r, b) = (shape[17][0], shape[17][1], shape[39][0], shape[40][1])
    eye_width = shape[39][0] - shape[36][0]
    eye_height = max(shape[40][1], shape[41][1]) - min(shape[37][1], shape[38][1])

    half_width = eye_width / 2
    half_height = eye_height / 2

    brow_eye_dist = (shape[37][1] - shape[19][1]) / 2

    l = int(l - half_width)
    r = int(r + half_width)
    t = int(t - half_height)
    b = int(b + half_height + brow_eye_dist)

    eye_region = frame_crop[t:b, l:r]
    if eye_region is None or eye_region.size == 0:
        return None, True

    # eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    # eye_region = cv2.cvtColor(eye_region, cv2.COLOR_GRAY2BGR)

    return eye_region, make_interference(eye_region)


def detect_eye_right(frame_crop, shape):
    # enlarge eye regions
    (l, t, r, b) = (shape[22][0], shape[22][1], shape[45][0], shape[46][1])
    eye_width = shape[45][0] - shape[42][0]
    eye_height = max(shape[47][1], shape[46][1]) - min(shape[43][1], shape[44][1])

    half_width = eye_width / 2
    half_height = eye_height / 2

    brow_eye_dist = (shape[43][1] - shape[23][1]) / 2

    l = int(l - half_width)
    r = int(r + half_width)
    t = int(t - half_height)
    b = int(b + half_height + brow_eye_dist)

    eye_region = frame_crop[t:b, l:r]
    if eye_region is None or eye_region.size == 0:
        return None, True

    # eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    # eye_region = cv2.cvtColor(eye_region, cv2.COLOR_GRAY2BGR)

    return eye_region, make_interference(eye_region)


def clear_test():
    # clear previous output images¬
    files = glob.glob(f'{TEST_DIR}*.jpg', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # test http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ ?
    # https://sites.google.com/view/utarldd/home

    clear_test()

    eye_close_counter = 0
    placeholder_max_w = 0
    placeholder_max_h = 0

    vid = cv2.VideoCapture(VIDEO_FILE)
    while True:
        _, frame = vid.read()

        face_list = detect_face(frame)
        if len(face_list) == 0:
            print('Face size empty')
            cv2.imshow('Image', frame)
            cv2.waitKey(1)
            continue

        for face in face_list:
            (startX, startY, endX, endY) = face_list[0]
            frame_crop = frame[startY:endY, startX:endX]

            # determine the facial landmarks for the face region, then
            height_frame, width_frame = frame_crop.shape[:2]

            # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
            shape = predictor(frame_crop, dlib.rectangle(0, 0, width_frame, height_frame))
            shape = face_utils.shape_to_np(shape)

            (left_eye_area, left_eye_opened) = detect_eye_left(frame_crop, shape)
            (right_eye_area, right_eye_opened) = detect_eye_right(frame_crop, shape)

            if left_eye_area is None or right_eye_area is None:
                print('Skip frame')
                continue

            # img_test = cv2.imread(
            #  '/Users/igla/PycharmProjects/DrowsinessClassification/eyes_state/opened/s0001_02127_0_0_1_0_0_01.png')
            # test = make_interference(img_test)

            isOpened = left_eye_opened is True or right_eye_opened is True
            if isOpened is False:
                eye_close_counter = eye_close_counter + 1
                cv2.imwrite(f'{TEST_DIR}left_{eye_close_counter}.jpg', left_eye_area)
                cv2.imwrite(f'{TEST_DIR}right_{eye_close_counter}.jpg', right_eye_area)

            cv2.putText(frame, f"Eyes closed {eye_close_counter}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2)
            opened = "Opened" if isOpened else "Closed"
            cv2.putText(frame, opened, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # placeholder for eye regions
            placeholder_max_w = max(placeholder_max_w, left_eye_area.shape[1] + (10 * 2))
            placeholder_max_w = max(placeholder_max_w, right_eye_area.shape[1] + (10 * 2))

            placeholder_max_h = max(placeholder_max_h, left_eye_area.shape[0] + 10)
            placeholder_max_h = max(placeholder_max_h, right_eye_area.shape[0] + 10)

            cv2.rectangle(frame, (10, 80), (10 + placeholder_max_w, 80 + placeholder_max_h * 2 + 10), (127, 255, 0, 0),
                          -1)
            x_offset = 20
            y_offset = 100
            frame[y_offset:y_offset + left_eye_area.shape[0],
            x_offset:x_offset + left_eye_area.shape[1]] = left_eye_area
            y_offset = y_offset + placeholder_max_h
            frame[y_offset:y_offset + right_eye_area.shape[0],
            x_offset:x_offset + right_eye_area.shape[1]] = right_eye_area

            for (x, y) in shape:
                cv2.circle(frame_crop, (x, y), 2, (0, 0, 255), -1)
            # cv2.imshow("Face", frame_crop)

            cv2.imshow("Image", frame)
            cv2.waitKey(1)
