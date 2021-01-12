import argparse
import os
os.environ['DISPLAY'] = ':0'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import pyautogui, sys
from helper_functions import draw_box, move_mouse_to_another_point, create_mobilenet_v2_with_preprocessing_function
import time



def mouse_controller(model, preprocessing_function, input_width, input_height, just_tracking):
    tracker = cv2.TrackerCSRT_create()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    bbox = cv2.selectROI(frame, False)
    x1,y1,xh,yh=bbox
    screen_width, screen_height = pyautogui.size()
    little_screen_width, little_screen_height, _ = frame.shape
    little_screen_width = little_screen_width // 2
    little_screen_height = little_screen_height // 2

    move_mouse_to_another_point(little_screen_width, little_screen_height, 
                                screen_width, screen_height, (2 * x1 + xh) // 2, (2 * y1 + yh) // 2)
    ok = tracker.init(frame, bbox)
    img_number = 0

    while(True):
        try:        
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            predicted = model(preprocessing_function(tf.cast(tf.expand_dims(cv2.resize(frame,(input_width,input_height)),0),
                tf.float32)),training=False)[0]
                
            if just_tracking != True:
                if tf.math.argmax(predicted) == 1:
                    pyautogui.click()

            success, bbox = tracker.update(frame)
            if success:
                draw_box(frame,bbox)
            else:
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                break
            
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            p_x, p_y=((p1[0] + p2[0])//2, int(p1[1] * 0.7 + p2[1] * 0.3))
            move_mouse_to_another_point(little_screen_width, little_screen_height, screen_width,screen_height,p_x,p_y)
            frame = cv2.circle(frame, (p_x,p_y), radius=2, color=(0, 0, 255), thickness=4)
            cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print("ERROR occoured")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default=256, type=int)
    parser.add_argument('--test_data_path', default=256, type=int)
    
    parser.add_argument('--input_width', default=224, type=int)
    parser.add_argument('--input_height', default=224, type=int)
    parser.add_argument('--num_units_1', default=128, type=int)
    parser.add_argument('--num_units_2', default=64, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--drouput_rate_1', default=0.10, type=float)
    parser.add_argument('--dropout_rate_2', default=0.05, type=float)
    parser.add_argument('--lambda_for_regularization', default=0.001, type=float)
    parser.add_argument('--activation_for_hidden_layers', default='relu', type=str)
    parser.add_argument('--activatoin_function', default="softmax", type=str)
    parser.add_argument('--base_trainable', default=0, type=int)
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--model_weight_path', type=str)
    parser.add_argument('--just_tracking', default=False, type=bool)
    args = parser.parse_args()

    model, preprocessing_function = create_mobilenet_v2_with_preprocessing_function(args.input_width, args.input_height, args.num_units_1, 
        args.num_units_2, args.num_classes, args.drouput_rate_1, args.dropout_rate_2, args.lambda_for_regularization, args.activation_for_hidden_layers,
        args.activatoin_function, args.base_trainable, args.model_weight_path)

    mouse_controller(model, preprocessing_function, args.input_width, args.input_height, args.just_tracking)
#python3 mouse_from_screen.py --just_tracking True