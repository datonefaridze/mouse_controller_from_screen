import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from helper_functions import load_dataset, create_mobilenet_v2_with_preprocessing_function

def start_training(model, train_data_path, test_data_path, batch_size, input_height, input_width, image_preprocessing_function,
    learning_rate, epochs):

    train_dataset, test_dataset = load_dataset(train_data_path, test_data_path, batch_size, input_height, input_width, image_preprocessing_function)
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    checkpoint = ModelCheckpoint(filepath = "best_mobilenet_model.hdf5",
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                mode='max',
                                period=1)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=model_optimizer,metrics=['accuracy'], loss=loss_object)
    hist = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs,  callbacks=[checkpoint], verbose=1) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_path', default="new_five_finger_data/train", type=str)
    parser.add_argument('--test_data_path', default="new_five_finger_data/test", type=str)
    
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_weight_path', type=str)

    args = parser.parse_args()
    model, preprocessing_function = create_mobilenet_v2_with_preprocessing_function(args.input_width, args.input_height, args.num_units_1, 
        args.num_units_2, args.num_classes, args.drouput_rate_1, args.dropout_rate_2, args.lambda_for_regularization, args.activation_for_hidden_layers,
        args.activatoin_function, args.base_trainable, args.model_weight_path)

    start_training(model, args.train_data_path, args.test_data_path, args.batch_size, args.input_height, args.input_width,
     preprocessing_function, args.learning_rate, args.epochs)



