import cv2
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def draw_box(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.putText(frame,"Tracking", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

def show(img):
    cv2.imshow('img',np.array(img, dtype=np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
def move_mouse_to_another_point(screen_width_1, screen_height_1, screen_width_2, screen_height_2, from_x, from_y):
    import pyautogui
    to_x = (from_x * screen_width_2) / screen_width_1
    to_y = (from_y * screen_height_2) / screen_height_1
    pyautogui.moveTo(to_x, to_y)


def create_mobilenet_v2_with_preprocessing_function(input_width, input_height, num_units_1, num_units_2, num_classes, drouput_rate_1, dropout_rate_2,
    lambda_for_regularization, activation_for_hidden_layers, activatoin_function, base_trainable, model_weight_path):

    mob_inp = tf.keras.layers.Input(shape=(input_width, input_height, 3))
    mobilenet = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(input_width,input_height,3), input_tensor=mob_inp)
    mobilenet.trainable = base_trainable

    mob_from_base = mobilenet.output
    mob_GAP = tf.keras.layers.GlobalAveragePooling2D()(mob_from_base)
    mob_dropout_1 = tf.keras.layers.Dropout(drouput_rate_1)(mob_GAP)
    mob_dense_1 = tf.keras.layers.Dense(num_units_1, activation=activation_for_hidden_layers,
      kernel_regularizer=regularizers.l2(lambda_for_regularization))(mob_dropout_1)
      
    mob_dropout_2 = tf.keras.layers.Dropout(dropout_rate_2)(mob_dense_1)
    mob_dense_2 = tf.keras.layers.Dense(num_units_2, activation=activation_for_hidden_layers,
      kernel_regularizer=regularizers.l2(lambda_for_regularization))(mob_dropout_2)
    mob_output = tf.keras.layers.Dense(num_classes, activation=activatoin_function)(mob_dense_2)
    model = tf.keras.models.Model(inputs=mob_inp, outputs=mob_output)

    if model_weight_path is not None:
        model.load_weights(model_weight_path)

    return model, tf.keras.applications.mobilenet_v2.preprocess_input



def load_dataset(train_data_path, test_data_path, batch_size, input_height, input_width, image_preprocessing_function):
    train_datagen = ImageDataGenerator(zoom_range = 0.2,
                                       rotation_range=15,
                                       preprocessing_function=image_preprocessing_function,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(preprocessing_function=image_preprocessing_function)
    train_dataset  = train_datagen.flow_from_directory(train_data_path,
                                    target_size = (input_height, input_width),
                                    batch_size = batch_size,
                                    class_mode = 'categorical',
                                    shuffle=True)
    test_dataset =  test_datagen.flow_from_directory(test_data_path,
                                            target_size = (input_height, input_width),
                                            batch_size = batch_size,
                                            class_mode = 'categorical',
                                            shuffle = True)

    return train_dataset,test_dataset
