import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras import activations
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model

def normalize(images):
    return (images.astype(np.float32)/255.0)

def denormalize(images):
    return np.clip(images*255, a_min=0, a_max=255).astype(np.uint8)

# Initiate model with required hyperparameters
class Model_Train():
    def __init__(self, summary_dir, checkpoint_dir, learning_rate, min_learning_rate):
        self.step = tf.Variable(0,dtype=tf.int64)
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate 
        self.build_model()
        log_dir = os.path.join(summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):
        self.generator = unet_16()

        #Learning rate decay for every 200 iterations
        self.lr_scheduler_fn =  tf.compat.v1.train.exponential_decay(self.learning_rate, self.step, 200, 0.1,  staircase=True)
        self.learning_rate = lambda : tf.maximum(self.min_learning_rate, self.lr_scheduler_fn())

        self.generator_optimizer = tf.keras.optimizers.Adam( self.learning_rate )

        """ saver """
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                        generator_optimizer=self.generator_optimizer,
                                        generator=self.generator,
                                        )
        self.save_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=3)
        self.save  = lambda : self.save_manager.save(checkpoint_number=self.step) #exaple : model.save()



    @tf.function
    def training(self, inputs):
        paired_input, paired_target = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            prediction = self.generator(paired_input) 
            total_loss = tf.reduce_mean(L1loss(paired_target, prediction))

        """ optimize """
        params_gradients = self.generator.trainable_variables
        generator_gradients = gen_tape.gradient(total_loss, params_gradients)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, params_gradients))

        inputs_concat = tf.concat([paired_input, paired_target], axis=2)
        return_dicts = {"inputs_concat" :inputs_concat}
        return_dicts.update({'total_loss' : total_loss})
        return_dicts.update({'Prediction': tf.concat([paired_input, prediction, paired_target],axis=2)})
        return return_dicts



    def train_step(self,iterator, summary_name = "train", log_interval = 100):
        """ training """
        result_logs_dict = self.training(iterator.__next__())

        """ log summary """
        if summary_name and self.step.numpy() % log_interval == 0:
            with self.train_summary_writer.as_default():
                for key, value in result_logs_dict.items():
                    value = value.numpy()
                    if len(value.shape) == 0:
                        tf.summary.scalar("{}_{}".format(summary_name,key), value, step=self.step)
                    elif len(value.shape) in [3,4]:
                        tf.summary.image("{}_{}".format(summary_name, key), denormalize(value), step=self.step)


        """ return log str """
        log = "Total_Loss : {} lr : {}".format(result_logs_dict["total_loss"], self.learning_rate().numpy())
        return log, [denormalize(result_logs_dict["Prediction"].numpy())]



def L1loss(input,target):
    #return tf.reduce_sum(tf.reduce_mean(tf.abs(input - target),axis=0))
    return tf.reduce_mean(tf.abs(input - target))



def identity(input_image_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=input_image_shape))
    return model


def simplest(input_image_shape):
    inputs = Input(input_image_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    model = Model(input=inputs, output=conv2)
    return model

def unet_16():
    return unet(base_num_filters=16)


def unet(input_size = [48, 48, 3], base_num_filters=64):
    inputs = Input(input_size)
    conv1 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(16 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(16 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

    conv6 = Conv2D(32 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(32 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)


    up6 = Conv2D(16 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(3, 3))(drop6))
    merge6 = concatenate([drop5, up6])
    conv6 = Conv2D(16 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(16 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(8 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv4, up7])
    conv7 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(4 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv3, up8])
    conv8 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(2 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv2, up9])
    conv9 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    up10 = Conv2D(base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv9))
    merge10 = concatenate([conv1, up10])
    conv10 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

    model = Model(inputs=[inputs], outputs=conv10)
    return model
