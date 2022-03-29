
import tensorflow as tf
from tensorflow.keras.layers import Layer

class L1DIST(Layer):
    def __init__(self,**kwargs):
        super().__init__()
    
    def call(self,input_img, validation_img):
        return tf.math.abs(input_img-validation_img)