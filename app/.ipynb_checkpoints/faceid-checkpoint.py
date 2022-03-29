
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import os
import numpy as np
import cv2
import tensorflow as tf
from layers import L1DIST


class CamApp(App):
    
    def build(self):
        
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint = (1,.1))
        self.verification_label = Label(text = "Verification Uninitiated", size_hint = (1,.1))

        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1./33.)
        
        self.model = tf.keras.models.load_model('siamese_model.h5',custom_objects={'L1DIST':L1DIST, "BinaryCrossentopy":tf.losses.BinaryCrossentropy()})
        return layout
    
    def update(self,*args):
        ret, frame = self.capture.read()
        # cut frame to 250X250px
        frame=frame[120:120+250,200:200+250,:]
        # show the captured image
        
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size = (frame.shape[1], frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt = 'ubyte')
        self.web_cam.texture = img_texture
        
    def preprocess(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize(img,(100,100))
        img = img/255.0
        return img
        
    def verify(self, *args):
        detection_threshold=.9
        verification_threshold=.8
        save_path=os.path.join('./application_data','input_image',"input_image.jpg")
        ret, frame = self.capture.read()
        frame=frame[120:120+250,200:200+250,:]
        cv2.imwrite(save_path,frame)
        
        results = []
        
        for image in os.listdir(os.path.join('./application_data','verification_images')):
            validation_img = self.preprocess(os.path.join('./application_data','verification_images',image))
            input_img = self.preprocess(os.path.join('./application_data','input_image',"input_image.jpg"))
            
            # make prediction 
            result = self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
            results.append(result)
        detection = np.sum(np.array(results)>detection_threshold)
        verification = detection / len(os.listdir(os.path.join('./application_data','verification_images')))
        verified = verification > verification_threshold
        
        self.verification_label.text = "Verified" if verified == True else 'Unverified'
        
        
        Logger.info(results)
        Logger.info(np.sum(np.array(results)>detection_threshold))
        Logger.info(np.sum(np.array(results)>.6))
        Logger.info(np.sum(np.array(results)>.4))
        Logger.info(np.sum(np.array(results)>.2))
        return results, verified
        
        
    
        
        
        
        
if __name__ == '__main__':
    CamApp().run()