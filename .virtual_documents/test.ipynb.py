import os
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


tf.__version__


# set up Gpu

# avoid dom errors by seting gpu memory consumption Growth

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


gpus


#create data dir
work_dir ="/home/mostafa/Documents/Draft/FacialVerification"
DATA_DIR = "data"
POS_DIR = os.path.join(DATA_DIR,'positive')
NEG_DIR = os.path.join(DATA_DIR,'negative')
ANK_DIR = os.path.join(DATA_DIR,'anchor')
#get_ipython().getoutput("mkdir data/positive")
#get_ipython().getoutput("mkdir data/negative")
#get_ipython().getoutput("mkdir data/anchor")


#get_ipython().getoutput("tar -xf lfw.tgz")







for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join(work_dir,'lfw',directory)):
        current_path = os.path.join(work_dir,'lfw',directory,file)
        destination_path = os.path.join(work_dir,NEG_DIR,file)
        os.replace(current_path,destination_path)










# establish aconnection to webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # reading frame
    ret, frame = cap.read()
    # cut frame to 250X250px
    frame=frame[120:120+250,200:200+250,:]
    # show the captured image
    if cv2.waitKey(1) & 0xFF == ord('a'):
        img_path = os.path.join(ANK_DIR,f"{uuid.uuid1()}.jpg")
        cv2.imwrite(img_path,frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        img_path = os.path.join(POS_DIR,f"{uuid.uuid1()}.jpg") 
        cv2.imwrite(img_path,frame)
    cv2.imshow("Collected Image",frame)
    
    #breaking the system when done
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release the webcam
cap.release()
# colse all image show
cv2.destroyAllWindows()


plt.imshow(frame[120:120+250,200:200+250,:])


anchor = tf.data.Dataset.list_files(ANK_DIR+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_DIR+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_DIR+'/*.jpg').take(300)


def preprocess(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img,(100,100))
    img = img/255.0
    return img


img=preprocess(anchor.as_numpy_iterator().next())
plt.imshow(img)


positive = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positive.concatenate(negative)


data.as_numpy_iterator().next()


def preprocess_twin(input_img,validation_img,label):
    return (preprocess(input_img),preprocess(validation_img),label)


input_img,validation_img,label=preprocess_twin(*(data.as_numpy_iterator().next()))


plt.imshow(input_img)


plt.imshow(validation_img)


label


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


plt.imshow(data.as_numpy_iterator().next()[1])


# training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

                       


# test partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)
                       


def make_embedding(): 
    inp = Input(shape=(100,100,3),name="Input Image")
    C1 = Conv2D(64,(10,10),activation='relu')(inp)
    M1 = MaxPooling2D(64,(2,2),padding='same')(C1)
    
    C2 = Conv2D(128,(7,7),activation='relu')(M1)
    M2 = MaxPooling2D(64,(2,2),padding='same')(C2)
    
    C3 = Conv2D(128,(4,4),activation='relu')(M2)
    M3 = MaxPooling2D(64,(2,2),padding='same')(C3)
    
    C4 = Conv2D(256,(4,4),activation='relu')(M3)
    F1 = Flatten()(C4)
    FC1 = Dense(units=4069,activation='sigmoid')(F1)

    
    return Model(inputs=[inp], outputs=[FC1], name='embedding')


model = make_embedding()


model.summary()


class L1DIST(Layer):
    def __init__(self,**kwargs):
        super().__init__()
    
    def call(self,input_img,validation_img):
        return tf.math.abs(input_img-validation_img)


def make_siames_model():
    input_image = Input(shape=(100,100,3), name = "Input Embedding")
    validation_image = Input(shape=(100,100,3), name = "Validation Embedding")
    
    siames_layer = L1DIST()
    siames_layer._name ="DistanceLayer"
    distances = siames_layer(model(input_image),model(validation_image))
    
    classifier = Dense(1, activation = 'sigmoid')(distances)
    
    return Model(inputs = [input_image, validation_image], outputs = [classifier], name = 'SiameseNetwork')
    


siamese_model = make_siames_model()


siamese_model.summary()


binary_cross_loss = tf.losses.BinaryCrossentropy()
optimizer = tf.optimizers.Adam(1e-4)




! mkdir training_checkpoints


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt=optimizer, siamese_model=siamese_model)


len(train_data.as_numpy_iterator().next())


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        #splet labels and data
        X = batch[:2]
        
        y = batch[2]
        
        #forward_propagation 
        
        yhat = siamese_model(X,training = True)
        loss = binary_cross_loss(y,yhat)
    
    #calculating Gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
        
    return loss


def train(data,epochs):
    for epoch in range(1,epochs+1):
        
        print(f"\nEpochs {epoch}/{epochs}")
        progpar = tf.keras.utils.Progbar(len(data))
        
        for idx, batch in enumerate(data):
            train_step(batch)
            progpar.update(idx+1)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            


train(train_data,50)


from tensorflow.keras.metrics import Precision, Recall


test_input, test_validate, label = test_data.as_numpy_iterator().next()


plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
plt.imshow(test_input[0])
plt.subplot(1,2,2)
plt.imshow(test_validate[0])


y_hat = model.predict([test_input, test_validate])
y_hat


[1 if prediction > .5 else 0 for prediction in y_hat ]


label


# Creating a metric object 
m = Recall()

# Calculating the recall value 
m.update_state(label, y_hat)

# Return Recall Result
m.result().numpy()


# Creating a metric object 
m = Precision()

# Calculating the recall value 
m.update_state(label, y_hat)

# Return Recall Result
m.result().numpy()


siamese_model.save("siamese_model.h5")


model = tf.keras.models.load_model('siamese_model.h5',custom_objects={'L1DIST':L1DIST, "BinaryCrossentopy":tf.losses.BinaryCrossentropy()})


model.summary()


get_ipython().getoutput("mkdir ./application_data")
get_ipython().getoutput("mkdir ./application_data/input_image")
get_ipython().getoutput("mkdir ./application_data/verification_images")



def verify(model, detection_threshold, verification_threshold):
    
    results = []
    
    for image in os.listdir(os.path.join('./application_data','verification_images')):
        validation_img = preprocess(os.path.join('./application_data','verification_images',image))
        input_img = preprocess(os.path.join('./application_data','input_image',"input_image.jpg"))
        
        # make prediction 
        result = model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
        results.append(result)
    detection = np.sum(np.array(results)>detection_threshold)
    verification = detection / len(os.listdir(os.path.join('./application_data','verification_images')))
    verified = verification > verification_threshold
    return results, verified
        
        
    


# establish aconnection to webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # reading frame
    ret, frame = cap.read()
    # cut frame to 250X250px
    frame=frame[120:120+250,200:200+250,:]
    # show the captured image
    if cv2.waitKey(1) & 0xFF == ord('v'):
        img_path = os.path.join('./application_data','input_image',"input_image.jpg")
        cv2.imwrite(img_path,frame)
        results, verified = verify(model, .9, .7)
        print(f"verifid = {verified} \n")

        
    cv2.imshow("Collected Image",frame)
    
    #breaking the system when done
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release the webcam
cap.release()
# colse all image show
cv2.destroyAllWindows()






