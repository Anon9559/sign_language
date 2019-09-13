"""
NOTES SECTION
test commit
test 2

"""

#%%
"""
Data preprocessing
"""
#%%
import os
gesture = 'single'
dir_name = "/Users/arjun/documents/git/hand_gesture/arj_database/singles"

test = os.listdir(dir_name)
counter = 0 
for subdir,dirs,files in os.walk(dir_name):
    for file in files:
        counter+=1
        new_file_name = gesture + '-' + str(counter) + '.png'
        filepath = subdir + os.sep + file
        os.rename(filepath,new_file_name)

#%%
"""
Labelling files correctly
"""
import os 
gestures = ['fist','index','loser','okay','open_5','peace']
for gesture in gestures:
    dir_name = "/Users/arjun/documents/git/hand_gesture/arj_database/train/"
    dir_name = dir_name + gesture + '/'
    test = os.listdir(dir_name)
    os.chdir(dir_name)
    counter = 0 
    for subdir,dirs,files in os.walk(dir_name):
        for file in files:
            counter+=1
            new_file_name = gesture + '-' + str(counter) + '.jpg'
            filepath = subdir + os.sep + file
            os.rename(filepath,new_file_name)
#%%
"""
Moved random 30 files from each trainin folder into its respective test 
folder using the linux command below:
    
shuf -n 30 -e * | xargs -i mv {} ~/Documents/git/hand_gesture/arj_database/test/peace/

Run this in each training folder and it moves 30 test images over to their folders
"""
#%%
""""""""""""""""""""""""""""""""""""""""
Building the CNN
""""""""""""""""""""""""""""""""""""""""
#%%
# Importing.
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense     
#%%
"""
Initialise the CNN.
"""
classifier = Sequential() # Creates a Sequential class. 
#%%
"""
Convolution.
-Classifier.add()
    32 filters of 3x3 dimension (first 3 args)
    Working with colour (3 channels) images. Using reduced image dimension to 
    reduce CPU load. 64x64 dimension.
    If using Theano backend should be input_shape = (3,64,64). We're using TF backend.
    Activation function ReLU to reduce nonlinearity.
 
"""
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
#%%
"""
Pooling.
-In general we take 2x2 as it keeps the information.
-Reduce size of filter maps (div by 2). Reduces complexity of model 
    without reducing performacne
"""
classifier.add(MaxPooling2D(pool_size = (2,2)))     

#%%
"""
2nd convolutional layer
"""
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))  

#%%
"""
Flattening.
Flattem feature maps in pooling layer. Gets ready to go into ANN
"""
classifier.add(Flatten())
#%%
"""
Full connection.
-units is how many nodes in the layer. Good to mess around with               
this
- First layer is the hidden layer.
- Second layer is the output layer. Using softmax function because 
we have > 2 output possibilities. If it was a binary classifier we'd use sigmoid.

"""
# Should 2nd one be units 11 or 1? 99% certain it should be 11, as I have 11 
# classes.
classifier.add(Dense(activation = 'relu', units=128 ))
classifier.add(Dense(activation = 'softmax', units=6 ))
#%%
"""
Compiling
- First param = optimiser (for choosing SGD)
- second is the loss function. If had a binary out come using binary cross e
- Third = performacne metric
"""
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
#%%
from keras.preprocessing.image import ImageDataGenerator
#%%
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#%%
test_datagen = ImageDataGenerator(rescale=1./255)
#%%
# Identifies training set and runs transforms.
training_set = train_datagen.flow_from_directory(
        'C:/Users/arjun/Documents/git/hand_gesture/arj_database/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
#%%
# Identifies test set and runs transforms.
test_set = test_datagen.flow_from_directory(
        'C:/Users/arjun/Documents/git/hand_gesture/arj_database/test',
        target_size=(64, 64), # Expects 64 x 64 images
        batch_size=32,
        class_mode='categorical')
#%%
# Fit and test.
classifier.fit_generator(
        training_set,
        steps_per_epoch=3180,
        epochs=15,
        validation_data=test_set,
        validation_steps=360)
#%%
# Save model 
classifier.save('hand_gesture_arj_model_3.h5')
#%%
# Save model weights 
classifier.save_weights('hand_gesture_arj_model_3_weights.h5')
#%%
# Load the model.
from keras.models import load_model
model = load_model('arj_model_3.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
from keras.models import load_model
model = load_model('arj_model_3.h5')
#%%
# Convert to JSON
model_json = model.to_json()
with open("final_model.json","w") as json_file:
    json_file.write(model_json)
#%%
# Use this to get array mapping to class label.
training_set.class_indices      
#%%
"""
Testing my images
"""
from keras.preprocessing import image
import numpy as np
file_path_root='C:/Users/arjun/Documents/git/hand_gesture/arj_database/singles/'
file_name = 'single-7'
file_extension = '.jpg'
full_path = file_path_root + file_name+file_extension

test_image = image.load_img(full_path,target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

# Change classifier to whatever model is loaded as e.g. model
result = classifier.predict(test_image)

if result[0][0] ==1:
    print('Fist')
elif result[0][1] ==1:
    print('Index')
elif result[0][2] ==1:
    print('Loser')
elif result[0][3] ==1:
    print('Okay')
elif result[0][4] ==1:
    print('Open_5')
elif result[0][5] ==1:
    print('Peace')

#%%
"""
Testing my training images
"""
from keras.preprocessing import image
import numpy as np
file_path_root='C:/Users/arjun/Documents/git/hand_gesture/arj_database/test/open/'
file_name = 'loser-10'
file_extension = '.png'
full_path = file_path_root + file_name+file_extension

test_image = image.load_img(full_path,target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

# Change classifier to whatever model is loaded as e.g. model
result = classifier.predict(test_image)

if result[0][0] ==1:
    print('Fist')
elif result[0][1] ==1:
    print('Index')
elif result[0][2] ==1:
    print('Loser')
elif result[0][3] ==1:
    print('Okay')
elif result[0][4] ==1:
    print('Open_5')
elif result[0][5] ==1:
    print('Peace')