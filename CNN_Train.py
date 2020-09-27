


import os,cv2
import numpy as np
import matplotlib.pyplot as plt



from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam




from keras import backend as K
K.set_image_data_format('channels_first')




data_path = "C:/Users/vpp19/OneDrive/Desktop/Input"




data_dir_list = os.listdir(data_path)

img_rows=2
img_cols=10
num_channel=1
num_epoch=5

# Define the number of classes
num_classes = 2

img_data_list=[]
img_data_list2=[]




for i in range(len(data_dir_list)):
    img_name =  str(i+1) + ".jpg"
    img_path = data_path + "//" + img_name
    print ('Loaded the images of dataset-'+'{}\n'.format(i+1))
    
    input_img=cv2.imread(img_path,1)
    #cv2.imshow('image', input_img)
    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img_resize=cv2.resize(input_img,(128,128))
    crop_img=input_img_resize[10:110,10:110]
    constant= cv2.copyMakeBorder(crop_img,10,10,10,10,cv2.BORDER_CONSTANT)
    img_data_list.append(input_img_resize)




img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print ("Image_data shape: ",img_data.shape)




if num_channel==1:
    if K.set_image_data_format('channels_first'):
        img_data= np.expand_dims(img_data, axis=1) 
        #img_data2= np.expand_dims(img_data2, axis=1) 
        print ("*Img_data shape: ",img_data.shape)
        #print ("*Img_data2 shape: ",img_data2.shape)

    else:
        img_data= np.expand_dims(img_data, axis=4) 
        #img_data2= np.expand_dims(img_data2, axis=4) 
        print ("*Img_data shape: ",img_data.shape)
        #print ("*Img_data2 shape: ",img_data2.shape)
	
else:
    if K.set_image_data_format('channels_first'):
        img_data=np.rollaxis(img_data,3,1)
        #img_data2=np.rollaxis(img_data2,3,1)
        print ("*Img_data shape: ",img_data.shape)
       # print ("*Img_data2 shape: ",img_data2.shape)




USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	# using sklearn for preprocessing
	from sklearn import preprocessing
	
	def image_to_feature_vector(image, size=(128, 128)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()
	
	img_data_list=[]
	for dataset in data_dir_list:
		img_list=os.listdir(data_path+'/'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in img_list:
			input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
			input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_flatten=image_to_feature_vector(input_img,(128,128))
			img_data_list.append(input_img_flatten)
	
	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	print (img_data.shape)
	img_data_scaled = preprocessing.scale(img_data)
	print (img_data_scaled.shape)
	
	print (np.mean(img_data_scaled))
	print (np.std(img_data_scaled))
	
	print (img_data_scaled.mean(axis=0))
	print (img_data_scaled.std(axis=0))
	
	if K.set_image_data_format('channels_first'):
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)
	
	
	if K.set_image_data_format('channels_first'):
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)






if USE_SKLEARN_PREPROCESSING:
	img_data=img_data_scaled
	
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
#---------------------------APPEALING / NOT-----------------------------------
labels[1:2200]=0
labels[2201:]=1

names = ['appeal','notappealing']
#-----------------------------------------------------------------------------
	  

	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = (img_data,Y)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



input_shape=img_data[0].shape


model = Sequential()

model.add(Convolution2D(64, (7, 7),strides=(2, 2), padding='same', data_format = "channels_last", activation='relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
#model.add(Dropout(0.75))

model.add(Convolution2D(64, (3, 3), padding='same', activation='relu',data_format = "channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Convolution2D(128, (3, 3), padding='same', activation='relu',data_format = "channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
#model.add(Dropout(0.75))

model.add(Convolution2D(128, (7, 7), padding='same', activation='relu',data_format = "channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
#model.add(Dropout(0.75))

model.add(Convolution2D(64, (5, 5), padding='same', activation='relu',data_format = "channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))



model.add(Flatten())


model.add(Dense(700, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])




hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=15, verbose=1, validation_data=(X_test, y_test))
    
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable    
# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(15)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel(5)
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel(5)
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])




from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)

print(Y_pred)




y_pred = model.predict_classes(X_test)
print(y_pred)




plt.show()
p=model.predict_proba(X_test)




from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")






