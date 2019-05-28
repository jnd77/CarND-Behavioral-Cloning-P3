import tensorflow as tf
import keras

from input_generator import load_samples, generator

# Load the samples
train_samples, validation_samples = load_samples('./driving_log.csv')

######################
# Build the model based on InceptionV3
######################
from keras.applications.inception_v3 import InceptionV3

# Load the model without the top
inception = InceptionV3(weights='imagenet', include_top=False, 
                        input_shape=(input_size,input_size,3))

# Freeze the weights
for layer in inception.layers:
    layer.trainable = False

# Create the input 
raw_input = keras.layers.Input(shape=(160, 320, 3))

# Crops the input
cropped_input = keras.layers.Cropping2D(cropping=((50,20), (0,0)))(raw_input)

# Preprocess the input
preprocessed_input = keras.layers.Lambda(lambda x: x/127.5 - 1.)(cropped_input)

# Apply the inception model
inception_output = inception(preprocessed_input)

# Add the layers at the end
x = keras.layers.GlobalAveragePooling2D()(inception_output)
x = keras.layers.Dense(512, activation = 'relu')(x)
predictions = keras.layers.Dense(1)(x)

# Finally build the model
model = keras.models.Model(inputs=raw_input, outputs=predictions)

######################
# Train the model
######################
# Set our batch size
batch_size=32

# Get the generators
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Compile and train the model using the generator function
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=ceil(len(train_samples) / batch_size), 
                                     validation_data=validation_generator, 
                                     validation_steps=ceil(len(validation_samples) / batch_size),
                                     epochs=5, verbose=1)

# Finally save the model
model.save('model.h5')

### Plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()