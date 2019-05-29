import math
import matplotlib.pyplot as plt

from input_generator import load_samples, generator
from network import build_model

######################
# Train the model
######################
# Set our batch size
batch_size = 32

data_folder = '/home/jnd/Downloads/data/'
# Get the samples
train_samples, validation_samples = load_samples(data_folder)

# Get the generators
train_generator = generator(data_folder, train_samples, batch_size=batch_size)
validation_generator = generator(data_folder, validation_samples, batch_size=batch_size)

# Build the model
model = build_model('nvidia')

# Compile and train the model using the generator function
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=math.ceil(len(train_samples) / batch_size),
                                     validation_data=validation_generator, 
                                     validation_steps=math.ceil(len(validation_samples) / batch_size),
                                     epochs=8, verbose=1)

# Finally save the model
model.save('model.h5')

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
