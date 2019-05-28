import csv
import cv2
import random
import numpy as np
import sklearn


def load_samples(csv_file):
    """
    This function returns the tuple train_samples, validation_samples
    """
    # Open the csv with the data
    with open('signnames.csv', 'r') as f:
        samples = list(csv.reader(f))
        
    train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples        


def generator(samples, batch_size=32):
    """
    This function returns a generator based on the samples
    (so they are not loaded into memory)
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                
                # Augment data 50% of the time
                if random.random() < 0.5:
                    images.append(center_image)
                    angles.append(center_angle)
                else:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(-center_angle)

            # trim image to only see section with road
            features = np.array(images)
            labels = np.array(angles)
            yield sklearn.utils.shuffle(features, labels)        