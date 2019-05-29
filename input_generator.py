import csv
import cv2
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_samples(data_folder):
    """
    This function returns the tuple train_samples, validation_samples
    """
    # Open the csv with the data
    with open(os.path.join(data_folder, 'driving_log.csv'), 'r') as f:
        reader = csv.reader(f)
        # Skip title row
        next(reader)
        samples = list(reader)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples        


def generator(data_folder, samples, batch_size=32):
    """
    This function returns a generator based on the samples
    (so they are not loaded into memory)
    """
    num_samples = len(samples)
    adjust_angle = 0.2

    # Loop forever so the generator never terminates
    while True:
        random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                is_center = False

                # Pick between center, left or right
                left_center_right = random.random()
                if left_center_right < 0.3:
                    # Left
                    name = os.path.join(data_folder, 'IMG', batch_sample[1].split('/')[-1])
                    angle += adjust_angle
                elif left_center_right > 0.7:
                    # Right
                    name = os.path.join(data_folder, 'IMG', batch_sample[2].split('/')[-1])
                    angle -= adjust_angle
                else:
                    name = os.path.join(data_folder, 'IMG', batch_sample[0].split('/')[-1])
                    is_center = True

                if not os.path.exists(name):
                    print(name)

                image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

                # Flip some of the time for center images
                if is_center and random.random() < 0.5:
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)
                else:
                    images.append(image)
                    angles.append(angle)

            # trim image to only see section with road
            features = np.array(images)
            labels = np.array(angles)
            yield shuffle(features, labels)