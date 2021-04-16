import cv2
import numpy as np
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt

def load_data(data_path, img_shape):
    data_dir_list = os.listdir(data_path)
    img_rows=img_shape[0]
    img_cols=img_shape[1]
    num_channel=img_shape[2]

    labels=[]
    data=[]
    len_list_img=0
    num_class=0
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)    
        for img in img_list:
            input_img = cv2.imread(data_path + '/'+ dataset + '/'+ img)
            data.append(input_img)
            labels.append(num_class)
            
        num_class+=1

    data = np.array(data)
    labels=np.array(labels)

    return data, labels

def proximity_testing(dataset_name="legumes", num_classes=10, nb_bins=8):
    img_shape = [224, 224, 3]
    train_data_path = 'Datasets/' + dataset_name + 'A'
    X_train, y_train = load_data(train_data_path, img_shape)
    test_data_path = 'Datasets/' + dataset_name + 'B'
    X_test, y_test = load_data(test_data_path, img_shape)
    print("")
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    predictions = []
    real_values = []

    hist_train = []

    for index, test_image in enumerate(X_test):
        differences = []
        for train_image in X_train:
            hist_train = cv2.calcHist([train_image], [0, 1, 2], None, [nb_bins, nb_bins, nb_bins], [0, 256, 0, 256, 0, 256])
            hist_train = cv2.normalize(hist_train, hist_train).flatten()

            hist_test = cv2.calcHist([test_image], [0, 1, 2], None, [nb_bins, nb_bins, nb_bins], [0, 256, 0, 256, 0, 256])
            hist_test = cv2.normalize(hist_test, hist_test).flatten()

            image_difference = cv2.compareHist(hist_train, hist_test, cv2.HISTCMP_CHISQR)
            
            differences.append(image_difference)

        predictions.append(np.argmin(differences))
        real_values.append(np.argmax(y_test[index]))

    print("Dataset: ", dataset_name)
    print(predictions)
    print(real_values)
    matches = 0
    total_distance = 0
    for i in range(len(predictions)):
        if predictions[i] == real_values[i]:
            matches += 1
        else:
            total_distance += abs(predictions[i] - real_values[i])

    accuracy = matches/y_test.shape[0]
    print("accuracy:", accuracy)
    print("total distance:", total_distance)

if __name__ == '__main__':
    
    nb_bins = 8
    
    proximity_testing("cuisine", 9, nb_bins)
    
    proximity_testing("legumes", 10, nb_bins)

    proximity_testing("magasin", 12, nb_bins)

    proximity_testing("neige", 10, nb_bins)

    proximity_testing("studio", 10, nb_bins)

    proximity_testing("visages", 10, nb_bins)

    proximity_testing("parc", 10, nb_bins)