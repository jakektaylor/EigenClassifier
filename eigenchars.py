import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
import math
import random

''' This class provides all of the functionality required to preprocess and classify images from the Chinese MNIST dataset. It also provides 
    functions for displaying information related to the images in the dataset.
'''
class EigenClassifier:

    ''' Purpose: This is the constructor for the EigenClassifier class.
        Parameters: -dataset_path: The path to the "Chinese MNIST in CSV - Digit Recognizer" dataset (.csv file)
                    -num_pca_components: The number of "eigenchars" to use for classification
                    -threshold: Threshold applied to the pixel values of each image in the function preprocess_images(self, threshold)
    '''
    def __init__(self, dataset_path, num_pca_components, threshold):
        #Read the dataset in as a Pandas DataFrame.
        dataset = pd.read_csv(dataset_path)

        #Remove NaN and duplicate rows.
        dataset = dataset.dropna()
        dataset = dataset.drop_duplicates()
        
        #Create a matrix to store the input images.
        self.X = dataset.iloc[:, :4096].to_numpy()                  #Shape is (15000, 4096)
        #Create the label vector.
        self.y = dataset.iloc[:, 4096:]
        self.y = self.y.drop("label", axis=1).loc[:, "character"]
        self.y = self.y.to_numpy() 

        #Encode the label vector.                                              
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)           #Shape is (15000, )

        #Display the label encoding
        print({self.label_encoder.transform(np.array([c]))[0]: c for c in self.label_encoder.classes_})

        #Preprocess the images.
        self.preprocess_images(threshold)

        #Split the dataset into training and testing datasets.
        self.train_ids, self.test_ids = train_test_split(np.arange(self.X.shape[0]), test_size=0.2, shuffle=True)

        #Produce the eigenfaces.
        self.fit_pca(num_pca_components)

    ''' Purpose: The purpose of this method is to approximately center each of the character images in the dataset. It does so by only including in the 
        final image a bounding box whose corners are the minimum and maximum indices at which nonzero values were detected, in both the x and y directions.
        Parameters: threshold-The threshold to apply to the input images
    '''
    def preprocess_images(self, threshold):
        for i, image in enumerate(self.X):
            image[image < threshold] = 0
            image[image >= threshold] = 255
            image = image.reshape((64, 64))
            nonzero = np.nonzero(image)                                                                 #Determine the nonzero values in the image
            if(len(nonzero[0]) != 0):                                                                       #Account for some images having all zero pixels
                image = image[min(nonzero[0]): max(nonzero[0]) + 1, min(nonzero[1]): max(nonzero[1]) + 1]   #Remove any image points outside the bounding box of nonzero values
                image = cv.resize(image.astype('float32'), (64, 64), interpolation=cv.INTER_LINEAR)
                image = image.astype('int64')
            self.X[i] = image.ravel()
    
    ''' Purpose: This function can be used to display the distribution of the original dataset (before splitting), training dataset or testing dataset
        according to the labels of the instances in the dataset.
        Parameters: -dataset: Can be "training", "testing" or any other value. 
                        If it is "training", the distribution for the training dataset is displayed, if it is "testing", the distribution 
                        for the testing dataset is displayed, and if it is any othe value, the distribution for the dataset prior to splitting 
                        it into a training and testing dataset is displayed.
    '''
    def display_label_distribution(self, dataset=None):
        plt.figure(1, figsize=(8, 6))                                    #Create the figure.
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        #NOTE: In all cases, we use self.y as a Pandas Series "labels" to take advantage of useful built-in Series methods
        if dataset == "training":                                                        
            labels = pd.Series(self.y[self.train_ids])                   
            plt.title("Distrubution of Training Dataset", fontsize=18)
        elif dataset == "testing":
            labels = pd.Series(self.y[self.test_ids])                    
            plt.title("Distribution of Testing Dataset", fontsize=18)
        else:                                                        
            labels = pd.Series(self.y)                                   
            plt.title("Distrubution of Overall Dataset", fontsize=18)

        plt.xlabel("Label", fontsize=18)
        plt.ylabel("Number of Occurences", fontsize=18)
        plt.bar(labels.value_counts().index, labels.value_counts().values)
        plt.show()

    '''Purpose: This function displays a random set of 4 images with the given 'label'.
       Parameters: -label: An integer in the closed interval [0, 14] representing the encoded label of one of the 15 characters 
                    in the dataset
    '''
    def display_images(self, label):
        #Check to make sure that the 'label' argument is valid.
        if label < 0 or label > 14:
            print("Error: The label must be an integer in the interval [0, 14]")
            return

        #Take a random sample of images with the given label.
        images = self.X[np.where(self.y == label)[0], :]

        #Choose 4 random images with the given label and display them.
        random_ids = random.sample(list(np.arange(images.shape[0])), 4)
        images = images.reshape((images.shape[0], 64, 64))[random_ids, :, :]
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f"Label: {label}")
        axs[0, 0].imshow(images[0], cmap="gray")
        axs[0, 1].imshow(images[1], cmap="gray")
        axs[1, 0].imshow(images[2], cmap="gray")
        axs[1, 1].imshow(images[3], cmap="gray")
        plt.show()

    ''' Purpose: This function is responsible for applying PCA to the training images and creating the 'eigencharacters'. These are a set of
        'num_components' images that explain most of the variance in the original set of images of size N (num_components < N).
        Parameters: num_components-The number of principal components to use when applying PCA.
    '''
    def fit_pca(self, num_components):
        
        #Get the rows of self.X that correspond to the training images (training dataset)
        X_train = self.X[self.train_ids, :]

        #Compute the mean image and subtract it from each image in the training dataset
        self.mean_image = np.mean(X_train, axis=0)
        X_train = np.subtract(X_train, self.mean_image)

        #Transpose the matrix 'X_train' and compute the covariance matrix.
        X_train = X_train.T
        covariance = np.cov(X_train)

        #Perform PCA on the covariance matrix, using whitening to reduce the amount of correlation between adjacent pixel values. https://paperswithcode.com/method/pca-whitening
        pca = PCA(n_components=num_components, whiten=True)
        pca.fit(covariance)

        #Compute the matrix of eigencharacters NOTE: The eigenchars are all unit vectors as a result of applying PCA.
        eigenchars = None
        for component in pca.components_:
            if eigenchars is None:
                eigenchars = component.reshape((X_train.shape[0], 1))
            else:
                eigenchars = np.append(eigenchars, component.reshape((X_train.shape[0], 1)), axis=1)

        eigenchars = eigenchars.T                               #Shape is (num_components, 4096) 
        self.eigenchars = eigenchars            
        print(f"The shape of the eigencharacters matrix is: {self.eigenchars.shape}")

    ''' Purpose: This method display a certain number of eigencharacters in the matrix 'self.eigenchars' in order of decreasing explained variance.
        Parameters: -num_characters: An integer value in the interval [0, self.eigenchars.shape[0]] indicating the number of eigencharacters to display.
    '''
    def display_eigenchars(self, num_characters):
        #If an invalid 'num_characters' argument was passed, change its value to a valid number.
        if num_characters > self.eigenchars.shape[0]: num_characters = self.eigenchars.shape[0]             
        elif num_characters < 1: num_characters = 1                                                         

        pixel_scaler = MinMaxScaler(feature_range=(0, 255))                             #Used to scale the values in each eigencharacter image 
                                                                                        #to between 0 and 255 (since they are stored as unit vectors)

        for i in range(num_characters):
            eigenchar = self.eigenchars[i, :].reshape((64, 64))                         #Reshape the image into its actual 2D shape            
            eigenchar = pixel_scaler.fit_transform(eigenchar).astype(np.ubyte)          #Makes all values in the eigenchar between 0 and 255
            plt.title(f"Eigencharacter {i}")
            plt.imshow(eigenchar, cmap="gray")
            plt.show()
    
    ''' Purpose: The purpose of this function is to classify the images in the test set. It then also computes an accuracy value for the model by 
        comparing the predicted label of each of the test set images to the actual label. The accuracy is computed as the number of correct predictions
        divided by the total number of predictions. At the end, if 'show_summary' is True, this function loops over all images in the test set and displays 
        the labeled predicted image beside the labeled test set image.
        Parameters: -show_summary: (See Purpose section)
    '''
    def perform_testing(self, show_summary=False):
        X_train = np.subtract(self.X[self.train_ids, :], self.mean_image)       #The training images
        y_train = self.y[self.train_ids]                                        #The training labels.
        
        #Project the training dataset into the eigenface space.
        X_train_proj = self.eigenchars @ X_train.T                              
        X_train_proj = X_train_proj.T                                           #Shape is (n_samples (training), num_eigenchars)

        #Vector to show the index of the best matching image for each image in the testing dataset.
        matching_images = np.zeros((len(self.test_ids), ))
        y_pred = np.zeros((len(self.test_ids), ))                               #Vector to store the actual predicted labels for each test image.

        X_test = np.subtract(self.X[self.test_ids, :], self.mean_image)         #The testing images.
        y_test = self.y[self.test_ids]                                          #The testing labels.

        #Project the testing images into the eigenface space.
        X_test_proj = self.eigenchars @ X_test.T
        X_test_proj = X_test_proj.T                                             #Shape is (n_samples (testing), num_eigenchars)

        #For each of the projected test set images, compute the projected training set image that it is closest to.
        #The corresponding, non-projected, non-centered training set image is the non-projected, non-centered test set's predicted 'matching_image'.
        #The label of the 'matching_image' is the predicted label for the non-projected, non-centered test set image.
        for i in range(X_test_proj.shape[0]):
            min_distance = None
            for j in range(X_train_proj.shape[0]):
                distance = self.compute_distance(X_test_proj[i, :], X_train_proj[j, :])
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    matching_images[i] = j
                    y_pred[i] = y_train[j]

        #Compute the accuracy of the model. 
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        print(f"Testing Accuracy: {accuracy}")

        if show_summary:
            #Display a summary of the test image and corresponding predicted image. 
            i = 0
            #The program will end the summary if the user enters 'q' or if it has already traversed all test images.
            while input("Do you want to continue? ").lower() != 'q' and i < matching_images.shape[0]:
                fig, axs = plt.subplots(1, 2)
                label_image = np.add(X_test[i, :], self.mean_image).reshape((64, 64))       #The non-projected, non-centered test (label) image.
                j = int(matching_images[i])
                pred_image = np.add(X_train[j, :], self.mean_image).reshape((64, 64))       #The non-projected, non-centered matching image.
    
                #Display the predicted image on the left and the test (label) image on the right.
                axs[0].set_title(f"Predicted image with label: {y_train[j]}")
                axs[0].imshow(pred_image, cmap="gray")
                axs[1].set_title(f"Label image with label: {y_test[i]}")
                axs[1].imshow(label_image, cmap="gray")
                plt.show()
                i+=1

    #Helper function used to compute the distance between vectors in the testing function using the L2 norm.
    def compute_distance(self, x1, x2):
        return np.linalg.norm(np.subtract(x1, x2))

if __name__ == "__main__":
    DATASET_PATH = os.path.join(os.getcwd(), "chineseMNIST.csv")
    classifier = EigenClassifier(DATASET_PATH, 50, 20)
    classifier.perform_testing(False)
    
