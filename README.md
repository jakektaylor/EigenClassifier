# EigenClassifier
This project makes use of the Eigenfaces algorithm used in the paper [1] for classifying faces to classify images of 15 different chinese characters.

The dataset that was used in this project was the “Chinese MNIST in CSV – Digit Recognizer” dataset uploaded to Kaggle.com by the user “Fedesoriano”. The dataset consists of 15,000 grayscale images of handwritten Chinese characters. Each of the 15,000 image instances in the dataset consists of 4098 columns, where the first 4096 columns represent the pixel values of the 64x64 character image, the 4097th column represents the numerical value of the character, which is a digit, and the last column represents the character in plain text. There are 1000 images of each of the 15 characters in the dataset. In order to have a single label for each of the images in the dataset, each of the 15 unique plain text character labels was encoded as a unique integer in the interval [0, 14]. A summary of the label encoding is provided below.

{0: '一', 1: '七', 2: '万', 3: '三', 4: '九', 5: '二', 6: '五', 7: '亿', 8: '八', 9: '六', 10: '十', 
11: '千', 12: '四', 13: '百', 14: '零'}

In order to assess the accuracy of the model, the dataset was randomly split into a training set of 12,000 images and a test set of 3,000 images. Using 50 principal components, or "eigenchars", PCA whitening, centering the character in each image and setting all pixel values < 20 to 0 and >= 20 to 255, an optimal test accuracy of 0.966 was achieved. 

NOTE: Instead of using the "chineseMNIST.csv" dataset, one can use the "chineseMNIST_subset.csv" dataset, which includes 30 randomly chosen images of each of the 15 characters for a total of 450 images. This will make the program complete faster however the testing accuracy is reduced. The best testing accuracy achieved with this dataset so far is 0.86.

To run my code, which is included in the file "eigenchars.py", the following libraries 
and corresponding versions were used:

- Python (version 3.12.3)
- opencv-python (version 4.9.0.80)
- numpy (version 1.26.4)
- pandas (version 2.2.1)
- matplotlib (version 3.8.4)
- scikit-learn (version 1.4.2)

The program then offers the following functionality :

1. To simply perform the testing and view the accuracy of the model, one can do the
following: 
    1. Unzip the contents of "datasets.zip" into the current working directory.
    2. In the if statement block "if \_\_name__ == '\_\_main__'", ensure the variable "DATASET_PATH" has the correct path to one of the datasets I included. (It should be the correct path if all files included in the submission are in the same directory and you are currently in that directory)
    3. Run the program using "python eigenchars.py", when the file "eigenchars.py" is in the current working directory. 
    4. Optional:
        - One can change the number of principal components used for PCA by changing the value of the second parameter in the EigenClassifier constructor call on line 238. The optimal value I found was 50, as mentioned in the report.
        - One can change the pixel threshold applied to the image pixels in the
        classifier by changing the value of the third parameter in the EigenClassifier constructor call on line 238. The optimal value I found was 20 as mentioned in the report.
        - One can choose to display a summary at the end of the testing, showing the labeled predicted image and the labeled correct image side-by-side, for each of the test images by calling "classifier.perform_testing(True)" (i.e. pass the argument 'True' to the function instead of 'False')

2. Additional display functionality
    - To display the distribution of instances in the training dataset according to their label, one can call "classifier.display_label_distribution('training')" after the "classifier" has been created on line 238. To display the distribution of instances in the testing dataset according to their label, one can call "classifer.display_label_distribution('testing')" instead. Finally, to display the distribution of instances in the entire dataset according to their label, you can call "classifer.display_label_distribution(arg)", where 'arg' is anything other than 'training' or 'testing'.
     - To display images with a given label, you can call "classifier.display_images(label)", where 'label' is any integer in the interval [0, 14]. This should be done after the 'classifier' has been created on line 238.
     - To display a set of "eigenchars", you can call "classifer.display_eigenchars(num_characters)", where num_characters is the number of "eigenchars" you wish to be displayed. The "eigenchars" will be displayed in order of explained variance. This should be done after the 'classifier' has been created on line 238.

[1]: Turk, M. A. & Pentland A. P. (1991). Face Recognition Using Eigenfaces. https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf