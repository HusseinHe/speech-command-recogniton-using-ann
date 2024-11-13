# speech-command-recogniton-using-ann
### Objective:
1)Data Loading and Preprocessing:
-Audio files from a specified directory are loaded, and Mel Frequency Cepstral Coefficients (MFCCs) are extracted as features. These MFCCs are resized to 64x64 and normalized.
-Labels corresponding to each audio file are encoded into numerical values, and the dataset is split into training and validation sets.
2)Data Augmentation:
-Gaussian noise is added to the training data to augment the dataset, increasing model robustness and preventing overfitting.
3)Model Architecture:
-A Convolutional Neural Network (CNN) is constructed with three convolutional layers, each followed by batch normalization, max pooling, and dropout for regularization.
-The model concludes with a dense layer and softmax activation to output probabilities for each class.
4)Training:
-The model is trained using the Adam optimizer with sparse categorical cross-entropy loss. Early stopping and learning rate reduction callbacks are employed to optimize training.
-The model trains for a maximum of 15 epochs, with early stopping if validation loss does not improve after 5 epochs.
5)Evaluation:
-The trained model is evaluated on the validation data, and performance metrics such as accuracy, a classification report, and confusion matrix are computed.
-Accuracy and loss curves for both training and validation sets are plotted to visualize the model's performance over epochs.
### Data info:
i used google speech commands dataset(the link:https://www.kaggle.com/datasets/neehakurelli/google-speech-commands/code )
The dataset consists of audio files stored in subdirectories, where each subdirectory corresponds to a specific class or label. Each audio file represents a speech or sound sample, and the goal is to classify 
these 
samples into their respective categories.
#### The dataset is structured as follows:
Directory Structure:
-Each label has its own folder within the dataset path.
-The files within each folder represent the audio samples for that label.
File Format:
-Audio files are assumed to be in a format compatible with the librosa library (e.g., WAV, MP3).
### Preprocessing Steps:
1)Loading Audio Files:
-Audio files are loaded using the librosa library at a fixed sample rate of 16 kHz for consistency.
-Each audio file is processed individually by reading its content and extracting features.
2)Feature Extraction (MFCCs):
-For each audio file, Mel Frequency Cepstral Coefficients (MFCCs) are extracted. MFCCs are commonly used features in speech recognition tasks as they capture the power spectrum of sound and are effective in 
representing speech patterns.
-64 MFCC coefficients are computed for each audio file.
3)Resizing and Normalizing Features:
-The extracted MFCC features are resized to a consistent shape of 64x64 to standardize the input size for the model.
-The features are then normalized to a range between 0 and 1 by applying a min-max normalization technique: (mfccs - min) / (max - min).
4)Label Encoding:
-The labels, which are the folder names representing the classes, are encoded into numerical values using LabelEncoder from scikit-learn. This transforms string labels into integers that can be used for training 
a machine learning model.
5)Train-Test Split:
-The dataset is split into training and validation sets using train_test_split. The data is split with 80% for training and 20% for validation. The split is stratified to ensure that the distribution of labels is 
consistent between the training and validation sets.
6)Data Augmentation:
-To improve model generalization and prevent overfitting, Gaussian noise is added to the training data. This is done by generating random noise and scaling it with a factor, then adding it to the training samples.
7)Reshaping Data for CNN Input:
-The input data is reshaped to include a single color channel ([..., np.newaxis]), as required by the Convolutional Neural Network (CNN), which expects a 3D input shape (height, width, channels).
### some instructions:
1)Set Up the Environment:
-Before running the code, ensure that you have all the required dependencies installed. You can install the necessary libraries using pip by running the following commands in your terminal or command prompt(pip 
install numpy pandas librosa tensorflow scikit-learn seaborn matplotlib tqdm)
2)Prepare Your Dataset:
Ensure you have your dataset ready and stored in the specified directory (D:\college\speech recognition\assignments\ass1\data\archive). The dataset should have subdirectories, where each subdirectory represents a 
class label, and the files within those subdirectories are audio files (e.g., WAV or MP3).
3)Adjust the data_path Variable:
Make sure the data_path in the script points to the correct location of your dataset
4)Run the Code:
Now, you can run the script in your Python environment. Here’s the basic flow of execution:
-Load the Audio Files: The code will load all the audio files from the data_path directory and extract MFCC features for each file.
-Preprocess the Data: The features will be resized and normalized, and the labels will be encoded.
-Split the Data: The dataset will be split into training and validation sets.
-Apply Augmentation: Gaussian noise will be added to the training data to augment the dataset.
-Build and Train the Model: The CNN model will be built and trained using the training data, and the validation data will be used to monitor performance.
-Evaluate and Visualize: After training, the model’s performance will be evaluated, and accuracy/loss curves along with a confusion matrix will be displayed.
5)Monitor Training:
During training, you can observe the following:
-Early Stopping: If the validation loss does not improve for 5 consecutive epochs, training will stop early to prevent overfitting.
-Learning Rate Scheduling: The learning rate will be reduced if the validation loss plateaus for 3 epochs.
6)Examine Results:
-Once the training completes, the validation accuracy and classification report will be displayed.
-A confusion matrix will be plotted to visualize the classification results.
-Accuracy and Loss curves will also be shown to help you assess the model’s performance during training and validation.
7)Troubleshooting:
-Ensure that your dataset has sufficient samples per class. If one class has very few samples, consider either balancing the dataset or using techniques like oversampling or augmentation.
-Check the librosa loading function for any issues with unsupported audio formats or corrupted files.
### Dependencies:
To run the speech recognition code, you'll need to install the following Python libraries:
-NumPy: For numerical operations on arrays.
-Pandas: For data handling and manipulation.
-Librosa: For audio processing and feature extraction (MFCCs).
-TensorFlow: For building and training the neural network model.
-Scikit-learn: For machine learning utilities like dataset splitting, label encoding, and metrics.
-Seaborn: For creating visualizations like the confusion matrix.
-Matplotlib: For plotting graphs and visualizations.
-TQDM: For displaying a progress bar during the data loading process.
