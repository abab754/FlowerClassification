Flower Classification Using ResNet-50
This project focuses on classifying images of flowers into one of 102 different categories using a pre-trained ResNet-50 deep learning model. The model is trained on a dataset that includes images of flowers, along with segmentation images and distance matrices. The primary goal is to accurately classify unseen flower images using transfer learning from a pre-trained model.

Project Overview
The project uses the following components:

ResNet-50: A pre-trained deep convolutional neural network from the ImageNet dataset.
102 Flower Species Dataset: Contains images of 102 different types of flowers, along with image labels, segmentation images, and distance matrices.
PyTorch: The deep learning framework used for model development and training.
The model achieves a final test accuracy of 86.7% on the dataset.

Dataset
The dataset used in this project can be found here. It contains:

102 different categories of flowers.
A minimum of 40 images per category.
Image labels (imagelabels.mat) and dataset splits for training, validation, and test sets (setid.mat).
Distance matrices (distancematrices102.mat) that provide pairwise distances between images based on features like color, texture, etc.
Segmentation images (102segmentations.tgz).
Dataset Structure
Once the dataset is organized, the directory structure looks like this:

bash
Copy code
flowers/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ... 
    ├── val/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ... 
    └── test/
        ├── class_1/
        ├── class_2/
        └── ...
Installation
To run this project on your local machine, follow the instructions below:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/flower-classification.git
cd flower-classification
Install the required Python packages: It is recommended to set up a virtual environment before installing the dependencies.

bash
Copy code
pip install -r requirements.txt
The main dependencies include:

torch
torchvision
scipy
numpy
Download and Prepare the Dataset:

Download the 102 Flower Category dataset.
Extract the images into a jpg folder.
Run the images.py script to organize the dataset into train, val, and test directories:
bash
Copy code
python images.py
Code Overview
1. load_mats.py
This script loads the dataset split information and the flower labels from the .mat files. It prints out the number of training, validation, and test samples, and is used for verification purposes.

2. load_data.py
This script defines the data transformations for training and validation, using the ImageFolder dataset structure. It also sets up the data loaders for training and validation.

3. images.py
This script organizes the images into train, val, and test directories based on the dataset splits provided in the .mat files.

4. training.py
This script trains the ResNet-50 model using the flower dataset. Key features include:

Transfer learning: Using a pre-trained ResNet-50 model, with the final fully connected layer replaced for 102 classes.
Data Augmentation: Random resizing, cropping, and horizontal flipping to make the model robust to variations in the data.
Learning Rate Scheduler: The learning rate is reduced after a certain number of epochs to improve convergence.
At the end of training, the model is saved as resnet50_flowers.pth.

5. Testing the Model
After training, the model is evaluated on the test dataset. The script outputs the test loss and accuracy, providing a measure of the model’s generalization performance.

Distance Matrices
Although the current model does not use the distance matrices, they can be integrated into the training pipeline as additional features to improve accuracy in future iterations of the project.
How to Train the Model
To train the model, simply run the training.py script:

bash
Copy code
python training.py
This will train the ResNet-50 model on the flower dataset for 10 epochs and save the trained model to a .pth file.

How to Test the Model
Once the model is trained, you can test it on the test dataset by running the following:

bash
Copy code
python training.py  # This will also include the test evaluation after training
The script will output the test loss and accuracy, giving you insights into the model’s performance on unseen data.

Results
The model achieves the following performance:

Training Accuracy: ~93.5%
Validation Accuracy: ~89.6%
Test Accuracy: 86.7%
These results indicate that the model generalizes well to unseen data, making it a robust solution for flower classification tasks.

Future Work
Integrating Distance Matrices: In future iterations, we could experiment with using the provided distance matrices as additional input features for the model. This could help capture similarities between flowers more effectively.
Fine-tuning Hyperparameters: Further fine-tuning of the model, including exploring different optimizers and learning rates, could lead to better accuracy.
