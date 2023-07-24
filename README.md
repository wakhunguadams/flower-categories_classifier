Flower Classification using Transfer Learning
Flower Classification

This project was completed as part of the AWS AI/ML Scholarship, and it aims to classify 102 different flower categories using transfer learning with a pre-trained deep learning model. Additionally, application modules have been created to allow users to interact with the model through the command line.

Overview
The project leverages the power of transfer learning to build an accurate flower classifier. Instead of training a deep neural network from scratch, we use a pre-trained model and fine-tune it to classify the flower categories. This approach saves time, computational resources, and yields better results. 

Data
The dataset used for training and validation contains images of 102 different flower categories. The images are organized into three subsets: training, validation, and test sets. These subsets are used to train the model, tune its hyperparameters, and evaluate its performance, respectively.

Transfer Learning
We employ the transfer learning technique by utilizing the VGG13 model, which is a pre-trained deep neural network that has been trained on a large dataset. We replace the classifier part of the VGG13 model with our own fully connected layers to adapt it to the specific flower classification task.

Command Line Application
The project includes two main command-line applications:

train.py: This script is used to train the flower classifier. It accepts various arguments such as the data directory, the architecture to use (default is VGG13), learning rate, hidden units, and the number of epochs. It also supports GPU training if available.

predict.py: This script is used to make predictions on single flower images. You can pass an image path and the trained model checkpoint as arguments, and it will display the top predicted flower classes along with their probabilities. The --top_k, --category_names, and --gpu options allow customization of the predictions.

Acknowledgments
This project was made possible by the AWS AI/ML Scholarship program and the valuable insights from the instructors and mentors.

Feel free to explore the code, and if you have any questions or suggestions, please don't hesitate to reach out. Happy classifying!
