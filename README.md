# anem.ai - Machine Learning 🤖
This repository contains code and files related to machine learning models for image classification using the MobileNet architecture. The model is drilled on a data set consisting of two classes: Anemic and Normal

## Machine Learning Roadmap
<p>
   <img src ="Images/ML Roadmap.png"/>
</p>

## Tools / Library / IDE
- Google Colaboratory
- Visual Studio Code
- TensorFlow
- Keras
- Scikit-learn
- Seaborn
- NumPy
- Matplotlib
- cv2
- Pandas

## Detection Model 
### Dataset
The dataset used for training and testing the model should be organized in the following directory structure:
This is a dataset containing conjunctiva images, consisting of 2 different classes. This dataset is used to train and test convolutional neural network models.

### Model Performance
<p>
   <img src ="Images/Loss.png"/>
</p>
<p>
   <img src ="Images/Accuracy.png"/>
</p>
<p>
   <img src ="Images/Performance.png"/>
</p>

## Chatbot Model 
### JSON
JSON (JavaScript Object Notation) is a commonly used format for data exchange between servers and clients.  JSON data is used to structure and manage the conversation, configuration, and response data of a chatbot. 

### Output
The chatbot model that has been created is able to answer questions with a context about anemia. .

# How To Use
## Detection Model
1. User takes a picture of an eye, through the anem.ai app
2. User selfie will be sent to ML Model in the Cloud and converted into an array form
3. Retrieve data and the model makes predictions based on the eye conjunctiva detected in the image, and returns an image with the user's eye conjunctiva analysis

## Chatbot Model
1. The user inputs the questions into the chatbot
2. The chatbot captures the input and starts processing it
3. The chatbot uses conditional logic to match input with predefined keywords or phrases to identify the user's intent
4. Returns the response and prompts the user to ask further questions
