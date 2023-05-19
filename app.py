#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from PIL import Image


# In[4]:

# set the path to the directory containing the images
path = "/Users/kajalshukla/Desktop/Quarter-3/ML & Predictive Analytics/ML_Final_Project/Final_Project"
model_path = os.path.join(path, 'bmi_pred_model_v2.h5')
model = load_model(model_path)


# In[5]:

img_dir = '/Users/kajalshukla/Desktop/Quarter-3/ML & Predictive Analytics/ML_Final_Project/BMI/Data/Images/'

# img_dir = '/Users/kajalshukla/Desktop/'

# In[6]:
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# In[7]:
def main():
    st.title("Real-time BMI Detection")

    # File Uploader
    st.subheader("File Uploader")
    uploaded_file = st.file_uploader("Please choose a file")

    # Display uploaded file and predict BMI
    if uploaded_file is not None:
        st.subheader(uploaded_file.name)
        st.image(uploaded_file)
        
        image_path = os.path.join(img_dir, uploaded_file.name)

        # Preprocess the image
        processed_img = preprocess_image(image_path)
    
        # Predict BMI
        predicted_bmi = model.predict(processed_img)

        # Display predicted BMI
        st.write("The predicted BMI is:", predicted_bmi[0][0])
        

if __name__ == '__main__':
    main()



