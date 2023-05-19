#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import streamlit as st
from mtcnn import MTCNN
import os
from tensorflow.keras.models import load_model


# In[2]:


# Load the pre-trained BMI model
# set the path to the directory containing the images
path = "/Users/kajalshukla/Desktop/Quarter-3/ML & Predictive Analytics/ML_Final_Project/Final_Project"
model_path = os.path.join(path, 'bmi_pred_model_v2.h5')
bmi_model = load_model(model_path)


# In[3]:


# Define the face detector
detector = MTCNN()


# In[4]:


# Function to run the real-time camera
def run_realtime_camera(confidence_threshold, smoothing_frames):
    # Create a video capture object
    cap = cv2.VideoCapture(0)

    # Create an empty list for storing BMI values
    bmi_values = []
    
    # Create a Streamlit placeholder for the video
    video_placeholder = st.empty()

    # Start the video capture loop
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Iterate over the detected faces
        for face in faces:
            # Extract the face coordinates
            x, y, width, height = face['box']
            x1, y1 = x + width, y + height

            # Preprocess the face image
            face_img = frame[y:y1, x:x1]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.array(face_img) / 255.0

            # Get the predicted BMI value
            predicted_bmi = bmi_model.predict(face_img)

            # Store the predicted BMI value
            bmi_values.append(predicted_bmi[0][0])

            # Apply smoothing to stabilize BMI values
            if len(bmi_values) > smoothing_frames:
                smoothed_bmi = np.mean(bmi_values[-smoothing_frames:])
            else:
                smoothed_bmi = np.mean(bmi_values)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            # Display the predicted BMI value on the frame if confidence is above threshold
            if face['confidence'] > confidence_threshold:
                cv2.putText(frame, f'BMI: {smoothed_bmi:.2f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame in Streamlit
#         st.image(frame, channels="BGR", use_column_width=True)

        # Display the frame in Streamlit
        video_placeholder.image(frame, channels='BGR')

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()


# In[5]:


def main():
    # Create a Streamlit sidebar for configuration
    st.sidebar.title("BMI Detection Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    smoothing_frames = st.sidebar.slider("Smoothing Frames", 1, 30, 10)

    # Create a Streamlit main section for displaying the app
    st.title("BMI Detection App")

    # Call the function to run the real-time camera
    run_realtime_camera(confidence_threshold, smoothing_frames)
    


# In[6]:


if __name__ == "__main__":
    # Load the pre-trained BMI model
#     bmi_model = load_model(model_path)

    # Define the face detector
#     detector = MTCNN()

    # Run the Streamlit app
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




