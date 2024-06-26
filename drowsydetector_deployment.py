# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:28:07 2024

@author: Lenovo
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the trained model
model_path = r"C:\Users\Lenovo\Downloads\Internship_proj_1\best_hypertuned.pt"
model = YOLO(model_path)

# Function to classify an image with probabilities for classes
# def classify_image(image):
#     results = model.predict(source=image, conf=0.25)
#     # labels = results[0].names
#     # label = labels[np.argmax(results[0].probs)]
#     probabilities = results[0].probs
    
#     # Example class labels
#     class_labels = ['Active', 'Drowsy']
    
#     # Convert tensor to list for easy processing
#     prob_list = probabilities.data.tolist()
    
#     # Create a dictionary to map class labels to their respective probabilities
#     prob_dict = {class_labels[i]: prob_list[i] for i in range(len(class_labels))}
    
#     # Format the output
#     label = ", ".join([f"{label} {prob:.2f}" for label, prob in prob_dict.items()])
#     return label

# Function to classify an image
def classify_image(image):
    results = model.predict(source=image, conf=0.25)
    probabilities = results[0].probs
    
    # Example class labels
    class_labels = ['Active', 'Drowsy']
    
    # Convert tensor to list for easy processing
    prob_list = probabilities.data.tolist()
    
    # Create a dictionary to map class labels to their respective probabilities
    prob_dict = {class_labels[i]: prob_list[i] for i in range(len(class_labels))}
    
    # Find the class with the highest probability
    max_prob_label = max(prob_dict, key=prob_dict.get)
    max_prob = prob_dict[max_prob_label]
    
    # Return the class name based on the probability
    if max_prob > 0.50:
        return max_prob_label
    elif prob_dict['Active'] == 0.50 and prob_dict['Drowsy'] == 0.50:
        return 'Active'
    else:
        return 'Active' if prob_dict['Active'] >= prob_dict['Drowsy'] else 'Drowsy'


# Streamlit App
st.title("Active vs Drowsy Classification")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    st.write("Classifying...")
    label = classify_image(image_cv)
    
    st.write(f"Prediction: **{label}**")


placeholder = st.empty()
frame_image = None


log_file_path = "webcam_logs.txt"


def save_logs(log_entries):
    with open(log_file_path, 'w') as log_file:
        log_file.writelines(log_entries)
        
# Live webcam feed classification
if st.button('Classify from Webcam', key='classify_webcam'):
    st.write("Starting webcam...")

    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.write("Error: Could not open webcam.")
    else:
        st.write("Press 'Stop' to stop the webcam.")
        placeholder = st.empty()  # Placeholder for video frames

        # stop_button = st.button('Stop', key='stop_webcam')
        
        # Initialize stop button state in session state
        if 'stop_button_pressed' not in st.session_state:
            st.session_state.stop_button_pressed = False

        # Create the stop button once outside the loop
        if st.button('Stop', key='stop_webcam'):
            st.session_state.stop_button_pressed = True
            st.write("Webcam stopped.")

        log_entries = []

        while not st.session_state.stop_button_pressed:
            ret, frame = video_capture.read()
            if not ret:
                st.write("Failed to capture image from webcam.")
                break

            label = classify_image(frame)
            
            # Log format
            log_entry = f"0: 416x416 {label}, {video_capture.get(cv2.CAP_PROP_POS_MSEC):.1f}ms\n"
            
            # Append log entry to list
            log_entries.append(log_entry)

            # Annotate the frame with the prediction label
            cv2.rectangle(frame, (0, 0), (90, 45), (255, 0, 0), -1)  # Blue background
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert the frame to displayable format in Streamlit
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_image = buffer.tobytes()

            # Display the frame
            placeholder.image(frame_image, use_column_width=True)
            
            save_logs(log_entries)
    
        # Release the webcam
        video_capture.release()




# Option to download log file
# if st.button('Download Log File', key='download_logs'):
#     save_logs(log_entries)
#     st.markdown(f"### [Download Log File](./{log_file_path})")