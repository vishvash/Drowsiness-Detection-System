# Drowsiness-Detection-System

🚀 Excited to share my latest project on Drowsy Detection in Online Classes! 📚💻

In the educational context, a significant number of students struggle with staying awake during lecture videos, leading to missed concepts and academic challenges. To address this, I developed a real-time drowsiness detection system using YOLOv8 and deployed it through Streamlit. 😎 

Here's a glimpse into my project:
🔍 Data Acquisition:
Utilized the Faces Dataset including Indian Faces with more than 20,000 images stored in Google Drive and mounted in Google Colab.
🔧 Data Preprocessing:
Resized images to 640, normalized (image/255), and adjusted image orientation.
🎨 Data Augmentation with Roboflow:
Applied horizontal flips, rotations (-15° to +15°), and blur (up to 2.5px) for robust training.
🧠 Model Development:
Tested multiple models including MobileNet V2, EfficientNetB0, Inception, Exception, ResNet50, and yolov5 cls.
Achieved over 95% accuracy with yolov8 cls.
🚀 Model Deployment:
Deployed yolov8 cls through Streamlit for real-time web camera feeds.
📊 Decision Support System:
Provides reporting and alerts with timestamps to indicate if a person is drowsy or active at specific topics.

This project aims to maximize student engagement and improve academic performance by detecting drowsiness and making timely interventions to make the topics more interesting.
