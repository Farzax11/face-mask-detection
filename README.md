# Face-Mask-Detection

This project is a real-time face mask detection system built using Flask, OpenCV, and TensorFlow. It uses a deep learning model to classify whether a person is wearing a mask or not and provides a live video feed with real-time detection.

Why I Built This Project:
During the COVID-19 pandemic, wearing a mask became essential for safety, but ensuring people followed this rule was a challenge. I created a smart solution that could detect face masks automatically using AI.
This project helps businesses and public places monitor mask compliance without needing manual checks. It uses OpenCV for face detection and a deep learning model to classify whether a person is wearing a mask or not.
By combining technology and safety, this system makes mask detection faster, easier, and more reliable. This is my way of using AI for a good cause! 😊
This project is not just about COVID-19—it’s about leveraging AI for public health, workplace safety, and environmental protection. 

Features
> Detects faces in real-time from webcam input
> Classifies each face as "Mask" or "No Mask"
> Displays a live video feed with detection results
> Provides real-time counts for masked and unmasked faces
> Simple Flask web interface for interaction

Tech Stack
Backend: Flask, TensorFlow/Keras
Frontend: HTML, CSS (Bootstrap), JavaScript
Computer Vision: OpenCV (Haarcascade for face detection)
Model: Pre-trained deep learning model for mask detection

Project Structure
📁 Face-Mask-Detection
│── 📁 static/                     # Static assets (CSS, JS, images)
│── 📁 templates/                   # HTML templates (index.html)
│── 📁 models/                      # Trained model file (mask_detector.h5)
│── detect.py                       # Face mask detection script
│── app.py                           # Flask application
│── requirements.txt                 # Required Python libraries
│── README.md                        # Project documentation

Installation & Setup

1️⃣ Clone the repository:
git clone https://github.com/your-username/Face-Mask-Detection.git
cd Face-Mask-Detection
2️⃣ Install dependencies:
pip install -r requirements.txt
3️⃣ Run the Flask App:
python app.py
4️⃣ Open the Web App:
After running the app, open http://127.0.0.1:5000/ in your browser.

Screenshots:

Model Details
The model is trained using MobileNetV2 for feature extraction.
Dataset: The model was trained on a dataset containing masked and unmasked face images.
Training script: Located in train.py (not included in this repo).

Limitations
Works best in good lighting conditions.
May struggle with partial face occlusion or low-quality webcam feeds.
Haarcascade-based face detection is not perfect for all angles and distances.

Contributing
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

License
This project is licensed under the MIT License. You are free to modify and distribute it as per the license terms.

Acknowledgments
OpenCV for face detection
TensorFlow/Keras for deep learning
Bootstrap for UI styling
Special thanks to contributors and the open-source community

Add this to GitHub:
Once your project is ready, push it to GitHub

AND ITS DONE:
feel free to mail me for any doubts or queries here:sarahx11634@gmail.com
