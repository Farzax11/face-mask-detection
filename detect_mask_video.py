# Import necessary packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from screeninfo import get_monitors

# Load the pre-trained face detector model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the pre-trained mask detector model
maskNet = load_model("mask_detector.h5")

# Function to detect faces and predict masks
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and create a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding boxes fall within frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert from BGR to RGB, resize, and preprocess
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Get screen size (for full-screen display)
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Create a named window for full screen display
cv2.namedWindow('Video', cv2.WINDOW_FULLSCREEN)

# Loop over the frames from the video stream
while True:
    frame = vs.read()

    # Detect faces and predict mask status
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Loop over detected faces and display the mask prediction
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine the label and color for bounding box
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display label and bounding box on the frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Resize the frame to the screen resolution (full screen)
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    # Show the resized frame in full screen
    cv2.imshow("Video", frame_resized)

    # Exit loop if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
vs.stop()
