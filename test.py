import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Load pre-trained facial expression recognition model
# (Assuming you have a pre-trained model stored as 'model.h5')
model = load_model('model.h5')

# Define class labels for facial expressions
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_expression(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection using a pre-trained cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Extract the region of interest (face)

        # Resize face ROI to match the input size of the model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Normalize the face ROI
        face_roi = face_roi / 255.0

        # Predict facial expression using the pre-trained model
        expression_probs = model.predict(face_roi)[0]
        predicted_expression = class_labels[np.argmax(expression_probs)]

        # Draw bounding box and predicted expression label on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, predicted_expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def main():
    st.title("Facial Expression Recognition")
    st.write("Upload an image or use the webcam to detect and classify facial expressions.")

    st.sidebar.title("Input Options")
    input_option = st.sidebar.selectbox(
        "Select Input Option",
        ("Upload Image", "Use Webcam")
    )

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image.convert("RGB"))
            image = detect_expression(image)
            st.image(image, channels="BGR", caption="Processed Image")

    elif input_option == "Use Webcam":
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            frame = detect_expression(frame)
            st.image(frame, channels="BGR", caption="Facial Expression Recognition")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
