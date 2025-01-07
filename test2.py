import re
import os
import cv2
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

# Paths to the model and label files
model_path = 'Models/EdgeTPU/model_edgetpu.tflite'
label_path = 'Models/EdgeTPU/labels.txt'

# Function to classify an image
def classify_image(interpreter, image):
    size = common.input_size(interpreter)
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    common.set_input(interpreter, resized_image)
    interpreter.invoke()
    return classify.get_classes(interpreter, top_k=1)

def main():
    try:
        # Load the model onto the TF Lite Interpreter
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        labels = read_label_file(label_path)
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image so it matches the training input
        frame = cv2.flip(frame, 1)

        # Classify the image
        results = classify_image(interpreter, frame)
        if results:
            label = labels[results[0].id]
            score = results[0].score
            # Display the classification result on the frame
            cv2.putText(frame, f'Label: {label}, Score: {score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
