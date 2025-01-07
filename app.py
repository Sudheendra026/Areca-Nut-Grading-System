import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Models/FloatingPoint/model_unquant.tflite")
interpreter.allocate_tensors()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

features = [
    '''Bazar Karikoka represents a grade of arecanuts that is generally of the lowest quality among the mentioned grades. These arecanuts may have significant defects, poor visual appearance, or lower weight, making them less desirable for consumers. Bazar Karikoka arecanuts are often sold at a lower price point due to their inferior quality compared to other grades.''',
    '''Bazar Ulli is another grade of arecanuts that falls below Bazar Chali and Bazar Fator in terms of quality. These arecanuts may have more noticeable defects, irregular shapes, or lower weight compared to higher grades. Bazar Ulli arecanuts are typically considered to be of medium quality and are priced accordingly in the market.''',
    '''Bazar Chali is a grade of arecanuts that typically represents a higher quality compared to other grades. These arecanuts are usually characterized by their superior visual appearance, uniformity in size, and minimal defects or blemishes. Bazar Chali arecanuts are often preferred by consumers and command a higher market price due to their premium quality.''',
    '''Bazar Fator is a grade of arecanuts that may exhibit slightly lower quality characteristics compared to Bazar Chali. These arecanuts might have minor imperfections, discolorations, or irregularities that make them less desirable for consumers. Bazar Fator arecanuts are usually priced lower than Bazar Chali due to their perceived lower quality.'''
]

# Function to classify an image
def classify_image(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image to required size and scale
    input_shape = input_details[0]['shape']
    input_data = np.array(image.resize((input_shape[1], input_shape[2])), dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data / 127.5) - 1.0  # Normalize the image to [-1, 1]

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_label_id = np.argmax(output_data)
    label = labels[top_label_id]
    score = output_data[0][top_label_id]
    feature = features[top_label_id]
    return label, score, feature

# Gradio interface
def classify_interface(image):
    label, score, feature = classify_image(image)
    return label, score, feature

# Custom HTML for title
custom_html = """
<style>
    .image-container img {
        width: 300px !important;
        height: 300px !important;
        object-fit: contain !important;
    }
</style>
<div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">
    Areca Nut Classifier
</div>
<div style="text-align: center; margin-bottom: 20px;">
    Upload an image of an Areca nut to classify it into one of the four grades.
</div>
"""

# Create Gradio interface
iface = gr.Blocks()

with iface:
    gr.HTML(custom_html)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Areca Nut Image")
            classify_button = gr.Button("Classify")
        with gr.Column():
            label_output = gr.Textbox(label="Label")
            score_output = gr.Textbox(label="Score")
            feature_output = gr.Textbox(label="Description", lines=5)

    classify_button.click(classify_interface, inputs=image_input, outputs=[label_output, score_output, feature_output])

iface.launch(share=True)
