import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, jsonify
import base64
from PIL import Image, ImageDraw, ImageFont

# Initialize the Flask app
app = Flask(__name__)

# Load your pre-trained model
model = load_model('my_cnn_model.keras')

# Define image upload folder
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to preprocess image
def prepare_image(filepath):
    # Load the image
    img = image.load_img(filepath, target_size=(227, 170))  # Update size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize (if required by your model)
    return img_array

# Helper function to annotate the image with prediction results
def annotate_image(filepath, result, confidence):
    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Add text (update positions as needed)
    text = f"{result}\nConfidence:{confidence:.2f}%"
    draw.text((10, 10), text, fill="red", font=font)

    # Save the annotated image
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], os.path.basename(filepath))
    img.save(output_path)
    return output_path

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Convert the input image to a base64 string for rendering
        with open(file_path, "rb") as image_file:
            input_img_str = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the image for prediction
        img_array = prepare_image(file_path)

        # Predict using the model
        prediction = model.predict(img_array)
        #accuracy = float(prediction[0][0])*100
        # Get the predicted probability
        predicted_prob = float(prediction[0][0])

        # Determine the result and confidence
        if predicted_prob > 0.5:
           result = "Crack Detected"
           confidence = predicted_prob * 100  # Confidence for positive prediction
        else:
           result = "No Crack Detected"
           confidence = (1 - predicted_prob) * 100  # Confidence for negative prediction


        # Annotate the image with prediction results
        output_path = annotate_image(file_path, result, confidence)

        # Convert the output image to a base64 string
        with open(output_path, "rb") as output_file:
            output_img_str = base64.b64encode(output_file.read()).decode('utf-8')

        # Return the result with the input and output images
        return render_template(
            'index.html',
            input_img_str=input_img_str,
            output_img_str=output_img_str,
            result=result,
            confidence =f"{confidence:.2f}"  # Display accuracy with two decimal places
        )
    else:
        return jsonify({"error": "Invalid file format"}), 400


if __name__ == "__main__":
    app.run(debug=True)
