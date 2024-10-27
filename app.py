from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import cv2
import os
import uuid
from ultralytics import YOLO
import easyocr
import json
from io import BytesIO
import base64

app = Flask(__name__)

# Initialize models and constants
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
UPLOAD_FOLDER = "./licenses_plates_imgs_detected/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

reader = easyocr.Reader(['en'], gpu=True)
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
vehicles = [2]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            text = result[1].upper()
            score = result[2]
            scores += score
            plate.append(text)

    if plate:
        return " ".join(plate), scores / len(plate)
    return None, None


def process_image(image_array):
    results = {}
    licenses_texts = []
    license_numbers = 0

    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    # Process vehicle detections
    if object_detections.boxes.cls.tolist():
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    # Process license plate detections
    license_plate_crops = []
    if license_detections.boxes.cls.tolist():
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]

            # Save crop
            img_name = f'{uuid.uuid1()}.jpg'
            crop_path = os.path.join(UPLOAD_FOLDER, img_name)
            cv2.imwrite(crop_path, license_plate_crop)

            # Process text
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            if license_plate_text and license_plate_text_score:
                licenses_texts.append(license_plate_text)
                license_plate_crops.append(base64.b64encode(cv2.imencode('.jpg', license_plate_crop)[1]).decode())

                results[license_numbers] = {
                    'car': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2],
                        'car_score': float(car_score)
                    },
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': float(score),
                        'text_score': float(license_plate_text_score)
                    }
                }
                license_numbers += 1

    # Convert the processed image to base64
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode()

    return {
        'processed_image': img_base64,
        'license_plates': license_plate_crops,
        'texts': licenses_texts,
        'details': results
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Read and process the image
        image = Image.open(file)
        image_array = np.array(image)
        results = process_image(image_array)
        return jsonify(results)

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)