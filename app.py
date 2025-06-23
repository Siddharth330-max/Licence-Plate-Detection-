from flask import Flask, request, render_template
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the ONNX model
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name  # dynamically get input name

# Define class names
CLASS_NAMES = ['license_plate']

# Preprocess input image
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)
    resized = cv2.resize(img, (640, 640))
    input_tensor = resized.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0) / 255.0
    return resized, input_tensor  # Use resized for drawing


# Postprocess outputs to extract boxes
def postprocess(outputs, image, conf_thresh=0.25):
    predictions = outputs[0][0]  # shape: (num_predictions, num_attrs)
    boxes, class_ids, scores = [], [], []

    for pred in predictions:
        objectness = pred[4]
        class_probs = pred[5:]
        class_id = int(np.argmax(class_probs))
        class_score = class_probs[class_id]
        confidence = objectness * class_score

        if confidence > conf_thresh:
            if class_id >= len(CLASS_NAMES):
                print(f"⚠️ Skipping class_id={class_id} — out of range")
                continue

            x_center, y_center, w, h = pred[0:4]
            h_img, w_img = image.shape[:2]

            # Rescale to original image size (model uses 640x640)
            x1 = int((x_center - w / 2) * w_img / 640)
            y1 = int((y_center - h / 2) * h_img / 640)
            x2 = int((x_center + w / 2) * w_img / 640)
            y2 = int((y_center + h / 2) * h_img / 640)

            boxes.append((x1, y1, x2, y2))
            class_ids.append(class_id)
            scores.append(confidence)

    return boxes, class_ids, scores




# Define route for frontend
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        orig_img, input_tensor = preprocess(filepath)
        print(f"📷 Original Image Shape: {orig_img.shape}")

        outputs = session.run(None, {input_name: input_tensor})
        print("📦 Model Output Shape:", outputs[0].shape)

        boxes, class_ids, scores = postprocess(outputs, orig_img)

        if boxes:
            print(f"✅ Found {len(boxes)} detections above threshold")
            for box, class_id, score in zip(boxes, class_ids, scores):
                label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
                x1, y1, x2, y2 = box
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            print("❌ No detections above confidence threshold.")

        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, orig_img)

        return render_template('index.html', result_img='uploads/result.jpg')

    return render_template('index.html', result_img=None)




if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
