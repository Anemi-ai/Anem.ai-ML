from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2  # Import OpenCV
import numpy as np

app = Flask(__name__)

dic = {
    0: ('Normal', 'Kadar hemoglobin darah Anda berada dalam kisaran normal.', 
        'Gejala anemia seperti Sakit kepala, Mudah lelah atau lemas tanpa alasan yang jelas, Pusing, Pucat, Detak jantung tidak teratur.'),
    1: ('Anemia', 'Anemia merupakan kondisi medis yang terjadi ketika jumlah sel darah merah dalam tubuh lebih rendah dari jumlah normal.', 
        'Gejala anemia seperti Sakit kepala, Mudah lelah atau lemas tanpa alasan yang jelas, Pusing, Pucat, Detak jantung tidak teratur.')
}

model = load_model('Model_MobileNet.h5')
model.make_predict_function()

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eyes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return len(eyes) > 0  # Return True if eyes are detected, otherwise False

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 224.0
    i = i.reshape(1, 224, 224, 3)
    probs = model.predict(i)[0]  # Probabilitas untuk semua kelas
    predicted_class = np.argmax(probs)  # Kelas dengan probabilitas tertinggi
    return dic[predicted_class], probs[predicted_class]  # Mengembalikan label dan probabilitas

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        
        # Detect eyes in the uploaded image
        if not detect_eyes(img_path):
            prediction = "Harap upload gambar mata anda."
            return render_template("classification.html", prediction=prediction, img_path=img_path)
        
        (result, description, gejala), confidence = predict_label(img_path)
        accuracy = f"{confidence * 100:.2f}%"
        prediction = {
            'result': result,
            'description': description,
            'gejala': gejala,
            'accuracy': accuracy
        }
        return render_template("classification.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
