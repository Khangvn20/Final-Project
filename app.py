from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tắt các thông báo về thông tin (info) và cảnh báo (warning)
from tensorflow.keras.applications import ResNet50, resnet50


# Tải mô hình ResNet50 với trọng số ImageNet
model = ResNet50(weights='imagenet')

app = Flask(__name__)

# Phân tích cảm xúc nhận xét khách hàng sử dụng Hugging Face API
HF_API_TOKEN = "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"  # Make sure this is before its usage
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# Chuyển đổi hình ảnh thành dạng mà mô hình có thể hiểu
def prepare_image(img):
    img = img.resize((224, 224))  # Đổi kích thước ảnh thành 224x224
    img_array = np.array(img)  # Chuyển hình ảnh thành mảng numpy
    img_array = resnet50.preprocess_input(img_array)  # Tiền xử lý ảnh cho ResNet50
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch (một batch ảnh)
    return img_array

def analyze_sentiment(review_text):
    api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    response = requests.post(api_url, headers=headers, json={"inputs": review_text})

    # Kiểm tra nếu phản hồi từ API thành công
    if response.status_code == 200:
        result = response.json()  # Trích xuất dữ liệu JSON từ đối tượng Response

        # Chuyển đổi nhãn thành cảm xúc dễ hiểu
        label_map = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }

        # Đổi nhãn thành các giá trị cảm xúc
        for item in result[0]:
            item['label'] = label_map.get(item['label'], item['label'])  # Chuyển nhãn thành cảm xúc dễ hiểu

        print("Processed Hugging Face API response:", result)  # In kết quả để kiểm tra
        return result  # Trả về kết quả trực tiếp dưới dạng dữ liệu JSON

    else:
        print(f"API error: {response.status_code} - {response.text}")
        return {"error": f"API error: {response.status_code} - {response.text}"}  # Trả về lỗi dưới dạng JSON


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Mở ảnh và xử lý tiền xử lý ảnh
        img = Image.open(file.stream)  # Mở ảnh từ luồng dữ liệu
        img_array = prepare_image(img)  # Tiền xử lý ảnh
        predictions = model.predict(img_array)  # Dự đoán với mô hình

        # Giải mã kết quả dự đoán
        decoded_predictions = resnet50.decode_predictions(predictions, top=3)[0]

        # Tạo kết quả trả về dưới dạng JSON
        result = []
        for i in decoded_predictions:
            result.append({"label": i[1], "probability": float(i[2])})

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route mới cho phân tích cảm xúc nhận xét khách hàng
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_from_form():
    review_text = request.form.get('review_text')  # Lấy nhận xét từ người dùng
    
    if review_text:
        sentiment = analyze_sentiment(review_text)
        return jsonify(sentiment)
    else:
        return jsonify({"error": "Nhận xét không được để trống"}), 400

if __name__ == "__main__":
 app.run(debug=True, host='0.0.0.0', port=5000)