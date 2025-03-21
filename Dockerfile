# Sử dụng image Python 3.9 làm base image
FROM python:3.9-slim

# Cài đặt các phụ thuộc cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo một thư mục làm việc
WORKDIR /app

# Sao chép tất cả tệp trong thư mục hiện tại vào thư mục làm việc trong container
COPY . /app

# Cài đặt các thư viện Python từ tệp requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mở port 5000
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
