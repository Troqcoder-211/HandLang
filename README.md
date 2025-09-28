### 🚀 Yêu cầu hệ thống

- Windows 10/11

- Python 3.10.x (khuyên dùng)

- Đã cài Visual Studio Code hoặc IDE bất kỳ

- Webcam hoạt động

### Cài đặt môi trường 
```bash 
# Clone repo
git clone https://github.com/Troqcoder-211/HandLang.git
cd HandLang

# Tạo virtual environment
python -m venv venv

# Kích hoạt venv (Windows PowerShell)
venv\Scripts\activate

# Cập nhật pip
python -m pip install --upgrade pip

# Cài đặt thư viện
pip install -r requirements.txt

```

### Kiểm tra môi trường 
```bash 
python test_env.py
```

### Kiểm tra camera 
```bash 
python test_camera.py
```

### Chạy demo Mediapipe Hand Tracking
```bash 
python test_mediapipe.py
```


### Chạy ứng dụng chính 
```bash 
python main.py
```