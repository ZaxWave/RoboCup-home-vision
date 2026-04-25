# RoboCup Home Vision System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

A comprehensive computer vision system for RoboCup@Home competition, featuring advanced object detection, face recognition, and multi-object tracking capabilities. Built with state-of-the-art deep learning models and optimized for real-time performance.

## ✨ Features

### 🔍 Object Detection
- **YOLOv8/YOLOv11 Integration**: High-accuracy object detection with multiple model variants
- **Real-time Processing**: Optimized for live video streams
- **Custom Training**: Support for domain-specific model fine-tuning

### 👤 Face Recognition
- **YuNet Face Detection**: Lightweight and efficient face detection
- **SFace Recognition**: Robust face embedding extraction
- **Feature Matching**: Advanced face comparison algorithms

### 📊 Multi-Object Tracking
- **DeepSORT Algorithm**: Reliable object tracking with re-identification
- **Kalman Filtering**: Smooth trajectory prediction
- **RealSense Integration**: Depth-aware tracking capabilities

### 🔧 Point Cloud Processing
- **3D Data Handling**: Point cloud data acquisition and processing
- **Sensor Fusion**: Integration with depth sensors
- **Spatial Analysis**: Advanced 3D spatial reasoning

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ZaxWave/RoboCup-home-vision.git
   cd RoboCup-home-vision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models**
   ```bash
   # Download YOLO models
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt
   
   # Download face detection models
   # Models are included in the face_detect/model/ directory
   ```

## 📖 Usage

### Object Detection
```python
from object_detect.detect import ObjectDetector

detector = ObjectDetector(model_path='yolov8n.pt')
results = detector.detect(image)
```

### Face Recognition
```python
from face_detect.face import FaceRecognizer

recognizer = FaceRecognizer()
faces = recognizer.detect_faces(image)
embeddings = recognizer.extract_features(faces)
```

### Multi-Object Tracking
```python
from track.yolov8_deepsort_tracking.main import Tracker

tracker = Tracker()
tracked_objects = tracker.track(video_stream)
```

## 📁 Project Structure

```
RoboCup-home-vision/
├── face_detect/              # Face detection and recognition module
│   ├── face.py              # Main face recognition class
│   ├── face_detect_real_sense.py  # RealSense integration
│   ├── feature/             # Face feature data
│   └── model/               # Pre-trained face models
├── object_detect/           # Object detection module
│   ├── detect.py           # Object detection implementation
│   ├── snap_shot.py        # Image capture utilities
│   └── point_cloud_data.txt # Point cloud data
├── track/                   # Multi-object tracking
│   └── yolov8-deepsort-tracking/
│       ├── main.py         # Tracking main script
│       ├── main2.py        # Alternative tracking implementation
│       └── deep_sort/      # DeepSORT algorithm
├── yolo/                    # YOLO model training and inference
│   └── ultralytics-main/   # Ultralytics YOLO framework
├── runs/                    # Training results and logs
├── env.txt                  # Environment configuration
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## 🔧 Configuration

### Environment Setup
```bash
# Copy environment template
cp env.txt .env

# Edit configuration
nano .env
```

### Model Configuration
- YOLO models: `yolov8n.pt`, `yolov11n.pt`
- Face models: `face_detection_yunet.onnx`, `face_recognition_sface.onnx`
- Tracking config: `deep_sort/configs/deep_sort.yaml`

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure compatibility with Python 3.8+

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**ZaxWave** - [GitHub Profile](https://github.com/ZaxWave)

Project Link: [https://github.com/ZaxWave/RoboCup-home-vision](https://github.com/ZaxWave/RoboCup-home-vision)

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [DeepSORT](https://github.com/nwojke/deep_sort) - Multi-object tracking
- [YuNet](https://github.com/ShiqiYu/libfacedetection) - Face detection
- [SFace](https://github.com/opencv/opencv_zoo) - Face recognition

---

<div align="center">
  <p>Built with ❤️ for RoboCup@Home competition</p>
  <p>
    <a href="#overview">Overview</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>