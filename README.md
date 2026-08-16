# 🌩️ Tropical Cloud Cluster (TCC) Detection using CNN

This project is built as part of **ISRO's Bharatiya Antariksh Hackathon 2025** and focuses on detecting **Tropical Cloud Clusters (TCCs)** from simulated IR satellite images using deep learning. The model classifies images as containing TCC or not, extracts critical cloud features, and estimates the severity level of weather patterns like rain, storm, and cyclone.

---

## 🧠 Project Overview

- 📥 **Input**: 64×64 grayscale IR satellite images (simulated from Google + Python-generated)
- 🛠️ **Preprocessing**: Resize, normalize, apply temperature thresholding
- 🔍 **CNN Model**: Classifies each image as **TCC** or **NoTCC**
- 📊 **Feature Extraction**: Pixel count, brightness temperature (min, mean, median), center coordinates, radius, and estimated cloud-top height
- ⚠️ **Severity Estimation**: Categorizes TCCs as:
  - ☁️ Normal  
  - 🌧️ Rain  
  - 🌧️🌧️ Heavy Rain  
  - 🌪️ Storm  
  - 🌀 Cyclone  

