# Neuro-Assistive Hardware (Raspberry Pi Zero 2W)

This folder contains the complete software stack for the wearable device.

## Setup Instructions

### 1. Requirements
- Raspberry Pi Zero 2W (or Pi 3/4)
- Pi Camera Module
- USB Microphone
- Speaker (3.5mm or USB)
- OLED Display (SSD1306 I2C)
- Haptic Motor (GPIO 17)
- Button (GPIO 27)

### 2. Installation on Pi

1. **Install OS**: Flash Raspberry Pi OS (Legacy/Buster recommended for camera stack or Bullseye with libcamera).
2. **Enable Interfaces**: Run `sudo raspi-config`
   - Enable Camera
   - Enable I2C (for OLED)
   - Enable SSH

3. **Install System Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip python3-opencv libatlas-base-dev espeak
   sudo apt-get install libopenjp2-7 libtiff5
   ```

4. **Install Python Libraries**:
   ```bash
   pip3 install opencv-python-headless face-recognition numpy pickle-mixin
   pip3 install tflite-runtime # Find specific wheel for your Pi version
   pip3 install luma.oled luma.core
   pip3 install vosk sounddevice pyttsx3
   pip3 install requests gpiozero RPi.GPIO
   ```
   *Note*: `face_recognition` and `dlib` can be slow to compile on Pi Zero. Use pre-built wheels if available (piwheels.org).

### 3. Deploying Models

1. **Train on PC**: 
   - Run `python hardware/ai_models/personal_fine_tuning.py` on your PC first to generate `models/face_recognition/known_faces.pkl` from the uploaded images.
   
2. **Copy to Pi**:
   Transfer the entire `hardware` folder to the Pi:
   ```bash
   scp -r hardware/ pi@<pi-ip-address>:/home/pi/neuro_assist/
   ```

3. **Download Base Models**:
   - Download `vosk-model-small-en-us-0.15` and extract to `hardware/models/`.
   - Ensure `emotion_model.tflite` is in `hardware/models/`.

### 4. Running the System

On the Raspberry Pi:
```bash
cd /home/pi/neuro_assist/hardware
python3 main.py
```

## Features
- **Face Recognition**: Detects Patient, Caregiver, Known Persons.
- **Emotion Analysis**: Checks for distress/pain.
- **Assistive AI**: Voice commands ("Help", "Time", "Where am I").
- **Alerts**: Sends alerts to backend if Safe Zone violated or Distress detected.
- **OLED**: visual feedback.

## Configuration
Edit `config.py` to change:
- API URL (IP of your backend server)
- Pins (Haptic, Button)
- Camera Rotation
