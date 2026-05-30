# 🫁 Chest X-Ray Pneumonia Detection System

> A deep learning web app that analyzes chest X-ray images and detects **Pneumonia vs Normal** using a custom CNN built with PyTorch — deployed on Streamlit.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://xray-pnumonia-detection-system-ogqoghv577ltvxn8bscb7s.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20App%20Runner-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

---

## 🚀 Live Demo

👉 **[Open the App](https://xray-pnumonia-detection-system-ogqoghv577ltvxn8bscb7s.streamlit.app/)** *(may take a few seconds to load)*

---

## 📸 Screenshots

**Upload your X-Ray → Get instant AI diagnosis**

![Upload Screen] <img width="1912" height="923" alt="Screenshot 2026-05-30 145237" src="https://github.com/user-attachments/assets/887e627c-f86c-47a5-8512-180d60ca468c" />


![Diagnosis Result]<img width="1908" height="910" alt="Screenshot 2026-05-30 145259" src="https://github.com/user-attachments/assets/22567a46-46de-48f8-be84-989a5de5c05a" />


> Upload a chest X-ray (JPG/PNG) → the model returns a **Pneumonia / Normal** prediction with a confidence probability breakdown.

---

## 🧠 What It Does

Pneumonia is a serious lung infection affecting millions globally. Early and accurate detection from chest X-rays is critical for timely treatment.

This app:
- Accepts chest X-ray images (JPG / PNG)
- Runs inference through a **custom CNN model** trained with PyTorch
- Returns a **binary diagnosis** — `Pneumonia Detected` or `Normal`
- Shows a **probability breakdown** (e.g., Pneumonia: 73.1% / Normal: 26.9%)
- Includes a medical disclaimer — *for research & educational use only*

---

## 🏗️ How It Works

```
Chest X-Ray Image (uploaded by user)
            ↓
   Image Preprocessing (resize, normalize)
            ↓
   Custom CNN Model (PyTorch)
            ↓
   Binary Classification → Pneumonia / Normal
            ↓
   Streamlit UI → Confidence Score + Probability Breakdown
```

**Model Architecture:** Custom Convolutional Neural Network (CNN) — 2 output classes  
**Dataset:** Provided by Apollo Diagnostic Center for research purposes

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | Custom CNN model training & inference |
| Streamlit | Web app UI |
| AWS S3 | Model artifact storage |
| AWS App Runner | Cloud deployment |
| GitHub Actions | CI/CD pipeline |

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Deepak77-ai/-normal-or-pnumonia-detection-using-X_ray-imeges/
cd -normal-or-pnumonia-detection-using-X_ray-imeges
```

**2. Create & activate a virtual environment**
```bash
python -m venv my_env
my_env\Scripts\activate        # Windows
# source my_env/bin/activate   # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up AWS credentials** — create a `.env` file:
```env
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_DEFAULT_REGION=<your_region>
```

**5. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
X-Ray-Pneumonia-Detection-System/
├── app.py                  # Streamlit application
├── train.py                # Model training script
├── xray/
│   ├── components/         # Training pipeline components
│   ├── pipeline/           # Prediction pipeline
│   ├── ml/                 # Model architecture
│   ├── cloud_storage/      # AWS S3 integration
│   ├── entity/             # Config & artifact entities
│   └── constant/           # Project constants
├── notebook/               # EDA & experimentation
├── scripts/                # Utility scripts
├── bentofile.yaml          # BentoML config
└── requirements.txt
```

---

## ⚠️ Medical Disclaimer

> This tool is for **educational and research purposes only**. It is **not a substitute** for professional medical diagnosis. Always consult a licensed radiologist or physician for clinical decisions.

---

## 📬 Contact

Made by **Deepak** · [GitHub](https://github.com/Deepak77-ai)
