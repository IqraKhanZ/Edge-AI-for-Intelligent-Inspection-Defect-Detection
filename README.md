# 🛩️ AeroLens Sentinel — Edge AI for Intelligent Inspection & Defect Detection

<p align="center">
  <img src="https://img.shields.io/badge/Status-Live%20on%20Render-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-YOLOv8%20ONNX-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Edge-Jetson%20Nano-red?style=for-the-badge" />
</p>

> **Autonomous Edge-Vision Defect Triangulation System for Aerospace MRO** — Dual Handheld / Drone Core

---

## 🔗 Live Deployment
**🌐 Deployed URL:** [https://edge-ai-for-intelligent-inspection-kxuh.onrender.com/](https://edge-ai-for-intelligent-inspection-kxuh.onrender.com/)

> Login with: **Username:** `admin` | **Password:** `admin`

---

## 📁 Project Drive (Dataset, Model Weights & Media)
**📂 Google Drive:** [https://drive.google.com/drive/folders/16QQFn-YHZ-_ild8pCnwd4QXvGdFMJe4w?usp=sharing](https://drive.google.com/drive/folders/16QQFn-YHZ-_ild8pCnwd4QXvGdFMJe4w?usp=sharing)

---

## 🚨 Problem Statement

Modern aerospace Maintenance, Repair & Overhaul (MRO) operations rely heavily on **manual visual inspection** of aircraft surfaces for structural defects such as cracks, corrosion, dents, and paint peel. This process is:

- **Slow and Labor-Intensive**: A single aircraft inspection can take 12–48 hours with multiple technicians.
- **Human Error-Prone**: Fatigue and inconsistency lead to missed defects that pose critical safety risks.
- **Expensive**: Grounded aircraft cost airlines an average of $150,000 per hour in lost revenue.
- **Not Scalable**: Inspections cannot keep pace with growing global aircraft fleets.
- **Cloud-Dependent**: Traditional AI inspection systems require internet connectivity, making them unsuitable for remote airstrips and offline hangar environments.

---

## 💡 Solution

**AeroLens Sentinel** is an autonomous **Edge AI inspection system** that runs entirely on-device (NVIDIA Jetson Nano / standard CPU) without requiring any cloud connectivity. It uses a trained **YOLOv8 neural network** combined with **OpenCV heuristic algorithms** to detect surface anomalies in real-time from uploaded inspection frames.

The system:
- Detects **4 critical defect classes**: Crack, Corrosion, Dent, and Paint Peel
- Assigns a **severity score** and **urgency band** (Immediate Ground / Scheduled Repair / Monitor)
- Generates a **downloadable MRO PDF report** with annotated scan images
- Stores all inspection records locally in a **SQLite telemetry database**
- Works fully **offline** in hangar environments

---

## 🔬 Technical Approach

### 1. Dataset & Training
- Custom annotated aerospace surface defect dataset (YOLO label format)
- YOLOv8-nano backbone trained for **30 epochs** achieving **0.995 mAP50**
- Model exported to **ONNX format** for hardware-agnostic inference

### 2. Hybrid Dual-Engine Inference Pipeline
The detection engine combines two parallel inference streams:
- **Deep Learning Stream**: YOLOv8 ONNX Runtime session detecting learned structural patterns
- **Heuristic Stream**: OpenCV adaptive contrast segmentation detecting texture anomalies
- Both outputs are pooled and filtered through **Non-Maximum Suppression (NMS)** to eliminate duplicate detections and false positives

### 3. Severity Triangulation Engine
Each detected defect is scored using a multi-factor formula:
```
Severity = Confidence × Zone_Criticality_Weight × Relative_Defect_Area
```
Zone criticality multipliers are configured per aircraft zone (e.g. Wing Spar = 1.0x, Engine Pylon = 1.5x, Fuselage = 0.8x)

### 4. MRO Reporting & Telemetry
- All scan records are persisted in a local **SQLite database**
- Annotated images with bounding boxes are saved per scan session
- A compiled **HTML/PDF MRO report** is auto-generated with all detected anomalies, confidence scores, severity ratings, and annotated images

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 AeroLens Sentinel System                │
├────────────────┬────────────────────────────────────────┤
│  Input Layer   │  Image Upload (JPG/PNG) via Dashboard  │
├────────────────┼────────────────────────────────────────┤
│  Detection     │  YOLOv8 ONNX Runtime  +  OpenCV NMS   │
├────────────────┼────────────────────────────────────────┤
│  Scoring       │  Severity Triangulation Engine          │
├────────────────┼────────────────────────────────────────┤
│  Storage       │  SQLite Telemetry + Annotated Images   │
├────────────────┼────────────────────────────────────────┤
│  Output        │  Dashboard UI + MRO PDF Report         │
└────────────────┴────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| AI Framework | YOLOv8 (Ultralytics) → ONNX Runtime |
| Vision Library | OpenCV (Headless) |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Deployment | Docker → Render (Free Tier) |
| Edge Hardware | NVIDIA Jetson Nano / CPU |
| Model Format | ONNX (CPU-optimized) |

---

## 🎯 Key Features

- ✅ **Real-time defect detection** from uploaded inspection images
- ✅ **4-class surface anomaly classification** (Crack, Corrosion, Dent, Paint Peel)
- ✅ **Annotated bounding box visualization** with color-coded severity
- ✅ **Per-zone criticality weighting** based on aircraft structural maps
- ✅ **SQLite telemetry logging** with full scan history
- ✅ **Downloadable MRO PDF report** with embedded annotated images
- ✅ **Session separator** distinguishing between different image upload batches
- ✅ **Fully offline capable** — no internet required during inspection
- ✅ **Dockerized deployment** on Render cloud

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/IqraKhanZ/Edge-AI-for-Intelligent-Inspection-Defect-Detection.git
cd Edge-AI-for-Intelligent-Inspection-Defect-Detection
```

### 2. Set up virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
# For running the dashboard only:
pip install -r requirements.txt

# For training the model as well:
pip install -r requirements-dev.txt
```

### 4. Run the dashboard
```bash
cd aerolens
python run.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training Epochs | 30 |
| Validation mAP50 | 0.995 |
| Inference Engine | ONNX Runtime (CPU) |
| Model Size | ~6 MB (ONNX) |

---

## 📂 Project Structure

```
aerolens/
├── config/              # System configuration
├── criticality_map/     # Zone criticality weights
├── dashboard/           # FastAPI backend + HTML frontend
│   ├── api.py
│   └── static/index.html
├── data/                # Dataset utilities
├── inference/           # Detector + Pipeline Orchestrator
│   ├── detector.py      # YOLOv8 ONNX + OpenCV hybrid engine
│   └── pipeline_orchestrator.py
├── models/              # Training scripts + ONNX weights
├── severity_engine/     # Severity scoring module
├── telemetry/           # SQLite logger
└── run.py               # Entry point
```

---

