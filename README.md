# TrueFrame AI — Digital Media Forensics Platform

**Academic Milestone:** Semester 8 Major Project-II  
**Discipline:** Computer Engineering  
**Focus:** Digital Media Forensics, Deepfake Detection, Model Explainability & Uncertainty Quantification  

---

## 1. Project Overview

**TrueFrame AI** is a research-oriented digital media forensics platform developed as a Semester 8 Major Project. Moving beyond a simple black-box "real vs. fake" binary classifier, TrueFrame AI is engineered to provide verifiable forensic evidence, uncertainty estimation, and explainable artifact localization on suspect images and videos.

### Core Capabilities
- **Spatial Deepfake Detection:** Face detection, landmark alignment, and feature extraction using parameter-efficient convolutional backbones.
- **Uncertainty-Aware Scoring:** Temperature-calibrated confidence scores with uncertainty intervals rather than arbitrary overconfident classifications.
- **Visual Forensic Evidence:** Spatial activation localization (Eigen-CAM / Grad-CAM) highlighting anomalous facial blending seams and texture inconsistencies.
- **Zero-Cost Deployment:** Engineered to run inference locally on standard multi-core CPUs without requiring dedicated GPUs, and deployable to free-tier cloud hosting.

---

## 2. Architecture & Design Principles

- **Frontend:** React 19 + TypeScript + Vite. Restrained, technical analyst interface without decorative fluff.
- **Backend:** FastAPI (Python 3.11). Asynchronous modular monolith with strict Pydantic v2 data validation schemas.
- **Inference Engine:** CPU-optimized PyTorch and ONNX Runtime execution providers.
- **Storage Strategy:** Ephemeral-first media processing. Raw uploads are verified, evaluated, and purged immediately to protect privacy and stay within zero-cost storage quotas.

---

## 3. Directory Layout

```
TrueFrame-AI/
├── backend/               # FastAPI backend service, schemas, and forensic pipeline
│   ├── .env.example       # Backend environment configuration template
│   └── ...
├── frontend/              # React + Vite + TypeScript web interface
│   ├── .env.example       # Frontend environment configuration template
│   └── ...
├── models/                # Model architecture definitions and weights directory
│   └── weights/           # Pretrained model weights (gitignored)
├── storage/               # Sandboxed temporary media storage (gitignored)
│   └── temp/
├── docs/                  # Architecture specifications, benchmarks, and viva notes
├── .gitignore             # Comprehensive ignore rules
└── README.md              # Project documentation
```

---

## 4. Academic & Forensic Disclaimer

Deepfake detection is an adversarial research domain. Deep learning models trained on specific manipulation distributions (e.g., FaceForensics++) exhibit domain shift and performance degradation when evaluated against unseen synthesis techniques or heavy social media compression. TrueFrame AI explicitly reports uncertainty intervals and evidentiary limitations in its forensic assessments.
