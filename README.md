# End-to-End Medical Anomaly Detection System (Beta-VAE + React)

A full-stack production application for unsupervised rare disease and lesion anomaly detection in medical imaging (Brain MRI).

---

## 1. System Architecture

- **AI Engine (`backend/`)**:
  - PyTorch-based $\beta$-Variational Autoencoder ($\beta$-VAE) trained strictly on healthy scans to model $P(X_{healthy})$.
  - Anomaly detection via pixel residual error maps: $E(u, v) = \| X_{original}(u, v) - X_{reconstructed}(u, v) \|^2$.
  - Post-processed with Gaussian spatial filtering and OpenCV thermal Jet color blending.
  - Zero MONAI dependency; strictly uses standard PyTorch, Torchvision, and OpenCV.
- **REST Service (`backend/app.py`)**:
  - FastAPI serving live inference, multi-modal image base64 responses, demo benchmark slices, and ROC-AUC metrics.
- **Frontend (`frontend/`)**:
  - React (Vite) + Tailwind CSS + Lucide Icons + Recharts.
  - 4-panel diagnostic visualizer:
    1. Input Scan ($X$)
    2. Healthy Manifold Reconstruction ($\hat{X}_{healthy}$)
    3. Thermal Jet Residual Heatmap
    4. Binary Lesion Segmentation Mask
  - Real-time threshold ($\tau$) and Gaussian smoothing sliders.
  - Latent coordinate space inspection chart ($\mu \in \mathbb{R}^{64}$).
  - Model ROC curve and Youden's J calibration benchmarks.

---

## 2. Quickstart & How to Run

### Step 1: Start the Backend FastAPI Server
In a terminal at the project root:
```bash
# Set Python path to current workspace
$env:PYTHONPATH="."    # Windows PowerShell
# or export PYTHONPATH="." on Linux/macOS

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
The REST API will be accessible at: `http://localhost:8000/docs`

---

### Step 2: Start the Frontend React Application
In a second terminal:
```bash
cd frontend
npm run dev
```
Open your browser at: `http://localhost:5173`

---

## 3. Training / Retraining the VAE Model
To train the model on fresh MRI scans with KL annealing:
```bash
$env:PYTHONPATH="."
python backend/services/train.py --epochs 20 --batch_size 16 --latent_dim 64 --warmup 8
```
Model weights are automatically saved to `backend/checkpoints/best_vae_model.pt`.
