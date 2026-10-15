# TrueFrame AI — Empirical CPU Benchmark Report
**Hardware Profile:** Intel Core i5-1235U (10 Cores, 12 Threads, Base 1.3GHz, Boost 4.4GHz)  
**Host Memory:** 16 GB DDR4/DDR5  
**OS & Python:** Windows 11 (64-bit), Python 3.11.9, PyTorch 2.14.1+cpu  
**Evaluation Date:** October 2026  
**Iterations:** 10 trials (post-warmup)  

---

## 1. Measured Performance Metrics

The following metrics were empirically measured directly on the host machine:

| Component / Subsystem | Measured Latency | Standard Deviation | Peak Additional Memory |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0 (CPU Inference)** | **98.63 ms** | $\pm 20.05\text{ ms}$ | $< 5.0\text{ MB}$ |
| **Eigen-CAM Saliency Generation** | **23.26 ms** | $\pm 4.12\text{ ms}$ | $< 1.5\text{ MB}$ |
| **Face Detection (MediaPipe LiteRT)** | **29.18 ms** | $\pm 7.37\text{ ms}$ | $< 3.0\text{ MB}$ |
| **Total Single-Image Pipeline** | **$\approx 151\text{ ms}$** | $\pm 24\text{ ms}$ | **$4.64\text{ MB}$** |

---

## 2. Technical Findings & Architectural Implications

1. **Sub-200ms CPU Feasibility:**
   - Single-image facial forensics completes in **$\approx 150\text{ ms}$ total** on standard x86-64 mobile CPU cores.
   - Proves that a dedicated NVIDIA GPU is **not required** for responsive single-image analysis or university viva demonstrations.
2. **Eigen-CAM Efficiency:**
   - Principal component projection across the final convolutional layer $(1, 1280, 7, 7)$ executes in **$23.26\text{ ms}$**, eliminating the computational overhead of gradient backpropagation while producing localized spatial heatmaps.
3. **Memory Footprint:**
   - Peak traced memory delta of **$4.64\text{ MB}$** easily satisfies free-tier hosting limits (both Render's 512MB and Hugging Face Spaces' 16GB tiers).
