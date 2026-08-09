# DMAIC-GML Engine

An interactive, edge-governed AI kernel demonstrating the application of Lean Six Sigma (DMAIC) Statistical Process Control (SPC) to govern deep learning model stability and prevent out-of-distribution (OOD) prediction drift[cite: 2, 3]. Built on MobileNetV2 with mandatory Global Average Pooling (GAP) and evaluated across 200 independent model lifecycles, DMAIC-GML translates backpropagation variance into a Six Sigma-controlled industrial process[cite: 2, 3].

🚀 **Live Interactive Prototype:** [https://dmaic-gml.streamlit.app/](https://dmaic-gml.streamlit.app/)

---

## Key Features

* **Six Sigma Quality Governance (DMAIC-GML):** Applies Define-Measure-Analyze-Improve-Control principles to deep neural network training, replacing heuristic hyperparameter tuning with structured variance reduction[cite: 2, 3].
* **Dual-Response Surface Optimization:** Implements Montgomery’s Dual-Response Methodology to simultaneously optimize location accuracy ($\mu$) and damp high-frequency process jitter ($\ln s^2$)[cite: 3].
* **Real-Time $I\text{-}mR$ Process Control Charts:** Dynamically plots individual predictions ($I$) and consecutive moving ranges ($mR$) against $3.0\sigma$ Shewhart control limits ($\text{UCL}/\text{LCL}$) to detect special-cause anomalies[cite: 2, 3].
* **Scale-Invariant Stochastic Stability Score ($S_s$):** Enforces an automated, scale-invariant deployment gatekeeper ($S_s \ge 0.95$) that blocks uncertified or volatile model states from reaching production[cite: 2, 3].
* **Dual-Domain Lexicographic Optimization:**
  * **In-Distribution (IID / PlantVillage):** Cell 7 configuration ($p=0.5, \lambda=1\text{e-}4$, Standard Augmentation) achieving $S_s = 0.9948$[cite: 3].
  * **Out-of-Distribution (OOD / Nigerian Cassava Field):** Cell 1 configuration ($p=0.2, \lambda=1\text{e-}5$, Standard Augmentation) achieving a **44.98% collapse in process jitter** ($\bar{mR} = 0.004672, S_s = 0.9729$) under severe environmental noise[cite: 2].
* **Architectural Integrity Enforcement:** Built on MobileNetV2 with Global Average Pooling (GAP) layers and Feature-Space SMOTE, eliminating spatial tensor explosion while preserving semantic feature maps[cite: 3].

---

## Technical Stack & Frameworks

* **Core Engine & Deep Learning:** Python, PyTorch (MobileNetV2 with GAP Architecture), Torchvision[cite: 3].
* **Quality & SPC Analytics:** NumPy, pandas, SciPy, Minitab-aligned $I\text{-}mR$ Shewhart Control Logic & ANOVA Variance Decomposition[cite: 2, 3].
* **Visualization & Interface:** Streamlit, Plotly, PIL[cite: 3].
* **Deployment & Cloud:** Streamlit Community Cloud / Google Cloud Platform (GCP) CI/CD[cite: 2, 3].

---

## Empirical Framework Summary ($N = 200$ Runs)

| Validation Horizon | Dataset Domain | Model Configuration | Process Mean | Average Moving Range ($\bar{mR}$) | Stability Score ($S_s$) | Governance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cycle 1: Baseline** | PlantVillage (IID) | Standard MobileNetV2 | 0.9692 | 0.002541 | Unstable (Special Cause) | Incapable[cite: 3] |
| **Cycle 3: Pre-Field Audit** | PlantVillage (IID) | Robust Cell 7 ($p=0.5, \lambda=1\text{e-}4$) | 0.9609 | 0.001871 | **0.9948** | **Certified (Optimal)**[cite: 3] |
| **Cycle 4: Post-Field OOD Baseline**| Cassava Field (OOD) | Stripped MobileNetV2 | 0.5823 | 0.010242 | Wide Jitter Band ($\Delta=0.0545$) | Un-governed Baseline[cite: 2] |
| **Cycle 6: Post-Field OOD Audit** | Cassava Field (OOD) | Robust Cell 1 ($p=0.2, \lambda=1\text{e-}5$) | 0.5526 | 0.005635 | **0.9729** | **Certified (44.98% Jitter Collapse)**[cite: 2] |

---

## Getting Started (Local Development)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/DMAIC-GML-Engine.git](https://github.com/your-username/DMAIC-GML-Engine.git)
   cd DMAIC-GML-Engine
