# DMAIC-GML Engine

An interactive, edge-governed AI kernel demonstrating the application of Lean Six Sigma (DMAIC) Statistical Process Control (SPC) to govern deep learning model stability and prevent out-of-distribution (OOD) prediction drift. Built on MobileNetV2 with mandatory Global Average Pooling (GAP) and evaluated across 200 independent model lifecycles, DMAIC-GML translates backpropagation variance into a Six Sigma-controlled industrial process.

🚀 **Live Interactive Prototype:** [https://dmaic-gml.streamlit.app/](https://dmaic-gml.streamlit.app/)

---

## Key Features

* **Six Sigma Quality Governance (DMAIC-GML):** Applies Define-Measure-Analyze-Improve-Control principles to deep neural network training, replacing heuristic hyperparameter tuning with structured variance reduction.
* **Dual-Response Surface Optimization:** Implements Montgomery’s Dual-Response Methodology to simultaneously optimize location accuracy ($\mu$) and damp high-frequency process jitter ($\ln s^2$).
* **Real-Time $I\text{-}mR$ Process Control Charts:** Dynamically plots individual predictions ($I$) and consecutive moving ranges ($mR$) against $3.0\sigma$ Shewhart control limits ($\text{UCL}/\text{LCL}$) to detect special-cause anomalies.
* **Scale-Invariant Stochastic Stability Score ($S_s$):** Enforces an automated, scale-invariant deployment gatekeeper ($S_s \ge 0.95$) that blocks uncertified or volatile model states from reaching production.
* **Dual-Domain Lexicographic Optimization:**
  * **In-Distribution (IID / PlantVillage):** Cell 7 configuration ($p=0.5, \lambda=1\text{e-}4$, Standard Augmentation) achieving $S_s = 0.9948$.
  * **Out-of-Distribution (OOD / Nigerian Cassava Field):** Cell 1 configuration ($p=0.2, \lambda=1\text{e-}5$, Standard Augmentation) achieving a **44.98% collapse in process jitter** ($\bar{mR} = 0.004672, S_s = 0.9729$) under severe environmental noise.
* **Architectural Integrity Enforcement:** Built on MobileNetV2 with Global Average Pooling (GAP) layers and Feature-Space SMOTE, eliminating spatial tensor explosion while preserving semantic feature maps.
  
## Runtime Determinism Verification ($mR_{\text{inf}} = 0.000000$)

The DMAIC-GML engine integrates an automated runtime verification gatekeeper prior to full edge deployment. During inference audit on static inputs, the system computes the consecutive moving range ($mR_{\text{inf}}$) across sequential forward passes:

$$mR_{\text{inf}} = |X_t - X_{t-1}|$$

* **Evaluation-Mode Enforcement:** Executes the forward pass under frozen evaluation mode (`model.eval()`) with autograd graphs disabled (`torch.no_grad()`).
* **Zero Runtime Jitter ($mR_{\text{inf}} = 0.000000$):** Passes on identical static inputs yield a moving range of zero, certifying that execution is strictly deterministic and free from dynamic state fluctuations.
* **Pre-Deployment Sanity Gate:** Acts as an automated Poka-Yoke (mistake-proofing) interlock to confirm that stochastic training regularizers (e.g., Dropout) and batch-dependent statistics are completely frozen before live $I\text{-}mR$ time-series governance begins.

---

## Theoretical Justification

* **Mathematical Determinism of Forward Inference:** In inference mode, a deep convolutional neural network operates as a static mathematical function:

  $$f(x; W) = \hat{y}$$

  When the model weights ($W$) are frozen and the input tensor ($x$) remains identical across consecutive passes, deterministic linear algebra and fixed activation layers must produce identical logits.
* **Deactivation of Stochastic Layers:** In evaluation mode, pseudo-random mechanisms such as Dropout ($p$) are deactivated, and Batch Normalization layers utilize static population running statistics rather than dynamic batch calculations.
* **Six Sigma Process Integrity:** Runtime determinism provides the necessary foundation for quality governance. Ensuring $mR_{\text{inf}} = 0.000000$ guarantees that any subsequent variation detected by the $I\text{-}mR$ charts during production originates exclusively from genuine data distribution shifts or environmental noise, rather than runtime execution anomalies.
---

## Technical Stack & Frameworks

* **Core Engine & Deep Learning:** Python, PyTorch (MobileNetV2 with GAP Architecture), Torchvision.
* **Quality & SPC Analytics:** NumPy, pandas, SciPy, Minitab-aligned $I\text{-}mR$ Shewhart Control Logic & ANOVA Variance Decomposition.
* **Visualization & Interface:** Streamlit, Plotly, PIL.
* **Deployment & Cloud:** Streamlit Community Cloud / Google Cloud Platform (GCP) CI/CD.

---

## Empirical Framework Summary ($N = 200$ Runs)

| Validation Horizon | Dataset Domain | Model Configuration | Process Mean | Average Moving Range ($\bar{mR}$) | Stability Score ($S_s$) | Governance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cycle 1: Baseline** | PlantVillage (IID) | Standard MobileNetV2 | 0.9692 | 0.002541 | Unstable (Special Cause) | Incapable |
| **Cycle 3: Pre-Field Audit** | PlantVillage (IID) | Robust Cell 7 ($p=0.5, \lambda=1\text{e-}4$) | 0.9609 | 0.001871 | **0.9948** | **Certified (Optimal)** |
| **Cycle 4: Post-Field OOD Baseline**| Cassava Field (OOD) | Stripped MobileNetV2 | 0.5823 | 0.010242 | Wide Jitter Band ($\Delta=0.0545$) | Un-governed Baseline |
| **Cycle 6: Post-Field OOD Audit** | Cassava Field (OOD) | Robust Cell 1 ($p=0.2, \lambda=1\text{e-}5$) | 0.5526 | 0.005635 | **0.9729** | **Certified (44.98% Jitter Collapse)** |

---

## Getting Started (Local Development)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/DMAIC-GML.git](https://github.com/your-username/DMAIC-GML.git)
   cd DMAIC-GML
