# XAI Refresher Overview

## Introduction

Welcome to the **XAI Refresher** implementation of the Interpretability for LLMs and Agents Bootcamp.
This folder covers foundational and advanced techniques in Explainable AI (XAI), with a focus on
post-hoc explanation methods for both traditional neural networks and modern vision-language models
(VLMs). We explore how to make model decisions interpretable through feature attribution,
segmentation-based perturbations, concept decomposition, and gradient-based visualization.

## Prerequisites

Before diving into the materials, ensure you have the following:

- Python 3.10 or higher
- PyTorch 2.x
- Basic familiarity with neural networks and image classification
- Familiarity with Python and Jupyter notebooks
- A CUDA-capable GPU is recommended for the concept grounding notebook

## Notebooks

The following Jupyter notebooks are provided in this folder:

1. **[LIME](lime.ipynb)** — Covers the LIME (Local Interpretable Model-agnostic Explanations)
   framework for image, tabular, and text models. Includes LORE (rule-based local explanations
   for tabular data) and DSEG-LIME (SAM-powered data-driven segmentation for richer image
   explanations).

2. **[SHAP](shap.ipynb)** — Introduces SHAP (SHapley Additive exPlanations) with KernelExplainer
   applied to a PyTorch MLP trained on the UCI Credit Card Default dataset. Covers SHAP value
   computation, summary plots, and how to interpret additive feature contributions.

3. **[CLIP Interpretability](clip.ipynb)** — Explores concept-based interpretability for
   vision-language models using CLIP. Covers representation-level analysis, Grad-CAM and
   EigenCAM heatmaps, and how embedding-space geometry relates to model decisions.

4. **[Concept Grounding](concept_grounding.ipynb)** — Demonstrates how to extract and decompose
   hidden-state features from LLaVA (7B) using Symmetric Non-negative Matrix Factorization
   (SNMF). Covers concept dictionary learning, multimodal grounding (text + image), and
   local interpretations per sample on COCO.

### Package Dependencies

The Concept Grounding notebook requires Java to be installed. On Linux, you can install it using:

```bash
sudo apt update
sudo apt install -y default-jre
```

## Resources

For further reading on the methods covered in this module:

- **LIME** — Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any
  Classifier", KDD 2016.
- **SHAP** — Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
  NeurIPS 2017.
- **DSEG-LIME** — Narayanan et al., "DSEG-LIME: Improving Image Explanation by Incorporating
  Feature Importance of Superpixels", 2024.
- **Concept Grounding in VLMs** — Toker et al., "Interpretability of Vision-Language Models
  via Concept Bottlenecks", 2024.
- **Grad-CAM** — Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.

## Getting Started

1. From the **root of the repository**, create a virtual environment and install the
   `xai-refresher` dependency group using `uv`:

   ```bash
   uv sync --group xai-refresher
   ```

   This creates a `.venv` in the repo root and installs all packages needed for this module
   (PyTorch, SHAP, LIME, Grad-CAM, SAM, etc.).

2. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

3. Start with **[lime.ipynb](lime.ipynb)** for a ground-up introduction to post-hoc explanation
   with LIME and its variants.

4. Proceed to **[shap.ipynb](shap.ipynb)** to explore Shapley-value-based attribution on a
   tabular classification task.

5. Move to **[clip.ipynb](clip.ipynb)** to see how gradient-based and representation-level
   explanations apply to vision-language models.

6. Finish with **[concept_grounding.ipynb](concept_grounding.ipynb)** for a deep dive into
   concept decomposition and grounding in LLaVA. Note: this notebook requires a GPU and
   will download model weights on first run.
