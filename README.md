# Shadow Removal Pipeline — Classical Computer Vision

A classical computer vision pipeline for automatic shadow detection and removal from images, without any deep learning. Built using LAB color space analysis, connected component filtering, CLAHE, and Gaussian mask blending, with full hyperparameter grid search optimization.

**Course**: CSC 481 – Computer Vision, DePaul University

---

## Overview

Shadows in images degrade downstream tasks like object detection, segmentation, and scene understanding. This project implements a fully hand-crafted shadow removal pipeline that:

1. Detects shadow regions using LAB L-channel percentile thresholding and connected component analysis
2. Refines the shadow mask morphologically
3. Corrects illumination using CLAHE + gamma-based brightness boost
4. Restores color saturation in shadow regions
5. Blends the result smoothly using Gaussian-blurred mask boundaries
6. Selects optimal hyperparameters via grid search over 108 combinations on a training set

---

## Pipeline

```
Input BGR Image
      ↓
LAB Color Space Conversion → Extract L (luminance) channel
      ↓
Percentile Threshold → raw dark-pixel mask
      ↓
Connected Component Filtering
  → discard small blobs (< 0.5% image area)
  → keep only blobs darker than 92% of global mean L
      ↓
Morphological Refinement (close → open)
      ↓
CLAHE — adaptive histogram equalization on L channel
      ↓
Gamma Correction — non-linear brightness boost on shadow pixels
      ↓
HSV Saturation Restoration — re-saturate desaturated shadow colors
      ↓
Gaussian-blurred Mask Blending — smooth boundary transitions
      ↓
Corrected Output Image
```

---

## Shadow Mask Strategies

Two approaches are implemented and compared:

| Method | Description |
|---|---|
| **Percentile threshold** | Flags the darkest N% of L-channel pixels as shadow candidates, tuned per dataset via grid search |
| **Otsu's method** | Automatically finds the optimal luminance split by maximising inter-class variance — used as a reference comparison |

Both strategies feed into the same connected component filter, which discards blobs that are too small or not meaningfully darker than the scene average.

---

## Hyperparameter Grid Search

108 combinations searched across 30 training images:

| Parameter | Values searched |
|---|---|
| `percentile` | 25, 35, 45 |
| `brightness_factor` | 1.3, 1.5, 1.8 |
| `morph_kernel_size` | 5, 7, 11 |
| `blur_kernel_size` | 15, 25, 35 |

**Scoring function** (no-reference, since no ground-truth shadow masks are available):

```
score = entropy_gain + edge_preservation + 2 × normalized_brightness_gain
```

Brightness gain is weighted 2× because shadow removal must actually brighten the shadow region — entropy and edge scores alone don't capture whether the shadow was removed.

---

## Evaluation Metrics (No-Reference)

| Metric | What it measures |
|---|---|
| **Entropy Gain** | Shannon entropy increase after correction — more detail recovered |
| **Edge Preservation Score (EPS)** | Ratio of Sobel edge strength retained — high EPS means no blurring of structure |
| **Brightness Gain** | Mean L-channel increase inside shadow regions specifically |

---

## Dataset

Uses the **shadow_USR** dataset — outdoor images with natural cast shadows across varied scenes. Split into training (hyperparameter search) and test (final evaluation) sets.

Dataset is not redistributed here. Place extracted images under `shadow_USR/shadow_train/` and `shadow_USR/shadow_test/`.

---

## Tech Stack

| Category | Tools |
|---|---|
| Image Processing | OpenCV (LAB, HSV, CLAHE, morphology, Sobel, Gaussian blur) |
| Numerical Computing | NumPy, SciPy (Shannon entropy) |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Environment | Google Colab, Python 3.x |

No deep learning frameworks used — entirely classical computer vision.

---

## Outputs

| File | Description |
|---|---|
| `hyperparam_plot.png` | Effect of each hyperparameter on average quality score |
| `final_results.png` | Per-image bar charts of entropy gain, EPS, and brightness gain on test set |

---

## How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/<your-username>/shadow-removal-cv.git
   cd shadow-removal-cv
   ```

2. Install dependencies
   ```bash
   pip install opencv-python numpy matplotlib pandas scipy
   ```

3. Place the `shadow_USR` dataset folder in the project root (or update `DATASET_ROOT` in the notebook)

4. Open `CSC481_Final_project.ipynb` and run all cells top to bottom

---

## Key Design Decisions

- **Connected component filtering over raw threshold** — raw percentile masks pick up many small dark patches (hair, clothing, dark objects) that are not shadows. Filtering by area and mean luminance relative to the global scene mean significantly reduces false positives.
- **Gamma correction instead of flat brightness boost** — gamma < 1 brightens dark pixels more aggressively than bright ones, producing a more natural result in shadow regions.
- **Gaussian mask blending** — hard mask edges produce visible seams at shadow boundaries. Blurring the mask before blending creates smooth, natural-looking transitions.
- **No-reference evaluation** — since pixel-level ground truth shadow masks are unavailable, scoring combines three complementary signals: information content (entropy), structural fidelity (edge preservation), and shadow-specific correction (brightness gain inside the mask).

---

## Author

**Bhuvanesh Kantamuneni** — DePaul University, MS Artificial Intelligence
