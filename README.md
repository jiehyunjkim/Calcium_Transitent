# Automated Analysis of Calcium Transients in Engineered Cardiac Microtissues

## Overview
This project analyzes motion in calcium fluorescence videos of engineered cardiac microtissues (eMTs). It uses tissue segmentation, calcium signal analysis, and optical flow to measure how motion changes during contraction and relaxation and across different tissue regions. 

## Pipeline Overview

![Pipeline](/viz/pipeline.png)

The pipeline first segments the tissue in each video frame. It then extracts a calcium fluorescence signal and uses its peaks to identify contraction (rise) and relaxation (decay) phases. Lucas–Kanade optical flow tracks motion within the tissue. A 3 × 2 grid divides the tissue into six regions, where velocity, motion direction, directional consistency, and velocity variation are measured separately for each phase.

## Ablation Study

We compared five analysis settings to see how combining phases or tissue regions changes the measured motion. **E** is the phase-aware, six-region analysis used in this study.

| Setting | Spatial regions | Separate contraction and relaxation? |
|:--|:--:|:--:|
| **A** | — | No motion tracking (calcium signal only) |
| **B** | 1 × 1 | No |
| **C** | 3 × 2 | No |
| **D** | 1 × 1 | Yes |
| **E** | 3 × 2 | Yes |

### Results

| Comparison | Measurement | Result |
|:--|:--|--:|
| C vs. E | Directional consistency (C / E) | 0.365 ± 0.209 / 0.835 ± 0.191 |
| B vs. E | Directional consistency (B / E) | 0.377 ± 0.234 / 0.835 ± 0.191 |
| B vs. E | Full-beat velocity relative to E | 9.1 ± 4.0% |
| C vs. E | Full-beat velocity relative to E | 17.4 ± 7.3% |
| D vs. E | Contraction velocity relative to E | 53.8 ± 19.2% |
| D vs. E | Relaxation velocity relative to E | 47.8 ± 18.5% |

Combining contraction and relaxation (C) reduced directional consistency from **0.835 to 0.365**. Removing the six-region division (D) lowered measured contraction and relaxation velocities by about **46% and 52%**, respectively. These comparisons show that the choice of time windows and tissue regions can substantially change optical flow measurements. They do not measure ground-truth motion accuracy.
