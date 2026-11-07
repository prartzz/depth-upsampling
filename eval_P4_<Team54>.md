# eval_P4_<Team54>: Edge-Preserving Depth Map Upsampling

Live demo: https://depth-upsampling-ncflvqguerfmecgrmz5xmd.streamlit.app/

## Setup
- Data: 500 evenly sampled NYU Depth V2 frames, 240x320, depth in metres clipped to 10 m.
- Split (contiguous): 350 train / 50 val / 100 test. All numbers below are on the 100 test frames.
- Factors: 2x, 4x, 8x. Noisy runs add 2 cm Gaussian noise and 5% outlier spikes to the low-res depth.
- Models: CNN-only U-Net and Hybrid (bicubic -> guided filter -> CNN refinement), each trained from scratch for 8 minutes on CPU (27 and 46 steps), on clean input only. No pretrained weights.
- Metrics: RMSE, MAE, boundary RMSE/MAE (band around GT depth edges), abs-rel, PSNR, SSIM, latency. RMSE/MAE use valid GT pixels only. Full tables: `outputs/results.md`, `outputs/results_noisy.md`.

## RMSE (m), lower is better
| Method | Clean 2x | Clean 4x | Clean 8x | Noisy 2x | Noisy 4x | Noisy 8x |
|---|---|---|---|---|---|---|
| Nearest | 0.0684 | 0.1158 | 0.1832 | 0.8364 | 0.8395 | 0.8479 |
| Bilinear | 0.0580 | 0.0964 | 0.1547 | 0.5348 | 0.5666 | 0.5871 |
| Bicubic | **0.0503** | 0.0864 | 0.1409 | 0.7169 | 0.7229 | 0.7303 |
| Joint bilateral | 0.0654 | **0.0831** | **0.1368** | 0.3141 | 0.4977 | 0.5162 |
| Guided filter | 0.1019 | 0.1101 | 0.1762 | 0.2221 | 0.3360 | 0.3792 |
| CNN-only | 0.1516 | 0.1672 | 0.2005 | 0.7496 | 0.7565 | 0.7641 |
| Hybrid | 0.1100 | 0.1175 | 0.1801 | **0.2171** | **0.3339** | **0.3780** |

## Boundary RMSE (m), lower is better
| Method | Clean 2x | Clean 4x | Clean 8x | Noisy 2x | Noisy 4x | Noisy 8x |
|---|---|---|---|---|---|---|
| Nearest | 0.1130 | 0.1911 | 0.3002 | 0.8142 | 0.8159 | 0.8579 |
| Bilinear | 0.0960 | 0.1595 | 0.2538 | 0.5237 | 0.5586 | 0.6124 |
| Bicubic | **0.0832** | 0.1426 | 0.2306 | 0.6967 | 0.7006 | 0.7393 |
| Joint bilateral | 0.1078 | **0.1370** | **0.2234** | 0.3335 | 0.5033 | 0.5544 |
| Guided filter | 0.1678 | 0.1814 | 0.2825 | 0.2611 | 0.3660 | 0.4438 |
| CNN-only | 0.1728 | 0.2084 | 0.2756 | 0.7268 | 0.7309 | 0.7683 |
| Hybrid | 0.1692 | 0.1825 | 0.2828 | **0.2571** | **0.3639** | **0.4423** |

## Latency (ms/image, CPU, clean run)
| Method | 2x | 4x | 8x |
|---|---|---|---|
| Nearest | 0.18 | 0.19 | 0.37 |
| Bilinear | 0.28 | 0.26 | 0.34 |
| Bicubic | 0.48 | 0.72 | 0.98 |
| Joint bilateral | 13.5 | 17.3 | 69.9 |
| Guided filter | 6.0 | 7.6 | 9.0 |
| CNN-only | 363.5 | 539.9 | 725.1 |
| Hybrid | 355.5 | 521.6 | 731.8 |

## Findings
1. **Clean input:** classical interpolation is best. Bicubic wins at 2x and joint bilateral at 4x and 8x, in RMSE and in boundary RMSE. The hybrid does not improve accuracy or boundary preservation over interpolation on clean data.
2. **Noisy input:** the hybrid has the lowest RMSE and boundary RMSE at all factors, reducing RMSE vs bicubic by about 70% (2x), 54% (4x) and 48% (8x), and boundary RMSE by about 63%, 48% and 40%.
3. **The noisy gain is mostly the guided filter.** The hybrid beats guided filtering alone by only 2.2% (2x), 0.6% (4x) and 0.3% (8x) in RMSE. The CNN, with 46 training steps, adds little.
4. **Bicubic is a weak noisy baseline.** It spreads outlier spikes and is worse than bilinear under noise. No median/outlier-removal baseline was run, so the noisy improvement overstates the practical gain.
5. **Cost:** CNN and hybrid take 350 to 730 ms/image on CPU against under 1 ms for interpolation.
6. **Error grows with factor.** At 8x, boundary RMSE is 0.22 to 0.30 m for every method on clean data.

## Failure cases
1. **Hybrid worse than its own guided-filter stage on clean data** (RMSE 0.1100 vs 0.1019 at 2x; 0.1175 vs 0.1101 at 4x; 0.1801 vs 0.1762 at 8x). The undertrained CNN degrades the guided output.
2. **CNN-only has a large systematic error.** Its MAE (about 0.14 m) is close to its RMSE on clean data and far above the other methods, which suggests a depth bias from undertraining.
3. **Texture copying and edge error in the hybrid.** On `scene_1` at 4x the hybrid error map shows faint wall and blind texture from the RGB guide, and the guided filter adds error along the table legs and edges (boundary RMSE 0.1026 vs 0.0708 for bicubic).

## Limitations
- Models are undertrained (CPU, 8 minutes) and were trained on clean input only.
- Only 500 NYU frames with a contiguous split, so this does not show generalization. There is no Middlebury/KITTI test.
- No hole filling or dedicated discontinuity detector. Missing-depth handling is limited to masking invalid pixels in loss and metrics.
- SSIM includes invalid-depth placeholder pixels. Peak memory (CPU) is not measured. No GPU was available.
- Guided filter eps/radius were not tuned.
