# Project Handoff

## Completed

- Recreated `.venv` from `requirements.txt` and selected it as the workspace Python interpreter.
- Prepared 500 evenly sampled NYU Depth V2 frames at 240x320.
- Trained the CNN-only and Hybrid models for 8 minutes each without noisy input. CNN completed 27 steps (validation loss 0.01807); Hybrid completed 46 steps (validation loss 0.00804).
- Ran clean and noisy evaluations and generated the qualitative and bar-chart figures.
- No pretrained weights or external APIs were used.
- Wrote and tested the Streamlit demo app (`streamlit_app.py`) and `make_samples.py` (exports 5 NYU test frames to `samples/`).Deployed on Streamlit Community Cloud from the fork `Sanjana2906/depth-upsampling` (Streamlit needs repo admin rights, which the original repo does not give contributors). Live URL: https://depth-upsampling-ncflvqguerfmecgrmz5xmd.streamlit.app/. Verified on desktop and mobile: the sample scene runs end to end with the same output (images, metrics table, error maps).

## Environment and measurements

- Python 3.11.9; PyTorch 2.15.0+cpu; OpenCV 5.0.0 with `cv2.ximgproc`.
- Demo app: Streamlit. `requirements.txt` now installs `opencv-contrib-python-headless` (same `cv2.ximgproc`, works on Streamlit Community Cloud) and adds `streamlit` and `pandas`. Do not install it alongside `opencv-contrib-python` or `opencv-python`.
- Hardware: Intel Core Ultra 5 125H CPU and Intel Arc graphics; no NVIDIA GPU was detected and `torch.cuda.is_available()` was false. Training and latency measurements therefore used the CPU.
- Latency is the mean per-image end-to-end time over 100 test frames. The CNN and Hybrid paths include bicubic upsampling, guided filtering, and the model forward pass.
- Dataset file: `C:\data\nyu_depth_v2_labeled.mat`, 2,972,037,809 bytes (reported by `ls -lh` as 2.8G; 2.77 GiB). The source file stays outside OneDrive.
- Split: 500 contiguous-order samples: 350 train, 50 validation, 100 test (70/10/20).
- Depth is linear and measured in metres; frames are 240x320 and depths are clipped to 10 m.

## Reproduction

Run the download in a separate Git Bash terminal:

```bash
mkdir -p /c/data
curl -L -o /c/data/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

From the workspace in Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python ps4.py prepare --mat /c/data/nyu_depth_v2_labeled.mat --samples 500
python ps4.py train --minutes 8
python ps4.py evaluate
python ps4.py evaluate --noise
python ps4.py figures
```

Demo app (needs `ps4.py`, `checkpoints/ckpt_cnn.pt`, `checkpoints/ckpt_hybrid.pt` and `data/processed/nyu_subset.npz` for the first step):

```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install -r requirements.txt
python make_samples.py          # once: writes samples/ (5 test frames + depth)
streamlit run streamlit_app.py
```

Deploy: push the repo to a public GitHub repo (without `.venv/` and `data/`), then on share.streamlit.io choose Create app, pick the repo and branch, set the main file to `streamlit_app.py`, and deploy. The repo must contain `streamlit_app.py`, `ps4.py`, `requirements.txt`, `samples/` and the two checkpoint files. Hugging Face Spaces was dropped because creating a Gradio Space now needs a paid plan.
The live app is deployed from the fork, so push updates to the fork's `main` branch to change it.

Results are in `outputs/results.csv`, `outputs/results.md`,
`outputs/results_noisy.csv`, and `outputs/results_noisy.md`. Figures are in
`outputs/figures/`; model checkpoints are in `checkpoints/`.

## Known limitations

- The training/evaluation run was CPU-only; model inference is comparatively slow.
- The experiment uses a 500-frame subset and a contiguous split, so it does not establish generalization to unseen datasets or independent scene splits.
- SSIM is computed over each full frame, including invalid-depth placeholder pixels; RMSE/MAE use only valid ground-truth pixels.
- Hybrid boundary RMSE did not improve over Bicubic at any factor in this run. Results were not tuned or altered.
- The demo needs an RGB image plus a depth map. There is no monocular depth model, so an RGB image alone is refused with a message.
- Streamlit Community Cloud's free tier has limited memory and puts apps with no traffic for 12 hours to sleep. Open the URL before judging.
- The bar chart uses clean evaluation results; noisy metrics are available in their separate CSV and Markdown files.
- Demo scene_1 at 4x, clean: Bicubic has the best RMSE (0.0308), MAE (0.0132) and boundary RMSE (0.0708). The hybrid is worst (RMSE 0.0474, boundary 0.1026) and about 160 ms vs 0.2 ms. Guided filter (0.0441) is worse than bicubic, so eps/radius are probably untuned.
- The hybrid error map shows faint background texture from the RGB guidance (texture-copying).
- Hole filling, the discontinuity detector and the Middlebury/KITTI test are not implemented. The audit marks missing-depth handling "insufficient", and SSIM includes invalid pixels.

## Teammate TODOs

- Create an architecture diagram and a design document.
- Create `eval_P4_<TeamID>.md` from `results.md`.
- Document 3 failure cases.
- Run an unseen-data test on Middlebury or KITTI.
- Explore hole filling, a depth-discontinuity detector, and an adaptive edge-confidence gate.
- Retrain both models on a GPU (Colab T4), then re-run evaluate, evaluate --noise and figures.
- Tune guided-filter eps and radius on the val split.
- Add peak CPU memory and fix SSIM to use valid pixels only.
- Format `eval_P4_<TeamID>.md` as the PS requires.
- Update the report claims to match the final numbers.
