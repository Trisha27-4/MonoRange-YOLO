# Monocular Object Detection + Object-Level Distance Estimation (KITTI)

Detect road objects in a single RGB image and predict **one camera-axis distance (metres) per detection**.

A YOLO26n detector is fine-tuned on KITTI. A small object-level depth head, adapted from
[YOLO MDE (Yu & Choi, 2022)](https://doi.org/10.3390/electronics11010076), is trained on the frozen detector's
neck features. A pinhole geometric estimate is used as both an interpretable baseline and the starting point of the
learned prediction.

```
RGB image ──► YOLO26n (frozen after stage 1) ──► boxes, classes, confidence
                    │
                    └─ 3 neck feature maps ──► ROI-Align ──► depth head ──► Z (m) per box
                                                                  ▲
                              geometric prior  Z_geo = fy·H_class / h_px
```

Output per detection: class, confidence, box, and depth. Example label: `Car 12.3m | geo 11.8m`.

## Results (`sub10` run: 10% of the training split, 500-image val split)

| Metric | Value |
|---|---|
| Detection mAP@50 / mAP@50-95 | 0.618 / 0.393 |
| Detection precision / recall | 0.690 / 0.587 |
| Recall at IoU 0.5 (depth pipeline, conf 0.5, GT up to 60 m) | 74.9% |
| Depth abs. rel. error: learned head / geometry only | **5.57%** / 16.43% |
| Depth MAE: head / geometry | 1.06 m / 1.88 m |
| Depth RMSE: head / geometry | 1.76 m / 2.87 m |
| Within 25% (d < 1.25): head / geometry | 97.1% / 90.7% |
| Latency, GTX 1650: detector + head | 27.0 ms (about 37 FPS) |

Depth is evaluated on matched detections (same class, IoU ≥ 0.5) with ground truth capped at 60 m.
Caveats: validation comes from a random frame-level split of the labelled training data, so numbers are optimistic;
the KITTI `testing/` split has no labels. Details are in the technical report and in `runs/<name>_report/report.md`.

## Repository layout

```
project/
├── training_pipeline.py     # prepare data, train detector, train depth head, evaluate, infer, smoke test
├── test.py                  # full evaluation: metrics, plots, report.md, annotated images
├── training/                # KITTI training split (you provide)
│   ├── calib/  image_2/  label_2/  velodyne/ (unused)
├── testing/                 # KITTI testing split (no labels; qualitative output only)
│   ├── calib/  image_2/
├── kitti_yolo/              # generated: YOLO-format images/labels, depth_labels.json, yaml files
└── runs/                    # generated
    ├── <name>_det/          # Ultralytics run: weights/best.pt, results.csv, config.json
    ├── <name>_depth/        # best.pt, last.pt, epochs.csv, config.json, eval.json, viz/
    └── <name>_report/       # report.md, metrics.json, per_object.csv, plots, val_vis/, test_vis/
```

If your training script is named `train.py`, `test.py` falls back to importing it automatically.

## Setup

Tested on Windows, Python 3.11, Ultralytics 8.4.x, NVIDIA GTX 1650 (4 GB).

1. Install a **CUDA build** of PyTorch and torchvision (check the selector at pytorch.org/get-started/locally):
   ```
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   The version must not end in `+cpu`, and the second value must be `True`.
2. Install the rest:
   ```
   pip install ultralytics opencv-python pillow matplotlib numpy
   ```
   If you see `numpy.core.multiarray failed to import`, upgrade matplotlib (`pip install -U matplotlib`).
3. Place KITTI under `training/` and `testing/` as shown above.

## Usage

Run from the project folder. `--name` tags every output folder so experiments do not overwrite each other.

```
# inspect model layers and what a freeze setting would lock
python training_pipeline.py --stage layers --freeze 10

# everything: prepare -> detector -> depth head -> evaluation
python training_pipeline.py --stage all --name sub10 --subset 0.10 --det_imgsz 640 --val_every 10

# or stage by stage
python training_pipeline.py --stage prepare
python training_pipeline.py --stage det   --name sub10 --subset 0.10 --det_imgsz 640
python training_pipeline.py --stage depth --name sub10 --subset 0.10 --val_every 10
python training_pipeline.py --stage eval  --name sub10

# quick smoke test: 15 epochs on 1/15th of the data (use --subset 0.02 for smaller)
python training_pipeline.py --stage test --subset 0.02 --det_imgsz 640 --batch 4

# draw boxes with distances on val images, a folder, a video or a webcam
python training_pipeline.py --stage infer --name sub10 --source val --n 20 --show_geo
python training_pipeline.py --stage infer --name sub10 --source path/to/video.mp4 --hfov 82

# full report: metrics, plots, annotated val and testing images
python test.py --name sub10 --conf 0.5
python test.py --name sub10 --device cpu        # CPU latency
```

### Key options

| Option | Meaning |
|---|---|
| `--name` | Experiment tag; outputs go to `runs/<name>_det`, `runs/<name>_depth`, `runs/<name>_report` |
| `--freeze N` or `0,1,2` | Freeze layers 0..N-1, or an explicit list, in the detector stage |
| `--subset f` | Fraction of the **training** data (same seeded subset in both stages) |
| `--val_subset f` | Fraction of validation data (default: full 500, or `--subset` in `--stage test`) |
| `--val_every N` | Validate every N epochs; the final epoch is always validated |
| `--det_imgsz` | Detector training size (multiple of 32); lower it if VRAM is tight |
| `--det_weights` | Override detector weights (default `runs/<name>_det/weights/best.pt`) |
| `--device` | `cuda:0` (default) or `cpu` |
| `--conf` | Confidence threshold for the depth pipeline and reports (default 0.5) |

## Method summary

- **Depth ground truth:** minimum camera-axis `z` over the 8 corners of each object's 3D box (YOLO MDE definition).
  Objects with depth below 0.5 m or degenerate boxes are dropped; evaluation is capped at 60 m.
- **Detector stage:** YOLO26n fine-tuned on 8 classes (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram,
  Misc). DontCare rows are skipped. Layers 0-9 (backbone through SPPF) are frozen by default.
- **Depth head:** ROI-Align (7x7) on the three neck feature maps, 1x1 reductions, 3x3 mixing, class embedding and
  box-geometry scalars, then an MLP. Output activation is the paper's `D = -14.4 log(sigmoid(o))`, computed as
  `14.4 softplus(-o)`.
- **Geometric prior:** `Z_geo = fy · H_class / h_px`, using each image's calibration `fy` and per-class mean height
  from the training labels. The head's final layer is zero-initialised so training starts at `Z_geo` and learns a residual.
- **Loss:** Huber (delta = 1) + relative error. Adam, lr 1e-3, x0.1 every 20 epochs, 60 epochs. Augmentation:
  horizontal flip and 10% box jitter. Training is staged (detector, then head), not joint.
- **Not implemented:** temporal loss (KITTI object frames are independent) and a geometric consistency loss
  (`LAMBDA_GEO = 0`).

Differences from the paper: YOLO26n instead of YOLOv4, a separate ROI head on frozen features instead of an extra
output channel, eight classes instead of two. This is an adaptation, not a reproduction.

## Outputs and logs

- **Per-epoch logs (train and val):** `runs/<name>_det/results.csv` (Ultralytics; strip header spaces when loading)
  and `runs/<name>_depth/epochs.csv`. Both stages also save `config.json`.
- **Checkpoints:** detector `runs/<name>_det/weights/best.pt`; depth head `runs/<name>_depth/best.pt` and `last.pt`.
- **Report:** `runs/<name>_report/report.md` and `metrics.json` contain detection metrics, depth metrics overall and by
  distance, class and difficulty, latency, and caveats. `per_object.csv` has one row per matched object.
- **Images:** `val_vis/` shows predicted box, learned depth, geometric depth and ground-truth depth;
  `test_vis/` shows predictions on KITTI `testing/` images. Box colour: red < 10 m, orange < 25 m, green otherwise.

## Known limitations

- Random frame-level train/val split leaks neighbouring frames, and validation also selects `best.pt`.
- No labelled test split: no official benchmark score without the KITTI evaluation server.
- DontCare regions are treated as background, which may lower precision.
- The detector was trained at 640 while depth evaluation and inference run at 384x1248.
- Inference currently runs the backbone twice (once for detections, once to capture features); a single-pass
  implementation would be faster.
- Latency was measured on a laptop GPU only. No quantisation, pruning, or edge-device benchmarks yet.
- Depth is trained with KITTI's camera; accuracy on other cameras needs the correct `--fy` or `--hfov` and is not validated.

## Troubleshooting

| Problem | Fix |
|---|---|
| Header shows `CPU`, or the script exits with a CUDA message | Reinstall the CUDA build of torch (Setup step 1) |
| `numpy.core.multiarray failed to import` | `pip install -U matplotlib` (then scipy, pandas, opencv if needed) |
| `cv2.imshow` raises "function is not implemented" | You have the headless OpenCV build. Outputs are saved to files, so no window is needed; or reinstall `opencv-python` |
| CUDA out of memory (4 GB card) | `--det_imgsz 640 --batch 4`, or `--batch 2` |
| Epochs unexpectedly slow in a subset run | Check the device line, then use `--subset`/`--val_subset` so validation is also reduced |
| Dataloader hangs on Windows | `--workers 0` or `2` |

## Roadmap

1. Pure-learned ablation (geometric prior off) and per-object error analysis by truncation and occlusion.
2. Full training split, larger detector image size, sequence-based split.
3. DontCare masking; optional joint fine-tuning of the neck.
4. Single-pass inference, ONNX export, FP16/INT8 quantisation, pruning, CPU and edge benchmarks.

## References

- Yu, J. and Choi, H. (2022). *YOLO MDE: Object Detection with Monocular Depth Estimation.* Electronics, 11(1), 76.
- Geiger, A., Lenz, P., Stiller, C. and Urtasun, R. (2013). *Vision Meets Robotics: The KITTI Dataset.* IJRR, 32(11).
- Ultralytics YOLO documentation: https://docs.ultralytics.com/
