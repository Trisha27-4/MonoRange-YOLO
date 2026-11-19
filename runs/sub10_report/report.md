# KITTI YOLO + object-depth report: `sub10`

## Setup
| item | value |
|---|---|
| detector weights | C:\Users\TRISHA\ERIC-robo\runs\sub10_det\weights\best.pt |
| detector imgsz | 640 |
| depth input | 384x1248 |
| frozen layers | 10 |
| train subset | 0.1 |
| detector epochs | 60 |
| depth epochs | 60 |
| geometric prior in head | True |
| val images | 500 |
| conf threshold | 0.5 |
| device | cuda:0 |

## Detection (Ultralytics, val split, IoU 0.5 / 0.5:0.95)
| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.690 | 0.587 | 0.618 | 0.393 |

| class | AP50 | AP50-95 |
|---|---|---|
| Car | 0.887 | 0.612 |
| Van | 0.565 | 0.397 |
| Truck | 0.728 | 0.567 |
| Pedestrian | 0.591 | 0.295 |
| Person_sitting | 0.278 | 0.175 |
| Cyclist | 0.650 | 0.327 |
| Tram | 0.788 | 0.469 |
| Misc | 0.457 | 0.298 |

## Depth (val, matched detections at conf 0.5, IoU >= 0.5, GT <= 60 m)
Recall at IoU 0.5 over 2448 GT objects: **74.9%**

| method | n | abs rel % | median rel % | MAE m | RMSE m | d<1.25 % |
|---|---|---|---|---|---|---|
| depth head | 1834 | 5.57 | 3.34 | 1.06 | 1.76 | 97.1 |
| geometric only | 1834 | 16.43 | 5.69 | 1.88 | 2.87 | 90.7 |

### By distance
| bin | n | head abs rel % | head MAE m | geo abs rel % | geo MAE m |
|---|---|---|---|---|---|
| 0-10 m | 379 | 10.46 | 0.48 | 51.98 | 1.28 |
| 10-20 m | 497 | 4.09 | 0.62 | 6.88 | 1.05 |
| 20-30 m | 366 | 4.65 | 1.17 | 7.37 | 1.85 |
| 30-40 m | 303 | 4.22 | 1.48 | 7.29 | 2.55 |
| 40-60 m | 289 | 4.26 | 2.02 | 7.29 | 3.46 |

### By class
| class | GT | recall | head abs rel % | geo abs rel % |
|---|---|---|---|---|
| Car | 1778 | 84.7% | 5.47 | 17.45 |
| Van | 155 | 54.8% | 7.15 | 16.78 |
| Truck | 42 | 64.3% | 6.88 | 16.03 |
| Pedestrian | 273 | 45.1% | 5.07 | 6.81 |
| Person_sitting | 19 | 15.8% | 5.81 | 8.81 |
| Cyclist | 111 | 55.0% | 4.18 | 6.40 |
| Tram | 28 | 57.1% | 14.27 | 16.94 |
| Misc | 42 | 31.0% | 4.58 | 35.56 |

### By KITTI difficulty (cumulative)
| level | n | head abs rel % | geo abs rel % |
|---|---|---|---|
| easy | 562 | 3.37 | 6.03 |
| moderate | 1238 | 3.79 | 6.71 |
| hard | 1485 | 4.30 | 7.50 |

## Latency
| device | images | detector ms | depth head ms | total ms | FPS (with depth) | FPS (geometric only) | peak VRAM MB |
|---|---|---|---|---|---|---|---|
| cuda:0 | 50 | 23.8 | 3.3 | 27.0 | 37.0 | 42.1 | 110 |
Detector ms includes Ultralytics pre/post-processing; the head is timed separately on cached features (the current inference path runs the backbone twice, so true single-pass cost is detector + head).

## Qualitative
- `val_vis/`: 30 val images with predicted box, learned depth, geometric depth and GT depth.
- `test_vis/`: 30 testing/ images, 132 detections (no GT exists for this split).
- Figures: `scatter_pred_vs_gt.png`, `error_by_distance.png`, `training_curves.png`.

## Caveats
- The val split also selected `best.pt`, so these numbers are slightly optimistic. A frame-level random split of KITTI leaks neighbouring frames between train and val.
- Geometric baseline uses the per-class mean real height from the training labels and the 2D box height. Truncated or occluded boxes bias it high.
- GT depth is the minimum z over the 8 corners of the 3D box (YOLO MDE definition), not the box centre.