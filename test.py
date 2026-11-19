"""
test.py  -  evaluation + report for the KITTI YOLO + object-depth model

What it produces in runs/<name>_report/
  report.md            all tables (detection, depth, per-distance/class/difficulty, latency, setup)
  metrics.json         the same numbers, machine-readable
  per_object.csv       one row per matched GT object (gt depth, head depth, geometric depth, errors ...)
  *.png                pred-vs-GT scatter, error by distance, training curves
  val_vis/             val images: predicted box + learned depth + geometric depth + GT depth
  test_vis/            KITTI testing/ images: predicted box + learned depth + geometric depth
  test_predictions.csv detections on testing/ images

IMPORTANT: KITTI's testing/ split has NO labels. Quantitative metrics therefore come from the held-out
val split (500 images); testing/ images are used for qualitative output and nothing else.

Run:  python test.py --name sub10 --conf 0.5
      python test.py --name sub10 --device cpu        # CPU latency numbers
"""
import os
import sys

if "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # keep Ultralytics from silently moving the model to GPU

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import training_pipeline as tp
except ImportError:                                    # your local copy is named train.py
    import train as tp

CLASSES, IMG_H, IMG_W = tp.CLASSES, tp.IMG_H, tp.IMG_W
MAX_D = tp.EVAL_MAX_DEPTH
BINS = [0, 10, 20, 30, 40, 60]


# ----------------------------------------------------------------------------- helpers
def to_tensor(frame_bgr):
    small = cv2.cvtColor(cv2.resize(frame_bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(small).permute(2, 0, 1).float() / 255


def predict(det, head, class_h, frame, fy_orig, conf):
    """-> (boxes_in_original_px, cls, depth, geo, score) as CPU numpy, or None."""
    H0, W0 = frame.shape[:2]
    sx, sy = IMG_W / W0, IMG_H / H0
    out = tp.predict_depths(det, head, class_h, to_tensor(frame), fy_orig * sy, conf)
    if out is None:
        return None
    out = [x.detach().cpu() for x in out]
    if len(out) == 4:                                   # older predict_depths without scores
        out.append(torch.ones(len(out[0])))
    boxes, cls, depth, geo, score = out
    boxes = boxes.numpy() / np.array([sx, sy, sx, sy], dtype=np.float32)
    return boxes, cls.numpy(), depth.numpy(), geo.numpy(), score.numpy()


def depth_color(d):
    return (60, 60, 230) if d < 10 else (0, 165, 255) if d < 25 else (80, 200, 80)   # BGR


def draw(img, pred, gts=None):
    """Predicted box + 'Class 12.3m | geo 11.8m'. Optional GT boxes (thin blue) with 'GT 12.1m' underneath."""
    img = img.copy()
    for (x1, y1, x2, y2), d in (gts or []):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 0), 1)
        cv2.putText(img, f"GT {d:.1f}m", (int(x1), int(y2) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if pred is None:
        return img
    for (x1, y1, x2, y2), c, d, g, s in zip(*pred):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        col = depth_color(d)
        txt = f"{CLASSES[c]} {d:.1f}m | geo {g:.1f}m"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yt = max(y1, th + bl + 2)
        cv2.rectangle(img, (x1, yt - th - bl - 2), (x1 + tw, yt), col, -1)
        cv2.putText(img, txt, (x1, yt - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def difficulty(h_px, trunc, occ):
    """KITTI-style cumulative difficulty flags (easy, moderate, hard)."""
    return (h_px >= 40 and occ == 0 and trunc <= 0.15,
            h_px >= 25 and occ <= 1 and trunc <= 0.30,
            h_px >= 25 and occ <= 2 and trunc <= 0.50)


def depth_metrics(gt, pred):
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    if len(gt) == 0:
        return None
    err = np.abs(gt - pred)
    rel = err / gt
    return dict(n=int(len(gt)), abs_rel=float(rel.mean() * 100), med_rel=float(np.median(rel) * 100),
                mae=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())),
                d1=float((np.maximum(gt / pred, pred / gt) < 1.25).mean() * 100))


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def fmt(m):
    return ["-"] * 6 if m is None else [m["n"], f"{m['abs_rel']:.2f}", f"{m['med_rel']:.2f}",
                                         f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['d1']:.1f}"]


DEPTH_HDR = ["n", "abs rel %", "median rel %", "MAE m", "RMSE m", "d<1.25 %"]


# ----------------------------------------------------------------------------- 1. detection metrics (Ultralytics)
def detection_metrics(a, out_dir, imgsz):
    try:
        r = YOLO(a.det_weights).val(data=str(tp.OUT / "kitti.yaml"), split="val", imgsz=imgsz, batch=4,
                                    device=a.device, project=str(out_dir), name="det_val", exist_ok=True,
                                    plots=True, verbose=False)
    except Exception as e:                              # keep the rest of the report alive
        print(f"[warn] detector validation failed: {e}")
        return dict(error=str(e))
    b = r.box
    per_class = {}
    for k, ci in enumerate(b.ap_class_index):
        per_class[r.names[int(ci)]] = dict(ap50=float(b.ap50[k]), ap=float(b.ap[k]) if hasattr(b, "ap") else None)
    return dict(precision=float(b.mp), recall=float(b.mr), map50=float(b.map50), map50_95=float(b.map),
                per_class=per_class, speed_ms=dict(r.speed))


# ----------------------------------------------------------------------------- 2. depth metrics on val
@torch.no_grad()
def depth_eval(det, head, class_h, meta, stems, a, out_dir):
    vis_dir = out_dir / "val_vis"
    vis_dir.mkdir(exist_ok=True)
    rows, n_gt, n_hit = [], defaultdict(int), defaultdict(int)
    for k, stem in enumerate(stems):
        m = meta["samples"][stem]
        frame = cv2.imread(str(tp.OUT / "images" / "val" / f"{stem}.png"))
        pred = predict(det, head, class_h, frame, m["fy"], a.conf)

        o = np.array(m["objs"], dtype=np.float32).reshape(-1, 8)
        o = o[o[:, 5] <= MAX_D]                         # [cls, x1, y1, x2, y2, depth, trunc, occ]
        gcls, gbox, gdep = o[:, 0].astype(int), o[:, 1:5], o[:, 5]
        for c in gcls:
            n_gt[CLASSES[c]] += 1

        if pred is not None and len(gdep):
            iou = box_iou(torch.from_numpy(gbox), torch.from_numpy(pred[0])).numpy()
            iou[gcls[:, None] != pred[1][None, :]] = 0
            for g in range(len(gdep)):
                j = int(iou[g].argmax())
                if iou[g, j] < 0.5:
                    continue
                n_hit[CLASSES[gcls[g]]] += 1
                h_px, trunc, occ = gbox[g, 3] - gbox[g, 1], o[g, 6], int(o[g, 7])
                e, mo, hd = difficulty(h_px, trunc, occ)
                rows.append(dict(stem=stem, cls=CLASSES[gcls[g]], gt=float(gdep[g]), head=float(pred[2][j]),
                                 geo=float(pred[3][j]), iou=float(iou[g, j]), conf=float(pred[4][j]),
                                 h_px=float(h_px), trunc=float(trunc), occ=occ, easy=e, moderate=mo, hard=hd))
        if k < a.n_vis:
            gts = [((b[0], b[1], b[2], b[3]), d) for b, d in zip(gbox, gdep)]
            cv2.imwrite(str(vis_dir / f"{stem}.jpg"), draw(frame, pred, gts))
    return rows, dict(n_gt), dict(n_hit)


def summarise(rows, n_gt, n_hit):
    gt = [r["gt"] for r in rows]
    S = dict(overall=dict(head=depth_metrics(gt, [r["head"] for r in rows]),
                          geo=depth_metrics(gt, [r["geo"] for r in rows])))
    S["by_distance"], S["by_class"], S["by_difficulty"] = {}, {}, {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = [r for r in rows if lo <= r["gt"] < hi]
        S["by_distance"][f"{lo}-{hi} m"] = dict(head=depth_metrics([r["gt"] for r in sel], [r["head"] for r in sel]),
                                                geo=depth_metrics([r["gt"] for r in sel], [r["geo"] for r in sel]))
    for c in CLASSES:
        sel = [r for r in rows if r["cls"] == c]
        S["by_class"][c] = dict(n_gt=n_gt.get(c, 0), recall=n_hit.get(c, 0) / max(n_gt.get(c, 0), 1),
                                head=depth_metrics([r["gt"] for r in sel], [r["head"] for r in sel]),
                                geo=depth_metrics([r["gt"] for r in sel], [r["geo"] for r in sel]))
    for lvl in ("easy", "moderate", "hard"):
        sel = [r for r in rows if r[lvl]]
        S["by_difficulty"][lvl] = dict(head=depth_metrics([r["gt"] for r in sel], [r["head"] for r in sel]),
                                       geo=depth_metrics([r["gt"] for r in sel], [r["geo"] for r in sel]))
    S["recall_iou50"] = sum(n_hit.values()) / max(sum(n_gt.values()), 1)
    S["n_gt_total"] = sum(n_gt.values())
    return S


# ----------------------------------------------------------------------------- 3. latency
@torch.no_grad()
def bench(det, head, class_h, meta, stems, dev, conf, n=50, warm=10):
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    t_det, t_head = [], []
    for i, stem in enumerate(stems[:n + warm]):
        frame = cv2.imread(str(tp.OUT / "images" / "val" / f"{stem}.png"))
        small = cv2.resize(frame, (IMG_W, IMG_H))
        sync(); t0 = time.perf_counter()
        r = det.yolo.predict(small, imgsz=(IMG_H, IMG_W), conf=conf, verbose=False)[0]   # incl. pre/post-processing
        sync(); t1 = time.perf_counter()
        if len(r.boxes) == 0:
            continue
        feats = det.features(to_tensor(frame)[None].to(dev))
        boxes, cls = r.boxes.xyxy.to(dev), r.boxes.cls.long().to(dev)
        fy = torch.full((len(boxes),), meta["samples"][stem]["fy"] * IMG_H / frame.shape[0], device=dev)
        zg = tp.geo_prior(boxes, cls, fy, class_h)
        sync(); t2 = time.perf_counter()
        head(feats, [boxes], cls, zg)
        sync(); t3 = time.perf_counter()
        if i >= warm:
            t_det.append((t1 - t0) * 1000)
            t_head.append((t3 - t2) * 1000)
    d, h = float(np.mean(t_det)), float(np.mean(t_head))
    return dict(device=str(dev), n=len(t_det), detector_ms=d, depth_head_ms=h, total_ms=d + h,
                fps_with_head=1000 / (d + h), fps_geometric_only=1000 / d,
                peak_vram_mb=float(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else None)


# ----------------------------------------------------------------------------- 4. testing/ images (no labels)
@torch.no_grad()
def run_test_images(det, head, class_h, a, out_dir):
    img_dir, cal_dir = Path(a.test_dir) / "image_2", Path(a.test_dir) / "calib"
    files = sorted(img_dir.glob("*.png"))
    files = files if a.n_test_vis < 0 else files[:a.n_test_vis]
    vis = out_dir / "test_vis"
    vis.mkdir(exist_ok=True)
    rows = []
    for f in files:
        frame = cv2.imread(str(f))
        cal = cal_dir / f"{f.stem}.txt"
        fy = float(tp.read_p2(cal)[1, 1]) if cal.exists() else tp.focal_from_hfov(frame.shape[1], 82.0) \
            if hasattr(tp, "focal_from_hfov") else 721.5
        pred = predict(det, head, class_h, frame, fy, a.conf)
        cv2.imwrite(str(vis / f"{f.stem}.jpg"), draw(frame, pred))
        if pred is not None:
            for (x1, y1, x2, y2), c, d, g, s in zip(*pred):
                rows.append([f.stem, CLASSES[c], f"{s:.3f}", f"{x1:.0f}", f"{y1:.0f}", f"{x2:.0f}", f"{y2:.0f}",
                             f"{d:.2f}", f"{g:.2f}"])
    with open(out_dir / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "class", "conf", "x1", "y1", "x2", "y2", "depth_head_m", "depth_geometric_m"])
        w.writerows(rows)
    return len(files), len(rows)


# ----------------------------------------------------------------------------- 5. plots
def plot_scatter(rows, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    gt = np.array([r["gt"] for r in rows])
    for a_, key, title in zip(ax, ("head", "geo"), ("Depth head", "Geometric (class height)")):
        a_.scatter(gt, [r[key] for r in rows], s=6, alpha=0.4)
        a_.plot([0, MAX_D], [0, MAX_D], "r--", lw=1)
        a_.set_title(title); a_.set_xlabel("GT depth (m)"); a_.set_xlim(0, MAX_D); a_.set_ylim(0, 1.5 * MAX_D)
    ax[0].set_ylabel("Predicted depth (m)")
    fig.tight_layout(); fig.savefig(out_dir / "scatter_pred_vs_gt.png", dpi=150); plt.close(fig)


def plot_bins(S, out_dir):
    labels = list(S["by_distance"])
    h = [(S["by_distance"][k]["head"] or {}).get("abs_rel", np.nan) for k in labels]
    g = [(S["by_distance"][k]["geo"] or {}).get("abs_rel", np.nan) for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, h, 0.4, label="depth head"); ax.bar(x + 0.2, g, 0.4, label="geometric")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("abs rel error (%)"); ax.legend()
    ax.set_title("Depth error by GT distance")
    fig.tight_layout(); fig.savefig(out_dir / "error_by_distance.png", dpi=150); plt.close(fig)


def _col(path, key):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        rd.fieldnames = [f.strip() for f in rd.fieldnames]
        return [float(r[key]) if r.get(key, "").strip() not in ("", None) else np.nan for r in rd]


def plot_curves(a, out_dir):
    dcsv = tp.RUNS / f"{a.name}_depth" / "epochs.csv"
    rcsv = tp.RUNS / f"{a.name}_det" / "results.csv"
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    if dcsv.exists():
        for k, t in (("loss", "Depth loss"), ("err_pct", "Depth error % (<=60 m)")):
            i = 0 if k == "loss" else 1
            tr, va = _col(dcsv, f"train_{k}"), _col(dcsv, f"val_{k}")
            ax[0, i].plot(range(1, len(tr) + 1), tr, label="train")
            ax[0, i].plot(range(1, len(va) + 1), va, "o-", ms=3, label="val")
            ax[0, i].set_title(t); ax[0, i].set_xlabel("epoch"); ax[0, i].legend()
    if rcsv.exists():
        tb, vb = _col(rcsv, "train/box_loss"), _col(rcsv, "val/box_loss")
        ax[1, 0].plot(tb, label="train box"); ax[1, 0].plot(vb, label="val box")
        ax[1, 0].set_title("Detector box loss"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].legend()
        ax[1, 1].plot(_col(rcsv, "metrics/mAP50(B)"), label="mAP50")
        ax[1, 1].plot(_col(rcsv, "metrics/mAP50-95(B)"), label="mAP50-95")
        ax[1, 1].set_title("Detector val mAP"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- 6. report
def write_report(a, out_dir, setup, D, S, B, test_info):
    L = [f"# KITTI YOLO + object-depth report: `{a.name}`", "", "## Setup",
         md_table(["item", "value"], [[k, v] for k, v in setup.items()]), ""]

    L += ["## Detection (Ultralytics, val split, IoU 0.5 / 0.5:0.95)"]
    if "error" in D:
        L += [f"Detector validation failed: `{D['error']}`", ""]
    else:
        L += [md_table(["precision", "recall", "mAP50", "mAP50-95"],
                       [[f"{D['precision']:.3f}", f"{D['recall']:.3f}", f"{D['map50']:.3f}", f"{D['map50_95']:.3f}"]]), "",
              md_table(["class", "AP50", "AP50-95"],
                       [[c, f"{v['ap50']:.3f}", "-" if v["ap"] is None else f"{v['ap']:.3f}"]
                        for c, v in D["per_class"].items()]), ""]

    ov = S["overall"]
    L += [f"## Depth (val, matched detections at conf {a.conf}, IoU >= 0.5, GT <= {MAX_D:.0f} m)",
          f"Recall at IoU 0.5 over {S['n_gt_total']} GT objects: **{S['recall_iou50'] * 100:.1f}%**", "",
          md_table(["method"] + DEPTH_HDR, [["depth head"] + fmt(ov["head"]), ["geometric only"] + fmt(ov["geo"])]), "",
          "### By distance",
          md_table(["bin", "n", "head abs rel %", "head MAE m", "geo abs rel %", "geo MAE m"],
                   [[k, (v["head"] or {}).get("n", 0),
                     *(["-", "-"] if v["head"] is None else [f"{v['head']['abs_rel']:.2f}", f"{v['head']['mae']:.2f}"]),
                     *(["-", "-"] if v["geo"] is None else [f"{v['geo']['abs_rel']:.2f}", f"{v['geo']['mae']:.2f}"])]
                    for k, v in S["by_distance"].items()]), "",
          "### By class",
          md_table(["class", "GT", "recall", "head abs rel %", "geo abs rel %"],
                   [[c, v["n_gt"], f"{v['recall'] * 100:.1f}%",
                     "-" if v["head"] is None else f"{v['head']['abs_rel']:.2f}",
                     "-" if v["geo"] is None else f"{v['geo']['abs_rel']:.2f}"] for c, v in S["by_class"].items()]), "",
          "### By KITTI difficulty (cumulative)",
          md_table(["level", "n", "head abs rel %", "geo abs rel %"],
                   [[k, (v["head"] or {}).get("n", 0),
                     "-" if v["head"] is None else f"{v['head']['abs_rel']:.2f}",
                     "-" if v["geo"] is None else f"{v['geo']['abs_rel']:.2f}"] for k, v in S["by_difficulty"].items()]), "",
          "## Latency",
          md_table(["device", "images", "detector ms", "depth head ms", "total ms", "FPS (with depth)",
                    "FPS (geometric only)", "peak VRAM MB"],
                   [[B["device"], B["n"], f"{B['detector_ms']:.1f}", f"{B['depth_head_ms']:.1f}", f"{B['total_ms']:.1f}",
                     f"{B['fps_with_head']:.1f}", f"{B['fps_geometric_only']:.1f}",
                     "-" if B["peak_vram_mb"] is None else f"{B['peak_vram_mb']:.0f}"]]),
          "Detector ms includes Ultralytics pre/post-processing; the head is timed separately on cached features "
          "(the current inference path runs the backbone twice, so true single-pass cost is detector + head).", "",
          "## Qualitative",
          f"- `val_vis/`: {a.n_vis} val images with predicted box, learned depth, geometric depth and GT depth.",
          f"- `test_vis/`: {test_info[0]} testing/ images, {test_info[1]} detections (no GT exists for this split).",
          "- Figures: `scatter_pred_vs_gt.png`, `error_by_distance.png`, `training_curves.png`.", "",
          "## Caveats",
          "- The val split also selected `best.pt`, so these numbers are slightly optimistic. A frame-level random "
          "split of KITTI leaks neighbouring frames between train and val.",
          "- Geometric baseline uses the per-class mean real height from the training labels and the 2D box height. "
          "Truncated or occluded boxes bias it high.",
          "- GT depth is the minimum z over the 8 corners of the 3D box (YOLO MDE definition), not the box centre."]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp1", help="experiment tag used in training")
    ap.add_argument("--det_weights", default=None)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val_n", type=int, default=-1, help="limit val images for depth metrics (-1 = all)")
    ap.add_argument("--n_vis", type=int, default=30, help="val images to save with annotations")
    ap.add_argument("--n_test_vis", type=int, default=30, help="testing/ images to annotate (-1 = all)")
    ap.add_argument("--test_dir", default=str(tp.ROOT / "testing"))
    ap.add_argument("--skip_det_val", action="store_true", help="skip the Ultralytics mAP pass")
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA requested but unavailable; use --device cpu")
    a.det_weights = a.det_weights or str(tp.RUNS / f"{a.name}_det" / "weights" / "best.pt")
    out_dir = tp.RUNS / f"{a.name}_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    meta = json.load(open(tp.OUT / "depth_labels.json"))
    stems = sorted(s for s, m in meta["samples"].items() if m["split"] == "val")
    if a.val_n > 0:
        stems = stems[:a.val_n]

    cfg = {}
    for tag in ("det", "depth"):
        p = tp.RUNS / f"{a.name}_{tag}" / "config.json"
        cfg[tag] = json.load(open(p)) if p.exists() else {}
    imgsz = cfg["det"].get("imgsz", IMG_W)
    da = cfg["det"].get("args", {})
    setup = {"detector weights": a.det_weights, "detector imgsz": imgsz, "depth input": f"{IMG_H}x{IMG_W}",
             "frozen layers": cfg["det"].get("freeze", "n/a"), "train subset": da.get("subset", "n/a"),
             "detector epochs": da.get("det_epochs", "n/a"), "depth epochs": cfg["depth"].get("args", {}).get("depth_epochs", "n/a"),
             "geometric prior in head": cfg["depth"].get("USE_GEO_PRIOR", "n/a"), "val images": len(stems),
             "conf threshold": a.conf, "device": a.device}

    print("[1/5] detector metrics ...")
    D = dict(error="skipped") if a.skip_det_val else detection_metrics(a, out_dir, imgsz)

    det = tp.FrozenDetector(a.det_weights, dev)
    head, class_h = tp.load_head(a, dev)

    print("[2/5] depth metrics on val ...")
    rows, n_gt, n_hit = depth_eval(det, head, class_h, meta, stems, a, out_dir)
    S = summarise(rows, n_gt, n_hit)
    with open(out_dir / "per_object.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)

    print("[3/5] latency ...")
    B = bench(det, head, class_h, meta, stems, dev, a.conf)

    print("[4/5] testing/ images ...")
    test_info = run_test_images(det, head, class_h, a, out_dir) if (Path(a.test_dir) / "image_2").exists() else (0, 0)

    print("[5/5] plots + report ...")
    if rows:
        plot_scatter(rows, out_dir)
        plot_bins(S, out_dir)
    plot_curves(a, out_dir)
    json.dump(dict(setup=setup, detection=D, depth=S, latency=B), open(out_dir / "metrics.json", "w"), indent=2)
    write_report(a, out_dir, setup, D, S, B, test_info)

    ov = S["overall"]
    msg = (f"head {ov['head']['abs_rel']:.2f}% vs geometric {ov['geo']['abs_rel']:.2f}%" if ov["head"]
           else "no matched detections (check --conf and the detector weights)")
    print(f"\nrecall@IoU0.5 {S['recall_iou50'] * 100:.1f}% | depth abs-rel: {msg} | "
          f"{B['fps_with_head']:.1f} FPS ({B['device']})\nreport -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
