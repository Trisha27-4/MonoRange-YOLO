"""
training_pipeline.py  -  KITTI: YOLO detector + object-level depth head
Follows Yu & Choi, "YOLO MDE" (Electronics 2022), adapted to a frozen-detector setup.

Stages
  layers  : print the layer table of the model so you can choose what to freeze
  prepare : KITTI -> YOLO format, GT depth = min z over the 8 corners of the 3D box,
            6980 / 500 train/val split (paper), per-class mean real height
  det     : fine-tune YOLO on KITTI, backbone frozen (Ultralytics `freeze`)
  depth   : freeze whole detector, train ROI depth head on its neck features
            (Huber + relative loss, log-sigmoid activation, geometric prior as input + residual base)
  test    : smoke test - 15 epochs on 1/15th (or --subset) of the data, runs det+depth+eval
  eval    : depth error on val GT boxes + end-to-end error on matched detections

Run:  python training_pipeline.py --stage all
Expected layout next to this file:  training/{calib,image_2,label_2,velodyne}
"""
import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, roi_align
from ultralytics import YOLO

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
KITTI = ROOT / "training"
OUT = ROOT / "kitti_yolo"                      # converted dataset
RUNS = ROOT / "runs"

CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}  # 'DontCare' is skipped

IMG_H, IMG_W = 384, 1248       # paper's large model input; both multiples of 32
VAL_N, SEED = 500, 0           # paper: 6980 train / 500 val
MIN_DEPTH, EVAL_MAX_DEPTH = 0.5, 60.0

DET_WEIGHTS = "yolo26n.pt"
FREEZE_DEFAULT = "10"          # int N = freeze layers 0..N-1, or comma list '0,1,2'. Inspect with --stage layers

BETA = 14.4                    # paper: D = -14.4 * log(sigmoid(o))
USE_GEO_PRIOR = True           # geometric distance: Z = fy * H_class / h_px, used as input + base of the output
LAMBDA_GEO = 0.0               # optional log-space consistency pull toward the geometric prior
BOX_JITTER = 0.10              # train-time box noise so the head tolerates imperfect detector boxes


# ----------------------------------------------------------------------------- KITTI parsing
def read_p2(calib_path):
    for line in open(calib_path):
        if line.startswith("P2:"):
            return np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
    raise ValueError(f"P2 not found in {calib_path}")


def min_corner_depth(dims, loc, ry):
    """Paper Eq.7: D = min_i z_i over the 8 vertices of the 3D box (camera coords, z = forward)."""
    h, w, l = dims
    x, y, z = loc                                       # bottom-centre of the box
    xs = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    zs = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    c, s = np.cos(ry), np.sin(ry)
    z_cam = z + (-s * xs + c * zs)                      # rotation about Y, then translate
    return float(z_cam.min())


def parse_label(path, img_w, img_h):
    objs = []
    for line in open(path):
        p = line.split()
        if p[0] not in CLS2ID:                          # DontCare
            continue
        x1, y1, x2, y2 = map(float, p[4:8])
        h, w, l = map(float, p[8:11])
        loc = tuple(map(float, p[11:14]))
        ry = float(p[14])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
        d = min_corner_depth((h, w, l), loc, ry)
        if x2 - x1 < 2 or y2 - y1 < 2 or d < MIN_DEPTH:
            continue
        objs.append(dict(cls=CLS2ID[p[0]], box=[x1, y1, x2, y2], depth=d, height=h,
                         trunc=float(p[1]), occ=int(p[2])))
    return objs


# ----------------------------------------------------------------------------- stage 0: prepare
def prepare():
    stems = sorted(p.stem for p in (KITTI / "image_2").glob("*.png"))
    random.Random(SEED).shuffle(stems)
    split = {"val": stems[:VAL_N], "train": stems[VAL_N:]}

    meta, heights = {}, {i: [] for i in range(len(CLASSES))}
    for name, items in split.items():
        (OUT / "images" / name).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / name).mkdir(parents=True, exist_ok=True)
        for stem in items:
            src = KITTI / "image_2" / f"{stem}.png"
            dst = OUT / "images" / name / f"{stem}.png"
            if not dst.exists():
                try:
                    os.link(src, dst)                   # no extra disk use
                except OSError:
                    shutil.copy2(src, dst)
            w, h = Image.open(src).size
            objs = parse_label(KITTI / "label_2" / f"{stem}.txt", w, h)
            with open(OUT / "labels" / name / f"{stem}.txt", "w") as f:
                for o in objs:
                    x1, y1, x2, y2 = o["box"]
                    f.write(f"{o['cls']} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                            f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")
                    if name == "train":
                        heights[o["cls"]].append(o["height"])
            meta[stem] = dict(split=name, W=w, H=h, fy=float(read_p2(KITTI / "calib" / f"{stem}.txt")[1, 1]),
                              objs=[[o["cls"], *o["box"], o["depth"], o["trunc"], o["occ"]] for o in objs])

    class_h = [float(np.mean(heights[i])) if heights[i] else 1.5 for i in range(len(CLASSES))]
    json.dump(dict(class_h=class_h, samples=meta), open(OUT / "depth_labels.json", "w"))

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (OUT / "kitti.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"train {len(split['train'])} | val {len(split['val'])} | class heights (m): "
          + ", ".join(f"{c}={h:.2f}" for c, h in zip(CLASSES, class_h)))


# ----------------------------------------------------------------------------- stage 1: detector
def parse_freeze(txt):
    txt = str(txt).strip()
    return [int(x) for x in txt.split(",")] if "," in txt else int(txt)


def list_layers(weights=DET_WEIGHTS, freeze=FREEZE_DEFAULT):
    """Layer table of the YOLO model; marks what a given --freeze setting would lock."""
    fr = parse_freeze(freeze)
    frozen = set(range(fr)) if isinstance(fr, int) else set(fr)
    net = YOLO(weights).model
    tot = trn = 0
    print(f"{'idx':>3}  {'module':<14}{'params':>12}  status")
    for i, m in enumerate(net.model):
        n = sum(p.numel() for p in m.parameters())
        tot += n
        trn += 0 if i in frozen else n
        print(f"{i:>3}  {type(m).__name__:<14}{n:>12,}  {'FROZEN' if i in frozen else 'trainable'}")
    print(f"total {tot:,} | trainable {trn:,} | frozen {tot - trn:,}  (Ultralytics also always freezes DFL)")


def train_detector(a):
    freeze = parse_freeze(a.freeze)
    list_layers(DET_WEIGHTS, a.freeze)
    run = f"{a.name}_det"
    data = OUT / "kitti.yaml"
    vf = val_frac(a)
    if a.subset < 1.0 or vf < 1.0:   # same seeded subsets as the depth stage (Ultralytics `fraction` would pick different images)
        meta = json.load(open(OUT / "depth_labels.json"))
        yaml_txt = (OUT / "kitti.yaml").read_text()
        for split, frac in (("train", a.subset), ("val", vf)):
            if frac < 1.0:
                stems = KittiDepthDataset(split, meta, False, frac).stems
                lst = OUT / f"{split}_subset_{frac:.3f}.txt"
                lst.write_text("\n".join((OUT / "images" / split / f"{x}.png").as_posix() for x in stems))
                yaml_txt = yaml_txt.replace(f"{split}: images/{split}", f"{split}: {lst.as_posix()}")
                print(f"[subset] detector {split}: {len(stems)} images")
        data = OUT / f"kitti_subset_{a.subset:.3f}_{vf:.3f}.yaml"
        data.write_text(yaml_txt)
    model = YOLO(DET_WEIGHTS)
    if a.val_every > 1:       # validate only every N-th epoch (+ the final one); relies on Ultralytics trainer internals
        model.add_callback("on_train_epoch_end",
                           lambda tr: setattr(tr.args, "val", (tr.epoch + 1) % a.val_every == 0))
    model.train(data=str(data), cache="ram" if a.subset < 1.0 else False, imgsz=a.det_imgsz, epochs=a.det_epochs, batch=a.batch,
                freeze=freeze, lr0=1e-3, project=str(RUNS), name=run, exist_ok=True,
                device=a.device or None, workers=a.workers)
    # Ultralytics writes results.csv every epoch: train/val box, cls, dfl losses + P, R, mAP50, mAP50-95.
    # Keep a stable copy and the experiment config next to it for reporting.
    run_dir = RUNS / run
    json.dump(dict(args=vars(a), freeze=freeze, weights=DET_WEIGHTS, imgsz=a.det_imgsz),
              open(run_dir / "config.json", "w"), indent=2)
    print(f"per-epoch detector metrics: {run_dir / 'results.csv'}  (strip header spaces when loading)")


# ----------------------------------------------------------------------------- depth model
class FrozenDetector:
    """Wraps a trained YOLO and captures the multi-scale neck features fed to the Detect head."""

    def __init__(self, weights, device):
        self.yolo = YOLO(weights)
        self.net = self.yolo.model.to(device).float().eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.feats = None
        self.net.model[-1].register_forward_pre_hook(lambda m, args: setattr(self, "feats", list(args[0])))
        with torch.no_grad():
            self.net(torch.zeros(1, 3, IMG_H, IMG_W, device=device))
        self.chans = [f.shape[1] for f in self.feats]
        self.strides = [IMG_H // f.shape[2] for f in self.feats]

    @torch.no_grad()
    def features(self, imgs):
        self.net(imgs)
        return self.feats


def softplus_inv(y):
    return torch.log(torch.expm1(y.clamp(1e-4, 30.0)))


class ObjectDepthHead(nn.Module):
    """ROI-align on each neck level -> one scalar depth per box. D = BETA * softplus(-o) = -BETA*log(sigmoid(o))."""

    def __init__(self, chans, strides, n_cls, pool=7, use_geo_prior=USE_GEO_PRIOR):
        super().__init__()
        self.strides, self.pool, self.use_geo_prior = strides, pool, use_geo_prior
        self.reduce = nn.ModuleList(nn.Sequential(nn.Conv2d(c, 64, 1), nn.BatchNorm2d(64), nn.SiLU()) for c in chans)
        self.mix = nn.Sequential(nn.Conv2d(64 * len(chans), 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU())
        self.cls_emb = nn.Embedding(n_cls, 16)
        self.fc = nn.Sequential(nn.Linear(128 * pool * pool + 16 + 4, 256), nn.SiLU(),
                                nn.Dropout(0.1), nn.Linear(256, 1))
        nn.init.zeros_(self.fc[-1].weight)               # with geo prior: starts exactly at the geometric estimate
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, feats, boxes, cls, z_geo):
        rois = [roi_align(r(f), boxes, self.pool, spatial_scale=1.0 / s, sampling_ratio=2, aligned=True)
                for r, f, s in zip(self.reduce, feats, self.strides)]
        x = self.mix(torch.cat(rois, 1)).flatten(1)
        b = torch.cat(boxes, 0)
        side = torch.stack([torch.log(z_geo), (b[:, 0] + b[:, 2]) / (2 * IMG_W),
                            (b[:, 1] + b[:, 3]) / (2 * IMG_H), (b[:, 3] - b[:, 1]) / IMG_H], 1)
        o = self.fc(torch.cat([x, self.cls_emb(cls), side], 1)).squeeze(1)
        if self.use_geo_prior:
            o = o - softplus_inv(z_geo / BETA)
        return BETA * F.softplus(-o)


def geo_prior(boxes, cls, fy, class_h):
    """Pinhole: Z = fy * H_real / h_px  (fy already scaled to the resized image)."""
    h_px = (boxes[:, 3] - boxes[:, 1]).clamp(min=2.0)
    return (fy * class_h[cls] / h_px).clamp(MIN_DEPTH, 150.0)


def depth_loss(pred, gt, z_geo):
    huber = F.huber_loss(pred, gt, delta=1.0)             # paper Eq.4 (delta = 1)
    rel = ((gt - pred).abs() / gt).mean()                 # paper Eq.5
    geo = (torch.log(pred) - torch.log(z_geo)).abs().mean()
    return huber + rel + LAMBDA_GEO * geo, huber, rel


# ----------------------------------------------------------------------------- data

class KittiDepthDataset(Dataset):
    def __init__(self, split, meta, train, frac=1.0):
        self.split, self.meta, self.train = split, meta["samples"], train
        self.stems = sorted(s for s, m in self.meta.items() if m["split"] == split)
        if frac < 1.0:                                   # seeded random subset (same subset every run)
            random.Random(SEED).shuffle(self.stems)
            self.stems = sorted(self.stems[:max(8, int(len(self.stems) * frac))])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem, m = self.stems[i], self.meta[self.stems[i]]
        img = cv2.imread(str(OUT / "images" / self.split / f"{stem}.png"))
        img = cv2.cvtColor(cv2.resize(img, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
        sx, sy = IMG_W / m["W"], IMG_H / m["H"]
        o = np.array(m["objs"], dtype=np.float32).reshape(-1, 8)
        cls, box, depth = o[:, 0].astype(np.int64), o[:, 1:5].copy(), o[:, 5].copy()
        box[:, [0, 2]] *= sx
        box[:, [1, 3]] *= sy
        if self.train:
            if random.random() < 0.5:                      # depth is invariant to horizontal flip
                img = img[:, ::-1].copy()
                box[:, [0, 2]] = IMG_W - box[:, [2, 0]]
            wh = box[:, 2:] - box[:, :2]
            box += (np.random.rand(len(box), 4) - 0.5) * BOX_JITTER * np.tile(wh, (1, 2))
            box[:, [0, 2]] = box[:, [0, 2]].clip(0, IMG_W)
            box[:, [1, 3]] = box[:, [1, 3]].clip(0, IMG_H)
        return (torch.from_numpy(img).permute(2, 0, 1).float() / 255, torch.from_numpy(box),
                torch.from_numpy(cls), torch.from_numpy(depth), m["fy"] * sy)


def collate(batch):
    imgs, boxes, cls, depth, fy = zip(*batch)
    return torch.stack(imgs), list(boxes), list(cls), list(depth), list(fy)


# ----------------------------------------------------------------------------- stage 2: depth training
def run_epoch(det, head, loader, class_h, dev, opt=None):
    train = opt is not None
    head.train(train)
    tot = dict(loss=0.0, huber=0.0, rel=0.0,            err=0.0, n=0, n_err=0)
    for imgs, boxes, cls, depth, fy in loader:
        if sum(len(b) for b in boxes) == 0:
            continue
        feats = det.features(imgs.to(dev))
        boxes = [b.to(dev) for b in boxes]
        cat_cls = torch.cat(cls).to(dev)
        gt = torch.cat(depth).to(dev)
        fy_box = torch.cat([torch.full((len(b),), f, device=dev) for b, f in zip(boxes, fy)])
        z_geo = geo_prior(torch.cat(boxes), cat_cls, fy_box, class_h)
        with torch.set_grad_enabled(train):
            pred = head(feats, boxes, cat_cls, z_geo)
            loss, hub, rel = depth_loss(pred, gt, z_geo)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        n = len(gt)
        keep = gt <= EVAL_MAX_DEPTH

        tot["loss"] += loss.item() * n
        tot["huber"] += hub.item() * n
        tot["rel"] += rel.item() * n

        tot["err"] += ((gt - pred).abs() / gt)[keep].sum().item()
        tot["n_err"] += int(keep.sum())
        tot["n"] += n
        
    n = max(tot["n"], 1)
    return {
    "loss": tot["loss"] / max(tot["n"], 1),
    "huber": tot["huber"] / max(tot["n"], 1),
    "rel": tot["rel"] / max(tot["n"], 1),
    "err": tot["err"] / max(tot["n_err"], 1)
}


def val_frac(a):
    """Val fraction: --val_subset if given; else full val, except in the smoke test where it follows --subset."""
    if a.val_subset > 0:
        return a.val_subset
    return a.subset if a.stage == "test" else 1.0


def depth_dir(a):
    d = RUNS / f"{a.name}_depth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_depth(a):
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = json.load(open(OUT / "depth_labels.json"))
    class_h = torch.tensor(meta["class_h"], device=dev)
    det = FrozenDetector(a.det_weights, dev)
    head = ObjectDepthHead(det.chans, det.strides, len(CLASSES)).to(dev)

    tr = DataLoader(KittiDepthDataset("train", meta, True, a.subset), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, collate_fn=collate, drop_last=True)
    va = DataLoader(KittiDepthDataset("val", meta, False, val_frac(a)), batch_size=a.batch, shuffle=False,
                    num_workers=a.workers, collate_fn=collate)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)   # paper: x0.1 every 20 epochs

    out = depth_dir(a)
    json.dump(dict(args=vars(a), BETA=BETA, USE_GEO_PRIOR=USE_GEO_PRIOR, LAMBDA_GEO=LAMBDA_GEO,
                   BOX_JITTER=BOX_JITTER, IMG_HW=[IMG_H, IMG_W], VAL_N=VAL_N, SEED=SEED,
                   head_params=sum(p.numel() for p in head.parameters())),
              open(out / "config.json", "w"), indent=2)
    fields = ["epoch", "lr", "train_loss", "train_huber", "train_rel", "train_err_pct",
              "val_loss", "val_huber", "val_rel", "val_err_pct", "epoch_sec"]
    best = float("inf")
    with open(out / "epochs.csv", "w", newline="") as fh:
        log = csv.DictWriter(fh, fieldnames=fields)
        log.writeheader()
        for ep in range(1, a.depth_epochs + 1):
            t0, lr = time.time(), opt.param_groups[0]["lr"]
            t = run_epoch(det, head, tr, class_h, dev, opt)
            do_val = ep % a.val_every == 0 or ep == a.depth_epochs
            v = run_epoch(det, head, va, class_h, dev) if do_val else None
            sched.step()
            log.writerow(dict(epoch=ep, lr=lr, train_loss=t["loss"], train_huber=t["huber"], train_rel=t["rel"],
                              train_err_pct=t["err"] * 100,
                              val_loss=v["loss"] if v else "", val_huber=v["huber"] if v else "",
                              val_rel=v["rel"] if v else "", val_err_pct=v["err"] * 100 if v else "",
                              epoch_sec=time.time() - t0))
            fh.flush()                                   # survives a crash / Ctrl-C
            print(f"ep {ep:3d} | train loss {t['loss']:.3f} err {t['err']*100:.2f}% | "
                  + (f"val loss {v['loss']:.3f} err@<=60m {v['err']*100:.2f}%" if v else "val skipped"))
            ck = dict(head=head.state_dict(), chans=det.chans, strides=det.strides,
                      class_h=meta["class_h"], det_weights=a.det_weights, epoch=ep)
            torch.save(ck, out / "last.pt")
            if v and v["loss"] < best:
                best = v["loss"]
                torch.save(ck, out / "best.pt")
    print(f"best val loss {best:.3f} | logs + checkpoints in {out}")


# ----------------------------------------------------------------------------- stage 3: evaluation
def load_head(a, dev):
    ck = torch.load(depth_dir(a) / "best.pt", map_location=dev)
    head = ObjectDepthHead(ck["chans"], ck["strides"], len(CLASSES)).to(dev)
    head.load_state_dict(ck["head"])
    return head.eval(), torch.tensor(ck["class_h"], device=dev)


@torch.no_grad()
def predict_depths(det, head, class_h, img_rgb_tensor, fy, conf):
    """Detector boxes -> depth head. Image is already IMG_H x IMG_W, so Ultralytics letterbox is identity."""
    dev = next(head.parameters()).device
    bgr = (img_rgb_tensor.permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8).copy()
    r = det.yolo.predict(bgr, imgsz=(IMG_H, IMG_W), conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    boxes, cls = r.boxes.xyxy.to(dev), r.boxes.cls.long().to(dev)
    feats = det.features(img_rgb_tensor[None].to(dev))      # own forward so hooked features match this image
    z_geo = geo_prior(boxes, cls, torch.full((len(boxes),), fy, device=dev), class_h)
    depth = head(feats, [boxes], cls, z_geo)
    return boxes, cls, depth, z_geo


def evaluate(a):
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = json.load(open(OUT / "depth_labels.json"))
    det = FrozenDetector(a.det_weights, dev)
    head, class_h = load_head(a, dev)
    ds = KittiDepthDataset("val", meta, False, val_frac(a))

    # (a) GT boxes -> isolates depth-head quality
    va = DataLoader(ds, batch_size=a.batch, collate_fn=collate)
    v = run_epoch(det, head, va, class_h, dev)
    print(f"[GT boxes]   depth error rate (<= {EVAL_MAX_DEPTH:.0f} m): {v['err']*100:.2f}%")
    res = dict(gt_boxes_err_pct=v["err"] * 100, gt_boxes_loss=v["loss"])

    # (b) detector boxes matched to GT (IoU >= 0.5, same class) -> paper's protocol
    errs, geo_errs, n_gt = [], [], 0
    for i in range(len(ds)):
        img, gbox, gcls, gdep, fy = ds[i]
        keep = gdep <= EVAL_MAX_DEPTH
        gbox, gcls, gdep = gbox[keep], gcls[keep], gdep[keep]
        n_gt += len(gdep)
        out = predict_depths(det, head, class_h, img, fy, a.conf)
        if out is None or len(gdep) == 0:
            continue
        pbox, pcls, pdep, pgeo = [t.cpu() for t in out]
        iou = box_iou(gbox, pbox)
        iou[gcls[:, None] != pcls[None, :]] = 0
        best_iou, best_j = iou.max(1)
        for g in torch.nonzero(best_iou >= 0.5).flatten():
            j = best_j[g]
            errs.append(abs(gdep[g] - pdep[j]).item() / gdep[g].item())
            geo_errs.append(abs(gdep[g] - pgeo[j]).item() / gdep[g].item())
    if errs:
        res.update(matched=len(errs), n_gt=n_gt, recall_iou50=len(errs) / max(n_gt, 1), conf=a.conf,
                   head_err_pct=float(np.mean(errs) * 100), geo_only_err_pct=float(np.mean(geo_errs) * 100))
    print(f"[detections] {res}")
    json.dump(res, open(depth_dir(a) / "eval.json", "w"), indent=2)


# ----------------------------------------------------------------------------- smoke test
def test_train(a):
    """Quick end-to-end check: prepare (if needed) -> detector -> depth head -> eval on a small subset."""
    t0 = time.time()
    a.det_epochs = a.depth_epochs = a.test_epochs
    a.subset = a.subset if a.subset < 1.0 else 1 / 15
    a.name = f"{a.name}_test"
    a.det_weights = str(RUNS / f"{a.name}_det" / "weights" / "best.pt")
    print(f"[test] {a.test_epochs} epochs on {a.subset:.3f} of the data | tag '{a.name}'")
    if not (OUT / "depth_labels.json").exists():
        prepare()
    train_detector(a)
    train_depth(a)
    evaluate(a)
    print(f"[test] finished in {(time.time() - t0) / 60:.1f} min -> inspect runs/{a.name}_det and runs/{a.name}_depth")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["layers", "prepare", "det", "depth", "eval", "all", "test"])
    ap.add_argument("--name", default="exp1", help="experiment tag; outputs go to runs/<name>_det and runs/<name>_depth")
    ap.add_argument("--freeze", default=FREEZE_DEFAULT, help="int N (layers 0..N-1) or comma list of layer indices")
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of the TRAIN data to use (e.g. 0.10)")
    ap.add_argument("--det_imgsz", type=int, default=IMG_W, help="detector train size (multiple of 32); lower it, e.g. 960, if VRAM is tight")
    ap.add_argument("--val_every", type=int, default=1, help="validate every N epochs (both stages); final epoch always validated")
    ap.add_argument("--val_subset", type=float, default=-1.0, help="val fraction; default = full val (or --subset in --stage test)")
    ap.add_argument("--test_epochs", type=int, default=15)
    ap.add_argument("--det_epochs", type=int, default=60)
    ap.add_argument("--depth_epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)       # head-only; paper used 1e-4 for the full network
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 (default), cuda:1, or cpu")
    ap.add_argument("--det_weights", default=None)
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False. Installed torch: "
                         f"{torch.__version__} (a '+cpu' suffix means the CPU-only build). See install steps.")
    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        print(f"[device] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    a.det_weights = a.det_weights or str(RUNS / f"{a.name}_det" / "weights" / "best.pt")

    if a.stage == "test":
        test_train(a)
        raise SystemExit

    if a.stage == "layers":
        list_layers(DET_WEIGHTS, a.freeze)

    if a.stage in ("prepare", "all"):
        prepare()
    if a.stage in ("det", "all"):
        train_detector(a)
    if a.stage in ("depth", "all"):
        train_depth(a)
    if a.stage in ("eval", "all"):
        evaluate(a)
