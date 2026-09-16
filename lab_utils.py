"""Scoring engine for the Day-5 segmentation lab.

Compares a student's CVAT export against golden ground truth and returns metrics,
a tier score, and anti-cheat flags. Two task families:

  semantic             <- CVAT "Segmentation mask 1.1"  (color PNGs + labelmap.txt)
  instance / panoptic  <- CVAT "COCO 1.0"               (annotations/*.json)

Design notes
- Metric is capped at HUMAN_CAP (0.85): two good annotators only agree ~0.85-0.92
  (Segment Anything §5), so 0.85 already earns full marks. Scores above CAP add
  nothing and scores at/above SUSPECT (0.985) raise a possible-cheating flag.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

HUMAN_CAP = 0.85      # IoU/mIoU value that earns full tier points (human agreement)
PQ_CAP = 0.65         # panoptic PQ is stricter; good human-vs-human PQ ~0.6-0.7
FLOOR = 0.40          # below this earns ~0 (a rough box, not a mask)
PQ_FLOOR = 0.20       # PQ floor (a rough attempt)
SUSPECT = 0.985       # at/above this -> possible ground-truth reuse (IoU tasks)
PQ_SUSPECT = 0.95     # near-perfect PQ is also implausible for human labeling

# ---------------------------------------------------------------------------
# zip helpers
# ---------------------------------------------------------------------------
def _open_zip(path):
    return zipfile.ZipFile(path) if str(path).endswith(".zip") else None


def _read_members(path):
    """Return {arcname: bytes} for a zip, or files under a directory."""
    p = Path(path)
    if p.is_dir():
        return {str(f.relative_to(p)): f.read_bytes() for f in p.rglob("*") if f.is_file()}
    with zipfile.ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist() if not n.endswith("/")}


# ---------------------------------------------------------------------------
# semantic / panoptic
# ---------------------------------------------------------------------------
def _parse_labelmap(text):
    """labelmap.txt 'name:R,G,B::' -> {(r,g,b): name}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, rgb = line.split(":")[0], line.split(":")[1]
        if not rgb:
            continue
        try:
            r, g, b = (int(x) for x in rgb.split(","))
        except ValueError:
            continue
        out[(r, g, b)] = name
    return out


def load_semantic_submission(path, class_to_id):
    """CVAT Segmentation-mask-1.1 -> {image_stem: HxW int label map (tier ids, else 255)}."""
    members = _read_members(path)
    lm_key = next((k for k in members if k.endswith("labelmap.txt")), None)
    if not lm_key:
        raise ValueError("labelmap.txt not found — export as 'Segmentation mask 1.1'")
    color2name = _parse_labelmap(members[lm_key].decode("utf-8", "ignore"))
    out = {}
    for k, b in members.items():
        if "SegmentationClass/" not in k or not k.lower().endswith(".png"):
            continue
        stem = Path(k).stem
        rgb = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        lab = np.full(rgb.shape[:2], 255, np.int32)
        for (r, g, b_), name in color2name.items():
            if name in class_to_id:
                lab[(rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b_)] = class_to_id[name]
        out[stem] = lab
    return out


def load_semantic_gt(gt_dir, keep_ids):
    out = {}
    for f in Path(gt_dir).glob("*.png"):
        lab = np.array(Image.open(f)).astype(np.int32)
        lab[~np.isin(lab, list(keep_ids))] = 255
        out[f.stem] = lab
    return out


def score_semantic(sub, gt, id_to_name):
    """Per-class IoU + mIoU + coverage over GT non-ignore pixels."""
    ids = sorted(id_to_name)
    inter = {i: 0 for i in ids}; union = {i: 0 for i in ids}
    covered = total = 0
    for stem, g in gt.items():
        s = sub.get(stem)
        if s is None:
            total += int((g != 255).sum())      # nothing submitted -> all missed
            continue
        if s.shape != g.shape:
            s = np.array(Image.fromarray(s.astype(np.uint8)).resize(
                (g.shape[1], g.shape[0]), Image.NEAREST)).astype(np.int32)
        valid = g != 255
        total += int(valid.sum())
        covered += int(((s != 255) & valid).sum())
        for i in ids:
            gi = (g == i); si = (s == i)
            inter[i] += int((gi & si).sum())
            union[i] += int((gi | si).sum())
    per_class = {id_to_name[i]: (inter[i] / union[i] if union[i] else None) for i in ids}
    present = [v for v in per_class.values() if v is not None]
    miou = float(np.mean(present)) if present else 0.0
    return {
        "metric": "mIoU", "value": miou,
        "per_class_iou": per_class,
        "coverage": (covered / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# instance
# ---------------------------------------------------------------------------
def _find_coco_json(members):
    cand = [k for k in members if k.endswith(".json") and "annotation" in k.lower()]
    if not cand:
        cand = [k for k in members if k.endswith(".json")]
    if not cand:
        raise ValueError("no COCO json found — export as 'COCO 1.0'")
    return json.loads(members[sorted(cand, key=len)[0]].decode("utf-8"))


def _rasterize(ann, h, w):
    seg = ann.get("segmentation")
    if seg is None:
        return None
    if isinstance(seg, list):                     # polygons
        if not seg:
            return None
        rles = cocomask.frPyObjects(seg, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(seg.get("counts"), list):     # uncompressed RLE
        rle = cocomask.frPyObjects(seg, h, w)
    else:                                          # compressed RLE
        rle = seg
    return cocomask.decode(rle).astype(bool)


def _instances_by_image(coco, class_names):
    imgs = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    out = {}                                       # file_stem -> [(class, mask)]
    for a in coco["annotations"]:
        cls = cats.get(a["category_id"])
        if cls not in class_names:
            continue
        im = imgs[a["image_id"]]
        m = _rasterize(a, im["height"], im["width"])
        if m is None or m.sum() == 0:
            continue
        out.setdefault(Path(im["file_name"]).stem, []).append((cls, m, a.get("segmentation")))
    return out


def _iou(a, b):
    i = int((a & b).sum()); u = int((a | b).sum())
    return i / u if u else 0.0


def score_instance(sub_coco, gt_coco, class_names):
    """Greedy IoU matching per class. Returns mean matched IoU, P/R@0.5, count error."""
    sub = _instances_by_image(sub_coco, class_names)
    gt = _instances_by_image(gt_coco, class_names)
    matched_ious, tp = [], 0
    fp = fn = 0
    gt_count = sub_count = 0
    identical = 0
    for stem, g_list in gt.items():
        s_list = sub.get(stem, [])
        gt_count += len(g_list); sub_count += len(s_list)
        used = set()
        # greedy: for each GT, take best free student mask of same class
        for gi, (gc, gm, gseg) in enumerate(g_list):
            best, bj = 0.0, -1
            for j, (sc, sm, sseg) in enumerate(s_list):
                if j in used or sc != gc:
                    continue
                v = _iou(gm, sm)
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.5:
                used.add(bj); tp += 1; matched_ious.append(best)
                if best >= 0.999 or _seg_equal(gseg, s_list[bj][2]):
                    identical += 1
            else:
                fn += 1
        fp += len(s_list) - len(used)
    for stem, s_list in sub.items():               # student images with no GT match at all
        if stem not in gt:
            fp += len(s_list); sub_count += len(s_list)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # combined metric: matched-mask quality gated by recall (missing objects hurt)
    metric = mean_iou * recall
    return {
        "metric": "mean_matched_IoU x recall", "value": metric,
        "mean_matched_iou": mean_iou, "precision@0.5": precision, "recall@0.5": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "gt_instances": gt_count, "submitted_instances": sub_count,
        "count_error": sub_count - gt_count,
        "_near_identical": identical,
    }


def _seg_equal(a, b):
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# panoptic (real PQ)
# ---------------------------------------------------------------------------
# A "panoptic map" is (seg: HxW int32 of segment ids, cats: {seg_id: class_name}).
# seg id 0 is reserved for void/unlabeled and never scored.
VOID = 0


def instances_to_panoptic(instances, class_names, stuff_names, h, w):
    """Paint a list of (class, mask, seg) into one non-overlapping panoptic map.

    Things -> one segment per instance. Stuff -> one segment per class (merged).
    Smaller masks are painted last so they win ties (thin things on big stuff).
    """
    seg = np.zeros((h, w), np.int32)
    cats = {}
    stuff_seg = {}                        # class -> seg id (merged)
    nxt = 1
    order = sorted(instances, key=lambda t: int(t[1].sum()), reverse=True)  # big first
    for cls, mask, _ in order:
        if cls not in class_names:
            continue
        if cls in stuff_names:
            sid = stuff_seg.get(cls)
            if sid is None:
                sid = stuff_seg[cls] = nxt; cats[sid] = cls; nxt += 1
        else:
            sid = nxt; cats[sid] = cls; nxt += 1
        seg[mask.astype(bool)] = sid      # later (smaller) wins overlaps
    return seg, cats


def rgb2id(rgb):
    rgb = rgb.astype(np.int64)
    return rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]


def load_panoptic_gt(pan_json, png_dir, catid2name, class_names):
    """COCO-panoptic json + PNGs -> {stem: (seg HxW, {seg_id: class_name})}.

    Segments whose class is not in class_names, or that are iscrowd, become void.
    """
    data = json.load(open(pan_json)) if not isinstance(pan_json, dict) else pan_json
    png_dir = Path(png_dir)
    out = {}
    for ann in data["annotations"]:
        stem = Path(ann["file_name"]).stem
        png = png_dir / ann["file_name"]
        if not png.exists():
            continue
        ids = rgb2id(np.array(Image.open(png).convert("RGB"))).astype(np.int32)
        cats = {}
        keep = np.zeros(ids.shape, bool)
        for s in ann["segments_info"]:
            name = catid2name.get(s["category_id"])
            if name in class_names and not s.get("iscrowd", 0):
                cats[int(s["id"])] = name
                keep |= ids == s["id"]
        seg = np.where(keep, ids, VOID).astype(np.int32)
        out[stem] = (seg, cats)
    return out


def _accumulate_pq(pred_seg, pred_cats, gt_seg, gt_cats, class_names, per_class):
    """Match one image's segments and add TP/FP/FN/IoU into per_class."""
    def by_class(seg, cats):
        out = {}
        ids, counts = np.unique(seg, return_counts=True)
        for sid, area in zip(ids, counts):
            if sid == VOID:
                continue
            c = cats.get(int(sid))
            if c in per_class:
                out.setdefault(c, {})[int(sid)] = int(area)
        return out

    gt_by = by_class(gt_seg, gt_cats)
    pr_by = by_class(pred_seg, pred_cats)
    for c in class_names:
        gts = gt_by.get(c, {}); prs = pr_by.get(c, {})
        matched_gt, matched_pr = set(), set()
        gmask = np.isin(gt_seg, list(gts)) if gts else np.zeros_like(gt_seg, bool)
        for pid, parea in prs.items():
            overlap = gt_seg[(pred_seg == pid) & gmask]
            if overlap.size == 0:
                continue
            ids, cnts = np.unique(overlap, return_counts=True)
            for gid, inter in zip(ids, cnts):
                gid = int(gid)
                if gid in matched_gt:
                    continue
                union = parea + gts[gid] - int(inter)
                iou = int(inter) / union if union else 0.0
                if iou > 0.5:
                    per_class[c]["tp"] += 1; per_class[c]["iou"] += iou
                    matched_gt.add(gid); matched_pr.add(pid)
                    break
        per_class[c]["fp"] += len(prs) - len(matched_pr)
        per_class[c]["fn"] += len(gts) - len(matched_gt)


def score_panoptic(pred_maps, gt_maps, class_names):
    """Standard Panoptic Quality (Kirillov et al. 2019), aggregated over images.

    pred_maps / gt_maps: {image_stem: (seg HxW int32, {seg_id: class_name})}.
    PQ = sum(IoU of matched) / (TP + 0.5 FP + 0.5 FN), matched iff IoU > 0.5.
    Returns PQ (value), SQ, RQ, per-class detail. Void (id 0) ignored.
    """
    per_class = {c: {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0} for c in class_names}
    for stem, (gseg, gcats) in gt_maps.items():
        pseg, pcats = pred_maps.get(stem, (np.zeros_like(gseg), {}))
        if pseg.shape != gseg.shape:
            pseg = np.array(Image.fromarray(pseg.astype(np.int32)).resize(
                (gseg.shape[1], gseg.shape[0]), Image.NEAREST)).astype(np.int32)
        _accumulate_pq(pseg, pcats, gseg, gcats, class_names, per_class)

    pqs, sqs, rqs, detail = [], [], [], {}
    for c, d in per_class.items():
        denom = d["tp"] + 0.5 * d["fp"] + 0.5 * d["fn"]
        if denom == 0:
            continue
        pq = d["iou"] / denom
        sq = d["iou"] / d["tp"] if d["tp"] else 0.0
        rq = d["tp"] / denom
        detail[c] = {"pq": round(pq, 3), "sq": round(sq, 3), "rq": round(rq, 3),
                     "tp": d["tp"], "fp": d["fp"], "fn": d["fn"]}
        pqs.append(pq); sqs.append(sq); rqs.append(rq)
    return {"metric": "PQ", "value": float(np.mean(pqs)) if pqs else 0.0,
            "SQ": float(np.mean(sqs)) if sqs else 0.0,
            "RQ": float(np.mean(rqs)) if rqs else 0.0, "per_class": detail}


def build_pred_panoptic(sub_coco, class_names, stuff_names):
    """Student COCO 1.0 export -> {stem: (seg, cats)} panoptic maps."""
    inst = _instances_by_image(sub_coco, class_names)
    imgs = {im["id"]: im for im in sub_coco["images"]}
    dims = {Path(im["file_name"]).stem: (im["height"], im["width"]) for im in imgs.values()}
    out = {}
    for stem, items in inst.items():
        h, w = dims[stem]
        out[stem] = instances_to_panoptic(items, class_names, stuff_names, h, w)
    return out


# ---------------------------------------------------------------------------
# score + anti-cheat
# ---------------------------------------------------------------------------
def metric_to_points(value, weight, cap=HUMAN_CAP, floor=FLOOR):
    frac = (min(value, cap) - floor) / (cap - floor)
    frac = max(0.0, min(1.0, frac))
    return round(frac * weight, 1)


def cheat_flags(task_type, result):
    flags = []
    thr = PQ_SUSPECT if task_type == "panoptic" else SUSPECT
    if result["value"] >= thr:
        flags.append(f"SUSPECT_PERFECT_MATCH: metric {result['value']:.3f} >= {thr} — "
                     "real human labeling is never this perfect. Possible ground-truth reuse.")
    if task_type == "instance":
        ni = result.get("_near_identical", 0); tp = result.get("tp", 0)
        if tp and ni / tp >= 0.9 and tp >= 3:
            flags.append(f"SUSPECT_IDENTICAL_GEOMETRY: {ni}/{tp} masks pixel-identical to GT.")
    if task_type in ("semantic", "panoptic"):
        vals = [v for v in result.get("per_class_iou", {}).values() if v is not None]
        if vals and min(vals) >= SUSPECT:
            flags.append("SUSPECT_ALL_CLASSES_PERFECT: every class IoU >= 0.985.")
    return flags
