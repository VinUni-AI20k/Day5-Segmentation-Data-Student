"""Build the Day-5 segmentation lab data: tiered + checkpoint images with golden GT.

Semantic tiers (BDD100K trainId)  -> GT = <id>.png (trainId label map)
Instance tiers (COCO)             -> GT = instances.json (COCO 1.0 subset)

Sources are reused from the sibling day5-segmentation-demo cache (COCO) and Kaggle
(BDD100K, single-file API). Writes data/manifest.json describing every task.

Run:  ~/miniconda3/envs/ai-lab/bin/python scripts/prepare_data.py
"""
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import requests
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
DAY5 = HERE.parent / "day5-segmentation-demo" / "data" / "coco"
CVAT_ENV = HERE.parent / "cvat" / ".env"

BDD_DS = "pawakapan/bdd100ksegw"
BDD_BASE = f"https://www.kaggle.com/api/v1/datasets/download/{BDD_DS}"

# BDD trainId -> (name, Cityscapes RGB) for the classes this lab uses
TRAINID = {
    0: ("road", (128, 64, 128)), 1: ("sidewalk", (244, 35, 232)),
    2: ("building", (70, 70, 70)), 5: ("pole", (153, 153, 153)),
    7: ("traffic sign", (220, 220, 0)), 8: ("vegetation", (107, 142, 35)),
    10: ("sky", (70, 130, 180)), 11: ("person", (220, 20, 60)),
    13: ("car", (0, 0, 142)), 14: ("truck", (0, 0, 70)), 15: ("bus", (0, 60, 100)),
}
NAME2TRAINID = {n: t for t, (n, _) in TRAINID.items()}

# ---- task definitions -------------------------------------------------------
# semantic task -> class list (names); instance task -> COCO cat names
SEMANTIC_CLASSES = {
    "easy_semantic": ["road", "sidewalk", "building", "vegetation", "sky"],
    # hard_panoptic is built via build_panoptic (type=panoptic) using PAN_CLASSES,
    # so it has no semantic entry here.
    "cp4_curb": ["road", "sidewalk"],
    "cp3_thin": ["pole", "traffic sign", "sky", "road"],
    "cp6_coverage": ["road", "sidewalk", "building", "vegetation", "sky", "car", "person"],
}
INSTANCE_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# COCO-Panoptic category name -> friendly class name students type in CVAT
PAN_ALIAS = {
    "road": "road", "pavement-merged": "sidewalk", "sky-other-merged": "sky",
    "tree-merged": "vegetation", "building-other-merged": "building",
    "person": "person", "car": "car", "bus": "bus", "truck": "truck",
    "motorcycle": "motorcycle", "bicycle": "bicycle", "traffic light": "traffic light",
}
PAN_CLASSES = ["road", "sidewalk", "building", "vegetation", "sky",
               "person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light"]
PAN_STUFF = ["road", "sidewalk", "building", "vegetation", "sky"]
PAN_COLORS = {  # for the CVAT labels / student reference
    "road": [128, 64, 128], "sidewalk": [244, 35, 232], "building": [70, 70, 70],
    "vegetation": [107, 142, 35], "sky": [70, 130, 180], "person": [220, 20, 60],
    "car": [0, 0, 142], "bus": [0, 60, 100], "truck": [0, 0, 70],
    "motorcycle": [0, 0, 230], "bicycle": [119, 11, 32], "traffic light": [250, 170, 30],
}

# 3 tiers (82) + 6 checkpoints (18) = 100 total
TASKS = {
    "easy_semantic":  dict(type="semantic",  weight=20, bdd=["817bca71-00000000", "7ee6d192-89e2408b", "81ae7cbb-6bc63a4a"]),
    "medium_instance": dict(type="instance", weight=32, coco=[181542, 373353, 458325]),
    "hard_panoptic":  dict(type="panoptic",  weight=30, pan=[460147, 350023]),
    # checkpoints (learning gates, 3 each)
    "cp1_holes":     dict(type="instance", weight=3, coco=[144300], edge="Holes: windows/gaps stay inside the mask — do NOT cut them out."),
    "cp2_slice":     dict(type="instance", weight=3, coco=[17627],  edge="Slice: adjacent same-class vehicles must be separate instances."),
    "cp5_occlusion": dict(type="instance", weight=3, coco=[336232], edge="Occlusion: an object split by another is still ONE mask; count instances right."),
    "cp3_thin":      dict(type="semantic", weight=3, bdd=["839f7736-abe28069"], edge="Thin structures: poles/signs need a 2-3px brush."),
    "cp4_curb":      dict(type="semantic", weight=3, bdd=["7d83710e-4697c3b2"], edge="Curb: road vs sidewalk is a functional boundary, same asphalt."),
    "cp6_coverage":  dict(type="semantic", weight=3, bdd=["7daa6479-67988f3f"], edge="Coverage: label every pixel — no unlabeled gaps (check coverage %)."),
}


def token():
    for line in CVAT_ENV.read_text().splitlines():
        if line.startswith("KAGGLE_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("KAGGLE_API_TOKEN not in ../cvat/.env")


def bdd_fetch(tok, kind, i):
    sub = "images/val" if kind == "img" else "labels/val"
    fn = f"bdd100k_seg/bdd100k/seg/{sub}/{i}.jpg" if kind == "img" \
        else f"bdd100k_seg/bdd100k/seg/{sub}/{i}_train_id.png"
    for _ in range(5):
        r = requests.get(BDD_BASE, headers={"Authorization": f"Bearer {tok}"},
                         params={"fileName": fn}, timeout=90)
        if r.status_code == 200 and r.content[:3] in (b"\xff\xd8\xff", b"\x89PN"):
            return r.content
        time.sleep(1.5)
    raise RuntimeError(f"fetch failed {fn}")


def build_semantic(tok, task, cfg, out_root, class_names):
    imgd = out_root / "images"; gtd = out_root / "groundtruth"
    imgd.mkdir(parents=True, exist_ok=True); gtd.mkdir(parents=True, exist_ok=True)
    keep_ids = {NAME2TRAINID[n] for n in class_names}
    for i in cfg["bdd"]:
        (imgd / f"{i}.jpg").write_bytes(bdd_fetch(tok, "img", i))
        lbl = np.array(Image.open(io.BytesIO(bdd_fetch(tok, "lbl", i))))
        # keep only this task's classes; everything else -> 255 ignore
        gt = np.full(lbl.shape, 255, np.uint8)
        for t in keep_ids:
            gt[lbl == t] = t
        Image.fromarray(gt).save(gtd / f"{i}.png")
    (out_root / "classes.json").write_text(json.dumps({
        "type": task, "classes": class_names,
        "trainid": {n: NAME2TRAINID[n] for n in class_names},
        "colors": {n: list(TRAINID[NAME2TRAINID[n]][1]) for n in class_names},
    }, indent=2))
    print(f"  {out_root.name}: {len(cfg['bdd'])} images, {len(class_names)} classes")


def build_instance(coco_all, cfg, out_root):
    imgd = out_root / "images"; gtd = out_root / "groundtruth"
    imgd.mkdir(parents=True, exist_ok=True); gtd.mkdir(parents=True, exist_ok=True)
    imgs = {im["id"]: im for im in coco_all["images"]}
    cats = [c for c in coco_all["categories"] if c["name"] in INSTANCE_CLASSES]
    keep_cat = {c["id"] for c in cats}
    ids = set(cfg["coco"])
    sel_imgs = [imgs[i] for i in cfg["coco"]]
    anns = [a for a in coco_all["annotations"]
            if a["image_id"] in ids and a["category_id"] in keep_cat and not a.get("iscrowd", 0)]
    for im in sel_imgs:
        dst = imgd / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(im["coco_url"], dst)
    gt = {"images": sel_imgs, "annotations": anns, "categories": cats}
    (gtd / "instances.json").write_text(json.dumps(gt))
    (out_root / "classes.json").write_text(json.dumps(
        {"type": "instance", "classes": INSTANCE_CLASSES}, indent=2))
    print(f"  {out_root.name}: {len(sel_imgs)} images, {len(anns)} instances")


def build_panoptic(coco_all, pan, cfg, out_root):
    imgd = out_root / "images"; gtd = out_root / "groundtruth"; pngd = gtd / "png"
    imgd.mkdir(parents=True, exist_ok=True); pngd.mkdir(parents=True, exist_ok=True)
    imgs = {im["id"]: im for im in coco_all["images"]}
    pan_by = {a["image_id"]: a for a in pan["annotations"]}
    src_png = DAY5 / "annotations" / "panoptic_val2017"
    catid2name = {c["id"]: PAN_ALIAS[c["name"]] for c in pan["categories"] if c["name"] in PAN_ALIAS}
    sel_imgs, sel_anns = [], []
    for iid in cfg["pan"]:
        im = imgs[iid]; sel_imgs.append(im)
        dst = imgd / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(im["coco_url"], dst)
        ann = pan_by[iid]; sel_anns.append(ann)
        (pngd / ann["file_name"]).write_bytes((src_png / ann["file_name"]).read_bytes())
    (gtd / "panoptic.json").write_text(json.dumps(
        {"images": sel_imgs, "annotations": sel_anns, "categories": pan["categories"]}))
    (out_root / "classes.json").write_text(json.dumps({
        "type": "panoptic", "classes": PAN_CLASSES, "stuff": PAN_STUFF,
        "catid2name": {str(k): v for k, v in catid2name.items()},
        "colors": {n: PAN_COLORS[n] for n in PAN_CLASSES},
    }, indent=2))
    print(f"  {out_root.name}: {len(sel_imgs)} panoptic images, {len(PAN_CLASSES)} classes")


def main():
    tok = token()
    coco_all = json.load(open(DAY5 / "annotations" / "instances_val2017.json"))
    pan = json.load(open(DAY5 / "annotations" / "panoptic_val2017.json"))
    manifest = {"tasks": {}}

    def build(name, cfg, root):
        if cfg["type"] == "instance":
            build_instance(coco_all, cfg, root)
        elif cfg["type"] == "panoptic":
            build_panoptic(coco_all, pan, cfg, root)
        else:
            build_semantic(tok, cfg["type"], cfg, root, SEMANTIC_CLASSES[cfg.get("classes", name)])

    for name, cfg in TASKS.items():
        root = DATA / ("checkpoints" if name.startswith("cp") else "tiers") / name
        build(name, cfg, root)
        manifest["tasks"][name] = {k: v for k, v in cfg.items() if k not in ("coco", "bdd", "pan")}
        manifest["tasks"][name]["path"] = str(root.relative_to(DATA))

    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {DATA/'manifest.json'}")


if __name__ == "__main__":
    main()
