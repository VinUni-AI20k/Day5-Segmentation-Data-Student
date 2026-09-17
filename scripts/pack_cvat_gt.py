"""Pack task ground truth into a CVAT-importable archive.

Semantic tasks  -> Segmentation mask 1.1 (.zip with SegmentationClass/, ImageSets/, labelmap.txt)
Instance tasks  -> COCO 1.0 (.zip with annotations/instances_default.json)
Panoptic tasks  -> COCO 1.0 (.zip; panoptic segments converted to per-segment RLE masks)

Usage:
  python scripts/pack_cvat_gt.py data/tiers/easy_semantic
  python scripts/pack_cvat_gt.py --task easy_semantic
  python scripts/pack_cvat_gt.py --all
  python scripts/pack_cvat_gt.py data/tiers/hard_panoptic -o /tmp/hard_panoptic_gt.zip

Import in CVAT:
  semantic  -> Actions -> Upload annotations -> Segmentation mask 1.1
  instance / panoptic -> Actions -> Upload annotations -> COCO 1.0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load_manifest() -> dict:
    return json.load(open(DATA / "manifest.json"))["tasks"]


def _resolve_task_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    tasks = _load_manifest()
    if arg not in tasks:
        raise SystemExit(f"unknown task '{arg}' and path does not exist: {arg}")
    return (DATA / tasks[arg]["path"]).resolve()


def _task_type(task_dir: Path) -> str:
    meta = json.load(open(task_dir / "classes.json"))
    return meta.get("type", "semantic")


def _zip_dir(src: Path, zip_path: Path) -> None:
    """Zip contents of src/ so archive root has its immediate children (no wrapper folder)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src).as_posix())


def _rgb2id(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]


def pack_semantic(task_dir: Path, out_dir: Path) -> Path:
    """Build Segmentation mask 1.1 package from trainId PNGs in groundtruth/."""
    meta = json.load(open(task_dir / "classes.json"))
    class_names = meta["classes"]
    trainid = meta["trainid"]
    colors = meta["colors"]
    id2rgb = {int(trainid[name]): tuple(colors[name]) for name in class_names}

    gt_dir = task_dir / "groundtruth"
    masks = sorted(gt_dir.glob("*.png"))
    if not masks:
        raise SystemExit(f"no PNG masks in {gt_dir}")

    seg_dir = out_dir / "SegmentationClass"
    iset_dir = out_dir / "ImageSets" / "Segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    iset_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for mask_path in masks:
        stem = mask_path.stem
        stems.append(stem)
        lab = np.array(Image.open(mask_path))
        if lab.ndim == 3:
            lab = lab[..., 0]
        color = np.zeros((*lab.shape, 3), np.uint8)  # ignore / background -> (0,0,0)
        for tid, rgb in id2rgb.items():
            color[lab == tid] = rgb
        Image.fromarray(color).save(seg_dir / f"{stem}.png")

    lines = ["background:0,0,0::"]
    for name in class_names:
        r, g, b = colors[name]
        lines.append(f"{name}:{r},{g},{b}::")
    (out_dir / "labelmap.txt").write_text("\n".join(lines) + "\n")
    (iset_dir / "default.txt").write_text("\n".join(stems) + "\n")
    return out_dir


def _coco_zip(task_dir: Path, out_dir: Path, coco: dict) -> Path:
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "instances_default.json").write_text(json.dumps(coco))
    return out_dir


def pack_instance(task_dir: Path, out_dir: Path) -> Path:
    gt_json = task_dir / "groundtruth" / "instances.json"
    if not gt_json.is_file():
        raise SystemExit(f"missing {gt_json}")
    coco = json.load(open(gt_json))
    return _coco_zip(task_dir, out_dir, coco)


def pack_panoptic(task_dir: Path, out_dir: Path) -> Path:
    meta = json.load(open(task_dir / "classes.json"))
    class_names = meta["classes"]
    catid2name = {int(k): v for k, v in meta["catid2name"].items()}

    pan_json = task_dir / "groundtruth" / "panoptic.json"
    png_dir = task_dir / "groundtruth" / "png"
    if not pan_json.is_file():
        raise SystemExit(f"missing {pan_json}")
    if not png_dir.is_dir():
        raise SystemExit(f"missing {png_dir}")

    pan = json.load(open(pan_json))
    name2catid = {n: i + 1 for i, n in enumerate(class_names)}

    images, annotations = [], []
    ann_id = 1
    for ann in pan["annotations"]:
        stem = Path(ann["file_name"]).stem
        png = png_dir / ann["file_name"]
        if not png.is_file():
            raise SystemExit(f"missing panoptic mask {png}")
        ids = _rgb2id(np.array(Image.open(png).convert("RGB")))
        im = next(x for x in pan["images"] if x["id"] == ann["image_id"])
        images.append({
            "id": im["id"],
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        })
        for seg in ann["segments_info"]:
            name = catid2name.get(seg["category_id"])
            if name is None or name not in class_names or seg.get("iscrowd", 0):
                continue
            m = (ids == seg["id"]).astype(np.uint8)
            if m.sum() == 0:
                continue
            rle = cocomask.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode()
            annotations.append({
                "id": ann_id,
                "image_id": im["id"],
                "category_id": name2catid[name],
                "segmentation": rle,
                "area": int(m.sum()),
                "bbox": list(cocomask.toBbox(rle)),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": f"Day5 panoptic GT for {task_dir.name}"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": name2catid[n], "name": n, "supercategory": ""} for n in class_names],
    }
    return _coco_zip(task_dir, out_dir, coco)


def pack_task(task_dir: Path, output: Path | None = None) -> Path:
    task_dir = task_dir.resolve()
    if not (task_dir / "classes.json").is_file():
        raise SystemExit(f"not a task dir (no classes.json): {task_dir}")
    if not (task_dir / "groundtruth").is_dir():
        raise SystemExit(f"missing groundtruth/: {task_dir}")

    ttype = _task_type(task_dir)
    if output is None:
        suffix = "segmask" if ttype == "semantic" else "coco"
        output = task_dir / f"cvat_gt_{suffix}.zip"
    else:
        output = Path(output).resolve()

    with tempfile.TemporaryDirectory(prefix="cvat_gt_") as tmp:
        staging = Path(tmp) / "package"
        staging.mkdir()
        if ttype == "semantic":
            pack_semantic(task_dir, staging)
            fmt = "Segmentation mask 1.1"
        elif ttype == "instance":
            pack_instance(task_dir, staging)
            fmt = "COCO 1.0"
        elif ttype == "panoptic":
            pack_panoptic(task_dir, staging)
            fmt = "COCO 1.0 (panoptic segments as masks)"
        else:
            raise SystemExit(f"unsupported task type: {ttype}")

        _zip_dir(staging, output)

    print(f"wrote {output}")
    print(f"  task: {task_dir.name} ({ttype})")
    print(f"  CVAT import format: {fmt}")
    if ttype == "semantic":
        print("  note: task must include a 'background' label (#000000) — "
              "run convert_label_cvat.py on classes.json before creating the task")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", nargs="?", help="task folder or manifest task name")
    ap.add_argument("-o", "--output", help="output .zip path")
    ap.add_argument("--all", action="store_true", help="pack every task in manifest.json")
    ap.add_argument("--list", action="store_true", help="list tasks and exit")
    args = ap.parse_args()

    tasks = _load_manifest()
    if args.list:
        for name, info in sorted(tasks.items()):
            print(f"{name:18s}  {info['type']:9s}  data/{info['path']}")
        return

    if args.all:
        for name, info in sorted(tasks.items()):
            pack_task(DATA / info["path"])
        return

    if not args.task_dir:
        ap.error("task_dir is required (or use --all / --list)")
    pack_task(_resolve_task_dir(args.task_dir), args.output)


if __name__ == "__main__":
    main()
