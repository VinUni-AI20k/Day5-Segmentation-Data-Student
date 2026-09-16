"""Score a student's CVAT export for one Day-5 segmentation task.

Usage:
  python scoring/score.py <task_name> <submission.zip|dir> [--group G] [--json out.json]
  python scoring/score.py --list                      # show every task, grouped by data dir

  task_name : easy_semantic | medium_instance | hard_panoptic | cp1_holes | ...
  --group   : tiers | checkpoints
              Scopes task resolution to that data dir. A task name is unique, so this is
              optional — but it disambiguates intent and guards against scoring a submission
              against the wrong dir.
  submission: CVAT export — "Segmentation mask 1.1" (semantic) or "COCO 1.0" (instance/panoptic).

The two data dirs (data/tiers, data/checkpoints) are discovered from data/manifest.json by
each task's path prefix. Before scoring, the tool checks that the submission's images actually
match the chosen task's ground truth and errors (instead of silently scoring 0) if they
don't — pointing you at the task(s) that do match.

Prints per-class / per-instance metrics, the tier points earned, and any anti-cheat flags.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lab_utils as L  # noqa: E402

DATA = ROOT / "data"
GROUPS = ("tiers", "checkpoints")


def load_registry():
    """Flatten manifest into {task_name: info+_group}, group derived from path prefix."""
    manifest = json.load(open(DATA / "manifest.json"))
    reg = {}
    for section in manifest.values():          # "tasks" (+ any future top-level sections)
        if not isinstance(section, dict):
            continue
        for name, info in section.items():
            group = str(info["path"]).split("/")[0]   # tiers | checkpoints
            reg[name] = {**info, "_group": group}
    return reg


def resolve_task(reg, name, group=None):
    if name not in reg:
        raise SystemExit(f"unknown task '{name}'.\n" + _grouped_list(reg))
    info = reg[name]
    if group and info["_group"] != group:
        same = sorted(n for n, i in reg.items() if i["_group"] == group)
        raise SystemExit(f"task '{name}' is in group '{info['_group']}', not '{group}'.\n"
                         f"tasks in '{group}': {', '.join(same) or '(none)'}")
    base = DATA / info["path"]
    classes = json.load(open(base / "classes.json"))
    return info, base, classes


def _grouped_list(reg):
    lines = ["known tasks (by data dir):"]
    for g in GROUPS:
        names = sorted(n for n, i in reg.items() if i["_group"] == g)
        for n in names:
            i = reg[n]
            w = f"  weight {i['weight']}" if i.get("weight") else ""
            lines.append(f"  [{g:11s}] {n:18s} {i['type']:9s}{w}")
    return "\n".join(lines)


# --- image-name overlap guard: does the submission match this task's GT? ---
def _gt_stems(base, ttype):
    gt = base / "groundtruth"
    if ttype == "semantic":
        return {p.stem for p in gt.glob("*.png")}
    if ttype == "panoptic":
        data = json.load(open(gt / "panoptic.json"))
        return {Path(a["file_name"]).stem for a in data["annotations"]}
    data = json.load(open(gt / "instances.json"))   # instance
    return {Path(im["file_name"]).stem for im in data["images"]}


def _submission_stems(path, ttype):
    members = L._read_members(path)
    if ttype == "semantic":
        return {Path(k).stem for k in members
                if "SegmentationClass/" in k and k.lower().endswith(".png")}
    coco = L._find_coco_json(members)               # instance / panoptic
    return {Path(im["file_name"]).stem for im in coco.get("images", [])}


def _guard_match(reg, task, base, ttype, submission):
    sub_stems = _submission_stems(submission, ttype)
    gt_stems = _gt_stems(base, ttype)
    if not sub_stems or not gt_stems or (sub_stems & gt_stems):
        return                                       # ok (or can't tell — let scoring proceed)
    # zero overlap: find which task(s) this submission actually matches
    matches = []
    for name, info in reg.items():
        try:
            if _submission_stems(submission, info["type"]) & _gt_stems(DATA / info["path"], info["type"]):
                matches.append(f"{name} [{info['_group']}]")
        except Exception:
            continue
    hint = ("\n  did you mean: " + ", ".join(sorted(matches))) if matches else \
           "\n  no task's ground truth contains these images."
    raise SystemExit(
        f"submission images do not match task '{task}' ({ttype}, group '{reg[task]['_group']}').\n"
        f"  submission images: {', '.join(sorted(sub_stems)) or '(none found)'}\n"
        f"  {task} GT images:  {', '.join(sorted(gt_stems))}"
        f"{hint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("submission", nargs="?")
    ap.add_argument("--group", choices=GROUPS, help="scope to a data dir: tiers | checkpoints")
    ap.add_argument("--list", action="store_true", help="list all tasks grouped by data dir and exit")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    reg = load_registry()
    if args.list:
        print(_grouped_list(reg))
        return
    if not args.task or not args.submission:
        ap.error("task and submission are required (or use --list)")

    info, base, classes = resolve_task(reg, args.task, args.group)
    ttype = info["type"]
    weight = info.get("weight", 0)
    gt_dir = base / "groundtruth"

    _guard_match(reg, args.task, base, ttype, args.submission)

    cap, floor = L.HUMAN_CAP, L.FLOOR
    if ttype == "semantic":
        name2id = classes["trainid"]                       # name -> trainId
        id2name = {v: k for k, v in name2id.items()}
        sub = L.load_semantic_submission(args.submission, name2id)
        gt = L.load_semantic_gt(gt_dir, set(name2id.values()))
        result = L.score_semantic(sub, gt, id2name)
    elif ttype == "panoptic":
        cap, floor = L.PQ_CAP, L.PQ_FLOOR
        class_names = set(classes["classes"])
        stuff = set(classes["stuff"])
        catid2name = {int(k): v for k, v in classes["catid2name"].items()}
        members = L._read_members(args.submission)
        sub_coco = L._find_coco_json(members)
        pred = L.build_pred_panoptic(sub_coco, class_names, stuff)
        gt = L.load_panoptic_gt(gt_dir / "panoptic.json", gt_dir / "png", catid2name, class_names)
        result = L.score_panoptic(pred, gt, class_names)
    else:
        members = L._read_members(args.submission)
        sub_coco = L._find_coco_json(members)
        gt_coco = json.load(open(gt_dir / "instances.json"))
        result = L.score_instance(sub_coco, gt_coco, set(classes["classes"]))

    points = L.metric_to_points(result["value"], weight, cap=cap, floor=floor)
    flags = L.cheat_flags(ttype, result)

    print(f"\n=== {args.task}  ({ttype}, group '{info['_group']}') ===")
    print(f"metric ({result['metric']}): {result['value']:.3f}   "
          f"points: {points} / {weight}")
    if ttype == "semantic":
        print(f"coverage (pixels labeled / GT labeled): {result['coverage']*100:.1f}%")
        print("per-class IoU:")
        for k, v in result["per_class_iou"].items():
            print(f"  {k:14s} {'n/a' if v is None else f'{v:.3f}'}")
    elif ttype == "panoptic":
        print(f"SQ (segment quality): {result['SQ']:.3f}   RQ (recognition quality): {result['RQ']:.3f}")
        print("per-class PQ (tp/fp/fn):")
        for k, d in result["per_class"].items():
            print(f"  {k:14s} PQ {d['pq']:.3f}  ({d['tp']}/{d['fp']}/{d['fn']})")
    else:
        print(f"mean matched IoU: {result['mean_matched_iou']:.3f}   "
              f"P@0.5: {result['precision@0.5']:.2f}   R@0.5: {result['recall@0.5']:.2f}")
        print(f"TP {result['tp']}  FP {result['fp']}  FN {result['fn']}   "
              f"count: submitted {result['submitted_instances']} vs GT {result['gt_instances']} "
              f"(error {result['count_error']:+d})")
    if info.get("edge"):
        print(f"edge case: {info['edge']}")
    if flags:
        print("\n*** ANTI-CHEAT FLAGS (evaluator, review) ***")
        for f in flags:
            print("  ! " + f)
    else:
        print("\nno anti-cheat flags.")

    out = {"task": args.task, "type": ttype, "group": info["_group"],
           "points": points, "weight": weight,
           "result": {k: v for k, v in result.items() if not k.startswith("_")},
           "cheat_flags": flags}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")
    return out


if __name__ == "__main__":
    main()
