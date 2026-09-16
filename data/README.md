# Dữ liệu Ngày 5

Ảnh + ground truth theo từng task. Ground truth trong `groundtruth/` dùng để **tự chấm**; đừng nộp thay bài.

## Nguồn

- **Instance (Medium, một số checkpoint):** COCO val2017 (Creative Commons; ảnh Flickr). Nhãn mask thể hiện (instance)
  lấy từ `instances_val2017.json`. Chỉ giữ 6 lớp phương tiện/người của bài.
- **Semantic (Easy, cp3/cp4/cp6):** BDD100K semantic segmentation (bản tái xuất Kaggle
  `pawakapan/bdd100ksegw`), nhãn theo trainId chuẩn Cityscapes. Ground truth là ảnh trainId, chỉ giữ các lớp của task
  (pixel lớp khác = 255 ignore, không tính điểm).

- **Panoptic (Hard, hard_panoptic):** COCO Panoptic val2017 — ground truth per-instance thật (mỗi vật một segment id + lớp stuff/things), chấm bằng PQ chuẩn COCO. Tên lớp COCO (`pavement-merged`…) đổi sang tên thân thiện (`sidewalk`…) trong `classes.json`.

Ảnh của bài **khác** với bộ minh hoạ của giảng viên (`day5-segmentation-demo`) để tránh trùng.

## Cấu trúc

```
data/
  manifest.json              mô tả mọi task (loại, trọng số, đường dẫn)
  tiers/<cấp>/               images/ + groundtruth/ + classes.json  (3 cấp = 82đ)
  checkpoints/<trạm>/        images/ + groundtruth/ + classes.json  (6 trạm = 18đ)
```

Ba cấp + sáu checkpoint = **100 điểm**. Không còn bộ graded riêng.

## Tạo lại dữ liệu

```bash
~/miniconda3/envs/ai-lab/bin/python scripts/prepare_data.py
```
Cần `KAGGLE_API_TOKEN` trong `../cvat/.env` (tải BDD) và cache COCO của `day5-segmentation-demo`.

## Ground truth giữ kín

`data/tiers/**/groundtruth/` và `data/checkpoints/**/groundtruth/` bị `.gitignore` loại khỏi kho học viên —
đáp án do người hướng dẫn giữ. Học viên gán nhãn và nộp export; người hướng dẫn chạy bộ chấm trên máy có ground truth.
