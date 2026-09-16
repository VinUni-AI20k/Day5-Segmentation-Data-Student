# Hướng dẫn từng bước — Ngày 5 Segmentation

## 0. Chuẩn bị

- CVAT mở được ở `http://localhost:8080` (xem `CVAT_SETUP.md`).
- Cài bộ chấm: `~/miniconda3/envs/ai-lab/bin/python -m pip install -r requirements.txt`.
- Đọc `guideline-mini-sheet.md` — định nghĩa lớp và quy tắc cho từng loại phân vùng.
- Ảnh đã có sẵn trong `data/tiers/<cấp>/images/` và `data/checkpoints/<trạm>/images/`.
  Ground truth (`groundtruth/`) do người hướng dẫn giữ (bị `.gitignore`); học viên nộp export, người hướng dẫn chấm.

> Quan trọng: **tên lớp phải đúng từng chữ** như trong `classes.json` của mỗi task
> (`road`, `sidewalk`, `car`…). Bộ chấm ghép nhãn theo tên; sai tên = lớp đó không được tính.

## 1. Easy — Semantic (20 điểm)

Ảnh: `data/tiers/easy_semantic/images` (3 ảnh BDD). Lớp: `road, sidewalk, building, vegetation, sky`.

1. Tạo project CVAT, thêm đúng 5 lớp trên (kiểu **mask** hoặc **any**).
2. Tạo task, tải 3 ảnh.
3. Tô kín từng vùng bằng **Brush** (hoặc Polygon). Vẽ **từ xa đến gần**; bật **Remove underlying pixels**
   để biên chung chỉ vẽ một lần.
4. Chú ý **bó vỉa (curb)**: `road` và `sidewalk` cùng màu nhựa nhưng khác chức năng — ranh giới là chỗ đường kết thúc.
5. **Save**. Menu → **Export job dataset → Segmentation mask 1.1**.
6. Chấm:
   ```bash
   ~/miniconda3/envs/ai-lab/bin/python scoring/score.py easy_semantic <export.zip>
   ```
   Xem `per-class IoU`. Lớp nào thấp → quay lại sửa, xuất lại, chấm lại.

## 2. Medium — Instance (32 điểm)

Ảnh: `data/tiers/medium_instance/images` (3 ảnh COCO). Lớp: `person, bicycle, car, motorcycle, bus, truck`.

1. Mỗi vật đếm được = **một** mask riêng (đừng gộp).
2. Dùng **Polygon**, **Brush**, hoặc **AI Tools → Segment Anything** (click + / click −) rồi sửa.
3. Ca khó: hai xe sát nhau bị SAM gộp một mask → **Slice (Alt+J)**. Vật bị cột đèn cắt đôi → vẫn **một** mask (Join J).
   Kính/lỗ thủng → **không** khoét.
4. Xuất **COCO 1.0** (kèm ảnh không bắt buộc cho chấm).
5. Chấm: `scoring/score.py medium_instance <export.zip>`. Xem `mean matched IoU`, `recall`, `count_error`.
   - `recall` thấp = bỏ sót vật. `precision` thấp = vẽ thừa. `count_error` = lệch số lượng.

## 3. Hard — Panoptic (30 điểm, chấm bằng PQ thật)

Ảnh: `data/tiers/hard_panoptic/images` (2 ảnh COCO đường phố). Lớp: 12 lớp (xem `classes.json`) —
stuff `road, sidewalk, building, vegetation, sky` + things `person, car, bus, truck, motorcycle, bicycle, traffic light`.

Panoptic = **mọi pixel đúng một nhãn** VÀ **things phải tách từng instance**.

1. Tô cả stuff (một mask cho mỗi vùng) lẫn things (mỗi vật **một** mask riêng). Dùng brush/polygon/SAM.
2. Vẽ từ xa đến gần; z-order `[` `]` + Remove underlying pixels để **không** hai mask chồng một pixel.
3. Pixel không quyết được → để trống (giảm điểm phần đó), đừng đoán bừa.
4. Xuất **COCO 1.0** (mỗi mask là một đối tượng; stuff cũng là mask). Chấm:
   `scoring/score.py hard_panoptic <export.zip>`.
   Đọc **PQ** (điểm chính), **SQ** (mask khít cỡ nào) và **RQ** (nhận đúng bao nhiêu vật). PQ khắt khe hơn IoU:
   phải khớp IoU > 0.5 mới tính là nhận đúng, và bị trừ cho vật thừa/thiếu. PQ ~0.6–0.65 đã là mức tốt.

> Panoptic dùng ground truth **COCO Panoptic** thật (mỗi vật một segment id). PQ tính theo định nghĩa chuẩn (Kirillov 2019: PQ = SQ × RQ, khớp IoU > 0.5); phần xử lý pixel VOID được đơn giản hoá nên có thể lệch nhẹ so với script COCO gốc.

## 4. Checkpoint — trạm ca đặc biệt (mỗi trạm 3 điểm)

Mỗi trạm là **một ảnh**, luyện đúng một lỗi. Đọc `brief` in ra khi chấm.

| Trạm | Loại | Luyện |
| --- | --- | --- |
| `cp1_holes` | instance | lỗ thủng: không khoét kính/khe |
| `cp2_slice` | instance | tách xe sát nhau thành từng instance |
| `cp5_occlusion` | instance | vật bị che vẫn một mask; đếm đúng |
| `cp3_thin` | semantic | nét mảnh: cột/biển, brush 2–3px |
| `cp4_curb` | semantic | ranh giới road/sidewalk |
| `cp6_coverage` | semantic | phủ kín, không sót pixel (xem coverage %) |

Chấm: `scoring/score.py cp1_holes <export.zip>` (tương tự cho các trạm khác).

## Chọn đúng thư mục dữ liệu (tiers / checkpoints)

Hai nhóm dữ liệu nằm ở `data/tiers` và `data/checkpoints`. `score.py` tự tìm
đúng thư mục theo `task_name` (trong `manifest.json`). Thêm hai tiện ích:

- `scoring/score.py --list` — liệt kê mọi task theo từng nhóm (type + weight).
- `scoring/score.py <task> <export.zip> --group tiers|checkpoints` — giới hạn theo
  nhóm; báo lỗi nếu task không thuộc nhóm đó.

Nếu ảnh trong bài nộp **không khớp** ground truth của task đã chọn, `score.py` **dừng và báo
lỗi** (thay vì chấm 0 âm thầm), kèm gợi ý task nào chứa đúng ảnh đó — tránh chọn nhầm thư mục.

## 5. Gộp điểm và báo cáo

1. Đặt mỗi export vào `submissions/<task_name>.zip` (ví dụ `submissions/easy_semantic.zip`).
2. Chạy `~/miniconda3/envs/ai-lab/bin/python scoring/scorecard.py`.
3. Xem `reports/SCORECARD.md` (một tổng /100 gồm 3 cấp + 6 checkpoint, kèm cờ chống gian lận).
4. Viết `REPORT.md` (mẫu ở `reports/REPORT_TEMPLATE.md`): cấp nào bạn làm, điểm, lớp/vật sai và **cách sửa dựa
   trên minh chứng**. Đừng chỉ báo con số — nối nó với một quy tắc.

## 6. Mẹo dùng SAM cho nhanh (không bắt buộc)

AI Tools → Interactors → **Segment Anything**. Click trái = điểm cộng, click phải = điểm trừ, **giữ Ctrl** trước
khi click. SAM tô, **bạn** đặt tên lớp và sửa lỗi máy (tràn biên, gộp hai vật, ăn bóng). Lần click đầu trên một ảnh
mất ~10–30s (chạy CPU), các click sau tức thì.
