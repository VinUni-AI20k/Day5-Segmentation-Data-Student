# Tiêu chí & cách tính điểm — Ngày 5

Điểm đo **chất lượng nhãn so với ground truth**, không phải để xếp hạng con người. Mục tiêu là học kỹ năng, không
phải chạy theo con số tuyệt đối.

## Cách quy metric → điểm

Mỗi task có một metric trong `[0, 1]`:

- **Semantic:** `mIoU` (trung bình IoU theo lớp) + `coverage` (% pixel đã phủ).
- **Instance:** `mean matched IoU × recall@0.5` — mask phải **vừa khít vừa đủ** (bỏ sót vật thì recall giảm).
- **Panoptic:** `PQ` (Panoptic Quality chuẩn COCO) = SQ × RQ; khắt khe hơn IoU.

Điểm của task (mốc "đầy đủ" khác nhau theo metric):

```
frac  = clamp( (metric − FLOOR) / (CAP − FLOOR), 0, 1 )
points = frac × trọng_số_task
```

| metric | FLOOR | CAP (đầy đủ điểm) | cờ nghi vấn ≥ |
| --- | ---: | ---: | ---: |
| mIoU / matched-IoU | 0.40 | **0.85** (mức đồng thuận người–người, SAM §5) | 0.985 |
| PQ (panoptic) | 0.20 | **0.65** (PQ khắt khe; người–người ~0.6–0.7) | 0.95 |

- Đạt CAP là điểm đầy đủ; cố kéo cao hơn **không** thêm điểm.
- Dưới FLOOR ≈ 0 điểm.

| metric | mức | ý nghĩa |
| --- | --- | --- |
| < 0.50 | Chưa đạt | mask lệch nhiều, sai lớp, hoặc bỏ sót |
| 0.50–0.70 | Cần bổ sung | đúng ý nhưng biên/òn thiếu |
| 0.70–0.85 | Tốt | gần mức đồng thuận giữa người với người |
| ≥ 0.85 | Đầy đủ | ngang mức con người; điểm tối đa |
| ≥ 0.985 | **Cờ nghi vấn** | xem mục chống gian lận |

## Trọng số

| Nhóm | Task | Trọng số |
| --- | --- | ---: |
| Cấp độ | easy_semantic | 20 |
| | medium_instance | 32 |
| | hard_panoptic | 30 |
| Checkpoint | cp1…cp6 | mỗi trạm 3 (tổng 18) |
| | **Tổng** | **100** |

`scoring/scorecard.py` cộng tất cả (3 cấp + 6 checkpoint) thành **một tổng tối đa 100**. Làm hết = 100%.

## Chống gian lận (in cho người chấm)

Bộ chấm gắn cờ khi kết quả **không giống nhãn người thật**:

- `SUSPECT_PERFECT_MATCH` — metric ≥ 0.985 (panoptic: PQ ≥ 0.95). Nhãn người không đạt mức này.
- `SUSPECT_IDENTICAL_GEOMETRY` (instance) — ≥ 90% mask trùng khít từng pixel với ground truth.
- `SUSPECT_ALL_CLASSES_PERFECT` (semantic) — mọi lớp IoU ≥ 0.985.

Có cờ **không tự động là 0 điểm** — nó báo người chấm kiểm tra: học viên có nộp thẳng ground truth không, có sửa
trực tiếp tệp xuất không. Ground truth (`groundtruth/`) do người hướng dẫn giữ, không phát trong kho học viên
(`.gitignore`), nên học viên không thể chép đáp án.

## Năng lực (định tính, đưa vào REPORT.md)

| Năng lực | Minh chứng rõ | Chưa chứng minh |
| --- | --- | --- |
| Chọn đúng loại phân vùng | phân biệt semantic/instance/panoptic và dùng đúng | tô một mảng chung cho vật đếm được |
| Hình học mask | biên sát, không tràn, không khoét lỗ sai | hộp lỏng, ăn bóng, khoét kính |
| Instance | tách đúng từng vật, đếm đúng | gộp hai xe, hoặc một xe thành hai |
| Panoptic | phủ kín, không chồng lấn, có lối thoát void | bỏ trống mảng lớn, hai mask tranh một pixel |
| Đọc điểm | nối IoU thấp với một quy tắc và sửa | chỉ báo số, không sửa |
| Trung thực | không nộp ground truth, không sửa tệp xuất | có cờ nghi vấn không giải trình |

## Minh chứng tối thiểu

- Ít nhất Easy + Medium đã chấm và có `reports/SCORECARD.md`.
- `REPORT.md` giải thích một lớp/vật điểm thấp và cách sửa.
- Không có cờ nghi vấn chưa giải trình.
