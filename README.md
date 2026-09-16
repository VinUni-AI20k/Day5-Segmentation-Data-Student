# Ngày 5 — Segmentation: luyện tập theo cấp độ và chấm điểm tự động

**Đối tượng:** học viên Giai đoạn 1. **Phạm vi:** Ngày 5 — gán nhãn phân vùng (segmentation).
**Thời lượng:** 4 giờ. **Mục tiêu:** thành thạo cả ba loại bài phân vùng — **semantic**, **instance**, **panoptic** —
qua ba cấp độ khó và các trạm luyện (checkpoint) cho từng ca đặc biệt.

Khác với bài chỉ "vẽ cho xong": bạn nộp nhãn, hệ thống **chấm tự động bằng cách so với ground truth** và trả về
điểm theo cấp độ. Bạn tự thấy mình sai ở đâu và luyện lại.

## Ý tưởng cốt lõi

1. **Ba cấp độ + sáu checkpoint, tổng 100 điểm.** Làm hết = 100%.

   | Cấp | Loại phân vùng | Điểm | Học kỹ năng gì |
   | --- | --- | ---: | --- |
   | **Easy** | Semantic (stuff) — mIoU | 20 | tô kín vùng lớn: road / sidewalk / building / vegetation / sky; ranh giới bó vỉa (curb) |
   | **Medium** | Instance (things) — matched IoU | 32 | tách từng vật đếm được; che khuất; lỗ thủng; hai xe một mask |
   | **Hard** | Panoptic (stuff + things) — **PQ** | 30 | mọi pixel đúng một nhãn **và** tách từng instance; điểm PQ thật (COCO Panoptic) |
   | **Checkpoint** | 6 trạm ca đặc biệt | 6 × 3 = 18 | lỗ thủng · tách vật · nét mảnh · bó vỉa · che khuất · phủ kín |

   Ba cấp (82) + sáu checkpoint (18) = **100 điểm**. Hoàn thành tất cả để đạt 100%.

2. **Chấm bằng ground truth có sẵn (thật 100%).** Ảnh + nhãn chuẩn lấy nguyên từ bộ dữ liệu thật:
   **COCO val2017** (instance masks + **COCO Panoptic** per-instance segments) và **BDD100K** (semantic trainId).
   Bạn xuất nhãn từ CVAT rồi chạy `scoring/score.py`; hệ thống báo **mIoU** (semantic), **matched IoU** (instance),
   **PQ / SQ / RQ** (panoptic) cho từng lớp, từng vật.

3. **Cờ chống gian lận.** Nhãn người thật gần như không bao giờ trùng khớp tuyệt đối với ground truth (hai người
   giỏi chỉ đồng thuận ~0,85–0,92). Nếu điểm của bạn **≥ 0,985** hoặc mask trùng khít từng pixel với ground truth,
   hệ thống in **cờ `SUSPECT`** cho người chấm — dấu hiệu có thể bạn đã nộp thẳng ground truth.

4. **Checkpoint cho từng ca đặc biệt.** Mỗi trạm là một ảnh nhỏ, luyện đúng một lỗi khó: lỗ thủng, tách vật (slice),
   nét mảnh, bó vỉa, che khuất, phủ kín. Mỗi trạm 3 điểm — nhanh, và cộng thẳng vào tổng 100.

5. **Ground truth do người hướng dẫn giữ.** Đáp án (`groundtruth/`) **không** nằm trong kho học viên
   (bị `.gitignore`). Học viên gán nhãn và nộp export; người hướng dẫn chạy bộ chấm trên máy có ground truth.

## Bạn sẽ dùng

- **CVAT** cục bộ (xem `CVAT_SETUP.md`). Có thể bật **AI Tools → Segment Anything** để tô nhanh rồi sửa.
- **conda env `ai-lab`** cho bộ chấm điểm: `~/miniconda3/envs/ai-lab/bin/python -m pip install -r requirements.txt`.

## Lịch thực hành 240 phút (gợi ý)

| Phút | Hoạt động |
| ---: | --- |
| 0–15 | Kiểm tra CVAT + `ai-lab`; đọc `guideline-mini-sheet.md` |
| 15–35 | **Easy (semantic):** tạo task 3 ảnh BDD, tô 5 lớp stuff, xuất `Segmentation mask 1.1`, chấm |
| 35–45 | Đọc điểm easy: lớp nào IoU thấp? sửa và chấm lại |
| 45–110 | **Medium (instance):** 3 ảnh COCO, tách từng vật (polygon/brush/SAM), xuất `COCO 1.0`, chấm |
| 110–120 | Nghỉ |
| 120–175 | **Hard (panoptic):** 2 ảnh COCO, phủ kín mọi pixel + đếm vật, xuất, chấm |
| 175–215 | **Checkpoint:** làm 4–6 trạm ca đặc biệt, chấm từng trạm |
| 215–235 | Chạy `scoring/scorecard.py` gộp điểm; viết `REPORT.md` |
| 235–240 | Đưa minh chứng lên kho GitHub cá nhân |

Học viên nhanh: làm hết 3 cấp + 6 checkpoint để đạt 100. Học viên chậm: hoàn thành Easy + Medium là đã đạt cơ bản.

## Bắt đầu nhanh

```bash
# 1. cài bộ chấm
~/miniconda3/envs/ai-lab/bin/python -m pip install -r requirements.txt

# 2. gán nhãn trong CVAT (ảnh ở data/tiers/<cấp>/images), xuất đúng định dạng
#    - semantic              -> "Segmentation mask 1.1"
#    - instance / panoptic   -> "COCO 1.0"

# 3. chấm một cấp
~/miniconda3/envs/ai-lab/bin/python scoring/score.py easy_semantic path/to/export.zip

# 4. gộp điểm tất cả (đặt các export vào submissions/<task>.zip)
~/miniconda3/envs/ai-lab/bin/python scoring/scorecard.py
```

## Tài liệu

- [Hướng dẫn từng bước](GUIDE.md) · [Tiêu chí & cách tính điểm](RUBRIC.md)
- [Phiếu quy tắc gán nhãn](guideline-mini-sheet.md) · [Nguồn dữ liệu](data/README.md)
- [Cài CVAT](CVAT_SETUP.md)

## Bài nộp (kho GitHub cá nhân)

`REPORT.md` + `guideline-mini-sheet.md` + `reports/scorecard.json` + `reports/SCORECARD.md`
+ các export của bạn. **Không** đưa `groundtruth/` (đáp án) vào bài nộp.

> **Nộp export nhãn:** mặc định `.gitignore` bỏ qua `submissions/` và `*.zip` (giữ kho gọn khi luyện).
> Để commit bài, **gỡ hai dòng đó khỏi `.gitignore`** rồi thêm export:
> ```bash
> # trong .gitignore, xoá (hoặc comment) 2 dòng:
> #   submissions/
> #   *.zip
> git add submissions/*.zip
> git commit -m "nộp bài Ngày 5"
> git push
> ```
> Chỉ commit export của **bạn** (`submissions/<task>.zip`) — tuyệt đối **không** commit `groundtruth/`.
