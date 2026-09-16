# Phiếu quy tắc gán nhãn — Ngày 5 Segmentation

**Họ và tên:** CHƯA ĐIỀN &nbsp;·&nbsp; **MSSV:** CHƯA ĐIỀN

## 1. Ba loại bài — chọn đúng loại trước khi vẽ

| Loại | Câu hỏi | Đếm được không | Ví dụ lớp |
| --- | --- | --- | --- |
| **Semantic** | "pixel này là loại gì?" | không | road, sidewalk, sky (stuff) |
| **Instance** | "pixel này thuộc *vật nào*?" | có | car #1, car #2 (things) |
| **Panoptic** | cả hai: mọi pixel một nhãn, vật thì thêm ID | cả hai | road + car#1 + car#2 |

Chọn nhầm loại = làm lại. Easy = semantic, Medium = instance, Hard = panoptic.

## 2. Lớp cố định (đúng tên từng chữ — bộ chấm ghép theo tên)

**Stuff (semantic/panoptic):** `road`, `sidewalk`, `building`, `vegetation`, `sky`. (checkpoint semantic thêm: `pole`, `traffic sign`)
**Things (instance/panoptic):** `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`.

| Lớp | Gán khi | Không gán |
| --- | --- | --- |
| `road` | mặt đường xe chạy | vỉa hè, bó vỉa |
| `sidewalk` | vỉa hè, phần người đi | lòng đường |
| `car` | ô tô con, SUV, taxi | xe tải có thùng, bus |
| `truck` | có thùng/ben/sàn hàng rõ | ô tô con, van |
| `bus` | thân khách dài, nhiều cửa sổ | van nhỏ |

## 3. Quy tắc hình học (mọi loại)

- Biên **sát phần nhìn thấy**; không đoán phần bị che.
- **Lỗ thủng** (kính xe, khe): tính là một phần của vật — **không khoét** (trừ khi guideline bảo khoét).
- Vật bị vật khác **cắt làm đôi**: vẫn là **một** mask (brush làm được; polygon thì Join `J`).
- Hai vật sát nhau bị gộp một mask: **Slice `Alt+J`** để tách.
- **Vẽ từ xa đến gần**; biên chung chỉ vẽ một lần (Remove underlying pixels / z-order `[` `]`).

## 4. Semantic & Panoptic

- Tô kín, không để hai mask tranh cùng một pixel.
- **Bó vỉa (curb):** `road` và `sidewalk` cùng nhựa, khác chức năng — ranh giới là chỗ đường kết thúc.
- **Nét mảnh** (cột, biển báo): brush 2–3px.
- Panoptic: **mọi pixel đúng một nhãn**; pixel không quyết được → ghi vào nhật ký, để trống (giảm coverage) thay vì đoán bừa.

## 5. Ba tình huống mơ hồ (điền trước khi xem điểm)

### A — `car` hay `truck`/`van`? (pickup có thùng, minibus…)
- Ảnh và vị trí vật: CHƯA ĐIỀN
- Dấu hiệu nhìn thấy: CHƯA ĐIỀN
- Quy tắc áp dụng + quyết định: CHƯA ĐIỀN

### B — Instance: hai vật hay một? (xe sát nhau / xe bị che cắt đôi)
- Ảnh và vị trí: CHƯA ĐIỀN
- Slice hay Join? Vì sao: CHƯA ĐIỀN

### C — Semantic: `road` hay `sidewalk` ở chỗ bó vỉa?
- Ảnh và vị trí: CHƯA ĐIỀN
- Bằng chứng ở mức phóng 100%: CHƯA ĐIỀN
- Quyết định: CHƯA ĐIỀN

## 6. Tự kiểm tra

- [ ] Đã chọn đúng loại phân vùng cho từng cấp.
- [ ] Tên lớp đúng từng chữ như `classes.json`.
- [ ] Không khoét lỗ thủng; không gộp/không tách nhầm instance.
- [ ] Panoptic: không có pixel bị hai mask; đã kiểm coverage.
- [ ] Đã tự chấm bằng `scoring/score.py` và sửa lớp/vật điểm thấp.
- [ ] Không nộp ground truth, không sửa trực tiếp tệp xuất.
