# Cài CVAT cho bài Ngày 5

Bài này dùng CVAT chạy cục bộ (Docker). Nếu lớp đã có sẵn CVAT ở `http://localhost:8080`, bỏ qua mục cài.

## Cách nhanh (đã có repo cvat trong khoá học)

```bash
cd ../cvat
./start-cvat.sh up        # khởi động stack
./start-cvat.sh status    # kiểm tra 18 container Up
./start-cvat.sh superuser # tạo tài khoản admin (một lần)
open http://localhost:8080
```

Dùng trình duyệt **Chrome/Edge** (không dùng Safari).

## AI Tools (Segment Anything) — tuỳ chọn

SAM đã được triển khai qua Nuclio trên máy giảng dạy (xem `../day5-segmentation-demo/README.md`, mục
"Serverless SAM"). Nếu AI Tools không có SAM, dùng **OpenCV → Intelligent scissors** (chạy trong trình duyệt,
không cần server).

## Xuất đúng định dạng

- Semantic → **Segmentation mask 1.1** (giữ màu theo lớp).
- Instance / Panoptic → **COCO 1.0** (panoptic: mỗi segment một mask).
- Không sửa tệp bên trong gói xuất.
