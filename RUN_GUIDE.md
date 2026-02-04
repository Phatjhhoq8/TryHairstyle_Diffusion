# Hướng dẫn chạy thử hệ thống TryHairStyle

Để chạy hệ thống này, bạn cần mở **3 Terminal** riêng biệt (trong WSL).

## 1. Khởi động Redis (Database)
Mở Terminal 1 và chạy:
```bash
redis-server
```
*(Nếu đã chạy ngầm rồi `redis-cli ping` trả về PONG thì bỏ qua bước này)*

## 2. Khởi động Backend API
Mở Terminal 2 và chạy:
```bash
cd ~/Desktop/TryHairStyle
source venv_wsl/bin/activate
export PYTHONPATH=.
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API sẽ chạy tại: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`

## 3. Khởi động AI Worker (Quan trọng)
Mở Terminal 3 và chạy (Quá trình này sẽ tải Model, mất khoảng 1-2 phút):
```bash
cd ~/Desktop/TryHairStyle
source venv_wsl/bin/activate
export PYTHONPATH=.
celery -A backend.app.tasks worker --loglevel=info --pool=solo
```
*(Lưu ý: `--pool=solo` giúp debug dễ hơn trên Windows/WSL và tránh lỗi pool process)*

## 4. Cách sử dụng (Test)
1. Truy cập `http://localhost:8000/docs`.
2. Dùng endpoint `/upload` để upload:
   - 1 ảnh khuôn mặt của bạn (User).
   - 1 ảnh mẫu tóc (Reference Hair).
   - **Lưu lại** tên file trả về (ví dụ: `abc.jpg`, `xyz.png`).
3. Dùng endpoint `/transfer` để gửi task:
   - Nhập `user_img`: `abc.jpg`
   - Nhập `hair_img`: `xyz.png`
   - Nhấn Execute.
   - Nhấn Execute.
4. Lấy `task_id` trả về và kiểm tra tại endpoint `/task/{task_id}`.

---

## 5. Chạy Demo Nhanh (CLI)
Nếu không muốn bật server, bạn có thể chạy thử trực tiếp bằng script sau:

```bash
cd ~/Desktop/TryHairStyle
source venv_wsl/bin/activate
export PYTHONPATH=.

# Lệnh chạy demo (dùng ảnh có sẵn trong dataset)
python demo_cli.py --user backend/data/dataset/ffhq/00000/00000.png --hair backend/data/dataset/ffhq/00000/00001.png
```
Kết quả sẽ được lưu thành file `cli_result.png`.
