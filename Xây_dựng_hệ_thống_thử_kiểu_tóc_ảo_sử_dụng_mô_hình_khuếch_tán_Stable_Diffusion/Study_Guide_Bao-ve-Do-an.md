# TÀI LIỆU ÔN TẬP VÀ BẢO VỆ ĐỒ ÁN: HỆ THỐNG THỬ KIỂU TÓC ẢO (TRYHAIRSTYLE)

*Tài liệu này tổng hợp toàn bộ các lý thuyết cốt lõi, cơ chế kỹ thuật và các câu hỏi dạng "phản biện" thường gặp nhất để bạn nắm vững kiến trúc hệ thống trước Hội đồng bảo vệ rành rọt nhất.*

---

## PHẦN 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG (THE ARCHITECTURE)
Hãy ghi nhớ 4 trụ cột công nghệ tạo nên ứng dụng và **lý do tại sao lại chọn chúng**:

1. **Frontend (React + Vite):** 
   - *Lý do:* Ứng dụng AI cần cập nhật trạng thái liên tục (Loading, % tiến độ, hiển thị ảnh tức thời, thu nhận tọa độ click chọn mặt). Trạng thái DOM động (Virtual DOM) của React giải quyết cực tốt việc render lại trang mà không bị trễ (lag) hay tải lại trang web (F5) làm mất gián đoạn trải nghiệm người dùng.
2. **Backend (Python FastAPI):** 
   - *Lý do:* FastAPI sinh ra để xử lý các luồng bất đồng bộ (async/await) với tốc độ C++, và quan trọng nhất là nó tương thích tuyệt đối với hệ sinh thái Trí tuệ Nhân tạo lõi của Python (PyTorch, Diffusers, OpenCV). Các framework khác như Node.js hay SpringBoot rất khó gọi trực tiếp thẳng các thư viện AI này.
3. **Queue & Broker (Redis):** 
   - *Lý do:* Khi có 10 người cùng bấm "Tạo Tóc" cùng một lúc, GPU chỉ xử lý được 1 người. Nếu Backend ôm hết, Server sẽ bị gãy (Timeout/Crash). Redis đóng vai trò như "Người Ghi Danh", ai đến trước nhận số trước, lưu vào ổ đệm RAM cực nhanh để phân phát từ từ cho Worker.
4. **Worker (Celery):** 
   - *Lý do:* Celery là một "cỗ máy cày" chạy ẩn phía sau lưng (Background Task). Nó lấy từng nhiệm vụ từ thẻ Redis ra, ép GPU chạy model AI nặng chục GB trong suốt 15-20 giây. Lúc Celery chạy, Backend FastAPI vẫn "rảnh rang" để tiếp khách mới, không bao giờ bị nghẽn cổ chai (Non-blocking).

---

## PHẦN 2: LUỒNG XỬ LÝ TRÍ TUỆ NHÂN TẠO (AI PIPELINE)
Đây là phần cốt lõi của môn học (Machine Learning / Deep Learning). Bạn cần nắm luồng chạy của MỘT bức ảnh qua 3 bước:

### 1. Bước Tiền Xử Lý (Computer Vision)
- **Tại sao phải có bước này?** Đưa cả ảnh thô vào Model sinh luôn sẽ làm rác kết quả. Ta cần thu hẹp phạm vi lấy nét.
- **Tìm mặt (YOLOv8-Face):** Phân tích cực nhanh và khoanh đúng vùng hình chữ nhật chứa khuôn mặt (`bbox`) để cắt nhỏ ảnh ra, giảm thiểu khối lượng pixel dư thừa cho AI.
- **Phân tách vùng (SegFormer):** Vẽ cái Mặt nạ (Mask) phân loại đâu là tóc, đâu là da mặt. Lý do: Khuôn mặt (mắt, mũi, miệng) phải được giữ nguyên, chỉ có vùng không gian "tóc" là để trống cho AI vẽ nét mới vào. Có áp dụng cơ chế *Mặt nạ động (Dynamic Mask Expansion)* tự nới rộng biên độ để hình thành độ cong nếp tóc bồng bềnh tự nhiên.
- **Lấy đặc trưng (InsightFace/AdaFace):** Việc vẽ AI rất dễ làm khuôn mặt bị Tây hóa, biến đổi nhân thế. InsightFace trích xuất bộ nhận dạng 512-chiều để tiêm vào mô hình ép AI phải giữ lại bản dạng gốc (Identity) của người dùng. AdaFace hỗ trợ góc mặt ngang nghiêng $> 45^\circ$.

### 2. Bước Sinh Ảnh (Stable Diffusion XL Inpainting)
- **Tại sao lại dùng lõi SDXL Inpainting?** SDXL là mô hình mạnh nhất về kết cấu giải phẫu học cơ thể hiện tại (vẽ da người chuẩn). Bản Inpainting được tinh chỉnh chỉ tập trung điền hình khối (tóc) vào khu vực bị "che lấp" bởi Mask, thay vì bắt sinh lại nguyên một con người mới (như Text2Image).
- Nạp module **IP-Adapter Plus**: Hút phong cách, kết cấu (texture) của bức ảnh tóc mẫu làm "điều kiện tham chiếu".
- Nạp module **ControlNet Depth + InstantID**: Ép AI bảo toàn 100% hình khối hộp sọ, 5 giác quan ngũ quan không bị méo.
> *Sự kết hợp này gọi là **Cơ chế Điều kiện kép (Dual-Conditioning)***.

### 3. Bước Hậu Xử Lý (Post-Processing)
- **Làm mềm viền:** AI sinh ra bức ảnh hình vuông chứa mặt & tóc mới, nhưng viền xung quanh rất sắc và lộ dải màu. Dùng thuật toán làm mờ `Gaussian Blur` tại đoạn dán ghép để trộn lẫn da mộc tự nhiên với tóc.
- **Đổi màu thần tốc (HSV Colorize):** Lý do: Gọi lại AI tốn 10 giây/lần. Thay vào đó, API Colorize tái sử dụng lớp Mask có sẵn, phủ mảng màu thao tác đại số ma trận OpenCV sang không gian Hue (HSV) chỉ với 0.2 giây. Trực quan và giải phóng tài nguyên hệ thống.

---

## PHẦN 3: "VŨ KHÍ" ĐIỂM CAO - BA ĐÓNG GÓP KỸ THUẬT ĐỂ KHOE
Hội đồng rất nể những bạn xử lý được lỗi của hệ thống lớn. Hãy học thuộc lòng 3 key này:

1. **Thuật toán Crop-and-Paste `bbox` (Giải quyết nạn mất Bối cảnh gốc):** 
   - *Vấn đề:* Bình thường đem thẳng cái ảnh vào AI tô tóc, con AI bị ảo giác vẽ đè/xóa luôn cái cây, bức tường phía sau người đứng.
   - *Cách giải:* Lấy tọa độ thẻ YOLO, cắt đúng cái hình chữ nhật ngay mặt đi sinh tóc (`Crop`). Đợi AI sinh tóc đẹp tuyệt vời ở mặt xong, ta bứng cái mặt đó dán đè ngược (`Paste`) lại vào vị trí trên cái hình to ban đầu. Vậy là khung cảnh sau lưng còn nguyên tỉ lệ 100%.

2. **Tiến trình luân chuyển Đa cấu hình (Celery Subprocess):**
   - *Vấn đề:* Có 2 mô hình ngon là `HairFusion` và `TryOnHairstyle`. Nếu nạp cả 2 vào RAM 1 lúc cho người dùng đổi qua lại thì vỡ VRAM (OOM - Out Of Memory GPU).
   - *Cách giải:* Trình quản trị worker sẽ mở rẽ nhánh một Tiến trình Ảo mới (Subprocess) cách ly RAM ở tiến trình cha. Chạy xong mô hình nào thì giết tiến trình đó thu hồi RAM.

3. **Huấn luyện Khắc khổ trên Colab (Fail-Safe + Caching):**
   - *Vấn đề:* Colab miễn phí Google có 15GB VRAM (T4), load file dataset từ thẻ Google Drive cực chậm 45 phút, hay bị đứt mạng giữa chừng mất tiền trình train model 10 tỷ tham số UNet.
   - *Cách giải:* Dùng **Two-Way I/O Auto-Caching** (Chép đè hàng loạt qua file tạm `/tmp/` máy chủ dùng như ổ SSD ảo) và tạo hàm tự xuất trọng số `.safetensors` lưu làm **Checkpoint** tự động. Rớt lúc nào Resume lại lúc đó.

---

## PHẦN 4: CÂU HỎI THƯỜNG GẶP (Q&A CHEAT SHEET)

**Q1: Thời gian xử lý của hệ thống mất bao lâu? Có đáp ứng chạy Video theo Thời Gian Thực (Real-time) không?**
> *Trả lời:* Hiện tại trên GPU cục bộ (RTX 3060 12GB), một ảnh sinh mất tốn cỡ chục giây tùy luồng (bất đồng bộ). Vì chi phí giải nén UNet khổng lồ, nó **chưa thể** chạy mượt real-time (tức 30 hình/giây). Tương lai em sẽ dùng kỹ thuật ép giảm vòng vặp như LCM (Latent Consistency Model) hoặc TensorRT để tăng tốc độ.

**Q2: Trình bày thuật toán xử lý hàng đợi? Điều gì xảy ra lúc user bấm lệnh Gen tóc?**
> *Trả lời:* Giao diện React gọi POST vào `/generate` kèm API hình ảnh 1 và 2 $\rightarrow$ Backend FastAPI tạo 1 cái vé số thẻ (Task ID) ghi xuống Redis $\rightarrow$ Ném mã Task ID về Web ngay lập tức để người dùng xem cái vòng xoay Loading $\rightarrow$ Lúc đó Celery Worker phía sau GPU mới bắt đầu lôi request đó ra chạy tà tà $\rightarrow$ Web cứ 2 giây tự ping polling API `/status` để xem xong chưa lấy link ảnh về hiển thị. (Tránh treo Timeout Request trình duyệt).

**Q3: Nhược điểm của mạng Inpainting ở đồ án của em là gì?**
> *Trả lời:* Thứ nhất là Vùng Biên Tóc. Khi gặp khuôn mặt chụp nguồn sáng siêu phức tạp (ngược sáng gắt) hoặc tóc mái lòa xòa qua mí mắt chằng chịt, viền Gaussian sinh ra vẫn để lộ vết mờ mờ cắt xén. Thứ hai là sự lấn át đặc trưng ánh sáng của IP-Adapter đôi lúc làm cháy sáng tông ảnh đi một xíu so với da mộc.

**Q4: Tại sao không nạp màu thẳng vào chuỗi Prompt tiếng Anh bằng AI để nó tự xử lý mà em tự đi làm Colorize tay (HSV)?**
> *Trả lời:* Yếu tố kỹ thuật phần mềm. Sinh ảnh tóc bằng model Diffusion tốn 10 giây/hành động. Nếu người dùng chỉ muốn "test xem tóc đổi màu vàng, đỏ có đẹp không", việc liên tục ép Pipeline AI chạy lại mất 10s là điều lãng phí tài nguyên máy chủ. Việc em thiết kế HSV Colorize xử lý không gian toán học ma trận ảnh OpenCV tốn chưa tới 0.2 giây, giảm gánh nặng hệ thống.

---

## PHẦN 5: CHIẾN THUẬT LÚC LÊN TRÌNH BÀY (DEMO)

1. Cười tự tin, chuẩn bị sẵn 5 tấm ảnh của người Việt (các bộ tóc thông thường) và 5 ảnh tóc gốc cực khó trên máy sẵn. Nhớ chọn ảnh có **Hình Nền sau lưng** phong phú (có cái xe máy, con vịt, cây cối...) để lúc DEMO mình **Bảo tồn Bối cảnh Nền (`bbox`)**, mọi người sẽ ồ lên.
2. Mở sẵn Browser và 1 màn hình Console Terminal đen đen chạy backend đằng sau (để các thầy thấy log Celery đang chạy load file ntn cho trực quan).
3. Đánh máy sẵn câu lệnh Prompt Tiếng Việt gõ "Một cô gái xinh đẹp", để khoe tính năng API Phiên Dịch tức thì (tiện ích mở rộng UX/UI).
4. Sử dụng tính năng "Đổi Màu" nhấp chọn qua lại (để minh thị kết quả Instant).

*Chúc bạn hoàn thành phòng thi phản biện thuyết phục và đạt kết quả tối đa! Mọi chi tiết đồ án đã được lập trình vững như bàn thạch!*
