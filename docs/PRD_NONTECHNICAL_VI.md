# LifeMate — Bản yêu cầu sản phẩm (bản không kỹ thuật)

**Zalo AI Challenge 2026, hạng mục LifeMate · Ngày: 01/11/2026 · Trạng thái: kế hoạch chính thức**
**Mốc cần đạt: nộp Docker cuối cùng trước 23:59 GMT+7 ngày 16/11/2026 (còn 15 ngày).**

Bản này dành cho người đọc không chuyên công nghệ. Toàn bộ chi tiết kỹ thuật nằm ở `docs/PRD.md`.

---

## 1. Bài toán cần giải

Người dùng nói bằng tiếng Việt: *"Tôi sắp về nhà, bật điều hòa phòng khách 26 độ và đặt xe đón tôi."*

Hệ thống phải:

1. **Nghe** → hiểu yêu cầu bằng lời.
2. **Biết** → đây là lúc nào, ở đâu, người dùng có những ứng dụng nào đang mở (đèn, điều hòa, ngân hàng, đặt xe, email…), và mỗi ứng dụng có những "công cụ" nào dùng được.
3. **Làm** → gọi đúng công cụ, đúng tham số, theo đúng thứ tự (tìm ID thiết bị trước, rồi mới tắt đèn).
4. **Dừng đúng chỗ** → nếu thiếu thông tin thì hỏi lại; nếu không ứng dụng nào làm được thì **không làm gì cả**.

Điểm quan trọng: **cuộc thi chấm điểm bằng kết quả, không bằng lời nói.** Hệ thống có thể nói hay nhưng nếu tắt nhầm đèn, bật nhầm điều hòa, hoặc gọi thừa một công cụ, mẫu đó được **0 điểm**.

## 2. Cách chấm điểm (đã kiểm chứng lại với dữ liệu thật)

Mỗi mẫu có hai điều kiện, phải đồng thời đúng:

| Điều kiện | Nghĩa là | Hệ quả |
|---|---|---|
| **Trạng thái cuối đúng** | Sau khi chạy xong, dữ liệu các ứng dụng phải đúng y như người dùng muốn | Sai một chi tiết → **0 điểm cả mẫu**, không có điểm cho đỡ |
| **Danh sách lệnh khớp** | Cùng tên công cụ, tham số không mờ phải giống hệt, tham số văn bản tự do được chấm theo độ giống từ | Chấm theo F1 (cân bằng độ chính xác và độ nhớ) |

Ba điều đã kiểm chứng bằng dữ liệu thật và cần nắm vững. Bảng dưới là **chi phí trung bình** của từng loại lỗi (đo trên 400 mẫu, điểm gốc 100):

| Lỗi | Điểm còn lại | Vì sao |
|---|---|---|
| Bỏ một tham số **bắt buộc** | **6,9** | Lệnh bị từ chối → sai trạng thái, mất trọn 89% mẫu |
| Sửa sai một tham số kiểu số (ví dụ "26" thay vì 26) | **21,3** | Sai giá trị → sai trạng thái |
| Bỏ lệnh cuối cùng | **19,2** | Phần lớn mẫu cần một lệnh ghi cuối cùng |
| Bỏ một tham số **tuỳ chọn** người dùng không nói | **51,1** | Sai tập tham số → không khớp lệnh nào |
| Chọn sai tên công cụ ở lệnh đầu | **59,6** | Nhưng vẫn còn chỗ cứu bằng các lệnh sau |
| Truyền số dạng chuỗi thay vì số | **77,3** | Lỗi kiểu dữ liệu, so sánh rất chặt |
| Thêm một lệnh thừa | **81,8** | Chỉ mất độ chính xác, **không** mất trạng thái |
| Đảo thứ tự các lệnh | **92,8** | Chỉ 29/400 mẫu bị ảnh hưởng trạng thái |
| Sửa sai một tham số **văn bản tự do** | **100** | Văn bản tự do được chấm theo độ giống từ và **không** ảnh hưởng trạng thái |

Ba kết luận rút ra từ bảng này:

- **Ưu tiên tuyệt đối là làm đúng phần "cứng":** tên công cụ, tham số bắt buộc, kiểu dữ liệu, số tiền, ngày giờ. Sai một chỗ ở đây mất trọn mẫu.
- **Phần "mềm" (văn bản tự do) rẻ hơn nhiều:** sai cũng không mất trạng thái, chỉ mất một phần điểm khớp. Vì vậy phải sửa lỗi phần cứng trước, phần mềm sau.
- **Thêm lệnh thừa là lỗi rẻ nhất trong nhóm "làm hỏng trạng thái"** — nhưng nó vẫn tốn ~18 điểm, nên vẫn phải kiểm soát chặt.

**Công cụ đo của chúng tôi đã được kiểm chứng:** chạy lại (replay) toàn bộ 1.545 mẫu có đáp án thông qua chính bộ công cụ của ban tổ chức cho **đúng 100,00/100** (cả 1.545 mẫu đều đạt tuyệt đối), và tái tạo đúng trạng thái cuối của **1.545/1.545** mẫu. Một bài nộp rỗng đạt đúng **7,18/100** — bằng đúng tỉ lệ câu ngoài phạm vi. Nghĩa là mọi con số đo nội bộ sau này đều đáng tin.

## 3. Bức tranh đối thủ (tính đến 01/11/2026)

- **357 đội** đã có điểm trên bảng xếp hạng công khai.
- Đội đứng đầu: **99,37**. Đội trung vị: **48,88**. Có **41 đội** ở mức 90 trở lên.
- Bảng công khai **chỉ mang tính tham khảo**; điểm chính thức quyết định trên bộ kiểm thử riêng, ngày 20–23/11.

Ý nghĩa: mục tiêu hợp lý là **vượt trung vị** trước, rồi **tiệm cận 90+**. Và vì điểm quyết định ở bộ riêng, bài học lớn nhất từ các đội đã làm trước là:

> Một bộ đo nội bộ "nghe có vẻ tốt" nhưng không mô phỏng đúng điều kiện chấm điểm thì **sai lệch có hệ thống**. Một đội đã cải thiện mọi chỉ số nội bộ và vẫn **rơi 0,34 điểm trên bảng chính thức**.

Vì vậy, thứ tự ưu tiên của chúng tôi là: **đo đúng trước, tối ưu sau.**

## 4. Chẩn đoán lỗi: nút thắt thật sự

Chúng tôi phân loại lỗi của hệ thống nền (đo trên bộ dữ liệu thật, giống cách một đội đối thủ đã làm):

| Nhóm lỗi | Tỉ lệ | Ý nghĩa |
|---|---|---|
| **Chọn sai công cụ** | **70,3%** | **Nút thắt lớn nhất.** Nghe đúng, hiểu đúng, nhưng gọi nhầm app |
| Thiếu thông tin, hỏi sai | 17,6% | Biết thiếu gì nhưng hỏi sai cách |
| Nghe sai phần số/ô nhập | 5,5% | Thính là giọng và con số |
| Sai tham số | 4,5% | Đúng tool, sai giá trị |
| Ngoài phạm vi | 2,1% | Không có app nào làm được nhưng vẫn cố |

**Kết luận chiến lược: phần lớn công sức phải dồn vào chọn đúng công cụ**, không phải vào chất lượng âm thanh hay mỹ thuật trình bày câu trả lời.

Một phát hiện nữa, rất quan trọng: khi chúng tôi **ép cho hệ thống đúng hai lệnh đầu tiên** của đáp án (tức đã giải quyết xong bước chọn công cụ), hệ thống vẫn **sai trạng thái cuối ở 29/30 mẫu thử**. Nghĩa là **mô hình 4 tỉ tham số không tự nhiên biết điều phối nhiều bước** — chiến lược "bắt chước từng bước" là chưa đủ. Phải **huấn luyện chính hành vi nhiều bước**, đó là phần cốt lõi của kế hoạch.

## 5. Kiến trúc giải pháp (mô tả không kỹ thuật)

```
Giọng nói (kể cả các lượt trước)
        │
        ▼
  Bộ nhận dạng lời nói  ──►  Lời nói + số đã chuẩn hoá ("hai triệu tám" → 2800000)
        │                                  │
        ▼                                  ▼
  Bối cảnh (giờ, địa điểm, hồ sơ)  +  Danh sách công cụ của app đang mở
        │                                  │
        └──────────────►  Mô hình ngôn ngữ (4,66 tỉ tham số)  ◄── chính sách xử lý bằng tiếng Việt
                                  │
                                  ▼
                    Lớp kiểm tra quy tắc (quyết định tất định, không học)
                    - công cụ có thuộc app đang mở không
                    - tham số có đúng kiểu không (kiểm tra trước khi thực thi, nên lỗi không bị tính điểm)
                    - ID có thật sự từ đâu đó lấy ra không (chống "bịa" ID)
                    - có gọi trùng không
                                  │
                                  ▼
                       Thực thi trên môi trường giả lập
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              Danh sách lệnh              Hash trạng thái cuối
                    └─────────────┬─────────────┘
                                  ▼
                        File nộp bài (JSONL)
```

Ba điểm cần lưu ý về kiến trúc này:

1. **Nhận dạng lời nói chạy trên CPU, mô hình ngôn ngữ chạy trên GPU.** Nên không tranh giành bộ nhớ GPU, và tổng thời gian xử lý một mẫu nằm trong ngân sách 30 giây.
2. **Lớp kiểm tra quy tắc nằm giữa mô hình và môi trường.** Nhiệm vụ của nó là *chặn* lệnh sai trước khi nó làm hỏng dữ liệu — vì lệnh sai làm hỏng dữ liệu thì mất trọn điểm mẫu đó.
3. **Chỉ gọi một lệnh mỗi lượt suy luận.** Dễ kiểm tra, dễ sửa, và mỗi lệnh đều có thể bị truy vết độc lập trong nhật ký.

## 6. Công nghệ chọn dùng (và vì sao)

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Nhận dạng lời nói | **gipformer1.5-68M** (68 triệu tham số, giấy phép MIT) | Nhỏ nhưng chính xác: độ lỗi từ khoảng 16% (chưa huấn luyện thêm) so với 27–36% của các lựa chọn 0,6B. Chạy nhanh gấp ~10 lần thời gian thực trên CPU |
| Mô hình ngôn ngữ | **Qwen3.5-4B** (4,66 tỉ tham số, Apache-2.0) | Mạnh nhất trong mức trần 5 tỉ; **kể cả cả tháp thị giác không dùng tới thì vẫn dưới trần**, nên không cần chờ ban tổ chức giải thích cách đếm tham số |
| Phục vụ mô hình | vLLM | Nhanh, hỗ trợ đầu ra JSON có ràng buộc theo schema, chạy offline |
| Huấn luyện | QLoRA trên GPU Colab | Vừa sức 1 GPU 24 GB, không cần huấn luyện trên máy này |
| Môi trường thực thi | **Chính bộ `zaic_tools` của ban tổ chức** | Bài nộp được chấm bằng đúng thư viện này, nên kết quả chạy cục bộ trùng khớp khi chấm |

**Tổng tham số hệ thống: ~4,73 tỉ** (dưới trần 6 tỉ). **Mô hình ngôn ngữ: 4,66 tỉ** (dưới trần 5 tỉ). **Độ dài ngữ cảnh: 16.384 token** (đúng trần) — và chúng tôi đã đo được: toàn bộ schema công cụ của một mẫu chỉ khoảng 3,1 nghìn token trung bình, tối đa 6 nghìn, nên **không cần cơ chế rút gọn công cụ phức tạp**.

## 7. Kế hoạch huấn luyện

Ba bước, tất cả chạy trên GPU Colab:

1. **Dữ liệu huấn luyện.** Mở rộng từng mẫu có đáp án thành chuỗi bước: đưa vào mô hình "tình trạng hiện tại + những gì đã thấy", và dạy nó bước tiếp theo đúng như đáp án. Khoảng 3.500 mẫu huấn luyện từ 1.545 mẫu gốc. **Dùng lời nhận dạng thật (có lỗi) trong dữ liệu huấn luyện**, để mô hình học cách chịu lỗi nhận dạng thật sự xảy ra lúc thi.
2. **Tinh chỉnh có kiểm soát (QLoRA).** Huấn luyện trên một GPU, kiểm tra trên tập **giữ lại các app chưa từng thấy** — mô phỏng đúng tình huống bài thi riêng, vì **11 app trong bộ kiểm thử công khai không hề xuất hiện trong tập huấn luyện**.
3. **Tinh chỉnh bằng dữ liệu tự sinh đã được chấm điểm.** Cho hệ thống tự thử nhiều cách trên tập huấn luyện, **chấm điểm bằng bộ đo đã kiểm chứng**, giữ lại những cách đạt 100 điểm, rồi huấn luyện tiếp. Đây là bước sửa đúng cái lỗi "sai bước thứ hai trở đi" mà chẩn đoán ở mục 4 đã chỉ ra.

**Chuẩn hoá số và ngày là bước bắt buộc, không phải chi tiết thẩm mỹ.** 51,8% câu nói có số bằng lời ("hai triệu tám", "một nửa", "30 phần trăm") trong khi tham số đáp án là số nguyên. Việc quy đổi này do **mã tất định** đảm nhiệm, không để mô hình tự tính. Tương tự, ngày tháng tương lai ("mai", "thứ Sáu") được tra sẵn bảng ngày để mô hình không bao giờ phải tính lịch.

## 8. Tuân thủ quy định (điểm bắt buộc)

| Quy định | Cách chúng tôi tuân thủ |
|---|---|
| Tổng tham số < 6 tỉ; LLM < 5 tỉ; ngữ cảnh ≤ 16.384 | 4,73 tỉ / 4,66 tỉ / 16.384 |
| Không được đọc trực tiếp `state.json` hay bất kỳ tệp nào trong `<split>_environment/` | Toàn bộ dữ liệu chỉ đi qua công cụ; bộ đo cũng vậy |
| Đúng **một** phiên môi trường cho mỗi mẫu; mọi lệnh gọi đều phải có trong bài nộp | Một `ToolEnv` mỗi mẫu, tạo một lần; bài nộp lấy trực tiếp từ nhật ký phiên |
| 30 giây mẫu; chạy hoàn toàn offline | Đo trung bình 10,91 giây/mẫu ở tham chiếu; có ngân sách 25 giây nội bộ |
| Chỉ dùng mô hình/mở phát hành trước 26/10/2026 | Cả hai mô hình đều phát hành trước hạn (08/2026 và 02/2026) |
| Không gán nhãn bộ kiểm thử công khai/riêng để huấn luyện | Chỉ dùng tập `train` có nhãn |
| Tất định, cùng seed | Giải mã greedy, seed cố định, bộ nhớ đệm nhận dạng theo đường dẫn tệp |
| Bài nộp Docker: mã nguồn + dữ liệu + checksum + link Drive | Đóng gói sẵn từ ngày 14/11, kiểm tra offline bằng `--network none` |

## 9. Lịch trình (15 ngày)

| Ngày | Việc | Điều kiện hoàn thành |
|---|---|---|
| 03/11 | Bộ đo + bộ chấm bài nộp chạy đúng | Đáp án đối chiếu đạt 100,00; bài nộp rỗng đạt đúng tỉ lệ ngoài phạm vi |
| 05/11 | Hệ thống nền (gipformer + Qwen3.5-4B + lớp kiểm tra) trên tập giữ lại app | Chẩn đoán lỗi có số; điểm nội bộ vượt trung vị bảng công khai |
| 08/11 | Xong tinh chỉnh QLoRA; nộp bài lần 1 | Bảng công khai xác nhận bộ đo dự đoán đúng hướng |
| 12/11 | Xong bước huấn luyện bằng dữ liệu tự sinh; đóng các phép thử | Chốt checkpoint tốt nhất; không quá 2 lần nộp |
| 14/11 | Dựng Docker, kiểm tra offline, đo độ trễ | File kết quả trùng bài nộp công khai |
| **16/11** | **Nộp Docker cuối (link Drive + checksum) trước 23:59** | Có xác nhận nộp |
| 20–23/11 | Bộ kiểm thử riêng: chạy ảnh đã đóng băng, nộp kết quả | Không sửa mã sau khi đóng băng |
| 02/12 | Công bố kết quả | Đánh giá mục tiêu top-2 tại đây, không phải trên bảng công khai |

## 10. Rủi ro đã được tính tới

| Rủi ro | Cách xử lý |
|---|---|
| Bài thi riêng có app chưa từng thấy ở bất kỳ bộ nào | Chính sách xử lý không viết theo từng app; dùng toàn bộ schema của mẫu; kiểm chứng trên tập giữ lại app |
| Cách tính hash trạng thái **khác nhau giữa các bộ dữ liệu** (đã kiểm chứng) | Không tự viết lại hàm hash; luôn dùng đúng bộ công cụ của bộ dữ liệu đang chạy |
| Quá 30 giây khi lịch sử hội thoại dài (tới 24 lượt) | Nhận dạng lời nói một lần rồi lưu cache; bản int8 trên CPU; ngân sách 25 giây |
| Colab rớt khi đang huấn luyện | Lưu checkpoint `last` mỗi vòng và `best` khi cải thiện lên Drive; `--resume` nối tiếp |
| Bản tạo dữ liệu huấn luyện mất thời gian do cài thư viện k2/icefall | Huấn luyện bộ nhận dạng lời nói là mục tiêu mở rộng, không phải kế hoạch chính |
| Điểm nội bộ tăng nhưng điểm bảng giảm | Chỉ nộp khi có lý do nội bộ; giới hạn theo hạn mức cho phép |

## 11. Câu hỏi đang chờ ban tổ chức

Một số điểm cần làm rõ bằng văn bản: 30 giây có tính thời gian khởi động container không; thư mục làm việc mà `result/` trỏ tới; cách bộ `zaic_tools` của bộ riêng được đưa vào ảnh; có được phép chạy dịch vụ nội bộ trong container không; có được phép dùng GPU đám mây để xử lý dữ liệu thi không. Các câu hỏi này không chặn việc bắt đầu, nhưng cần hỏi sớm.

## 12. Nguyên tắc chung

- Một bước làm, một lý do: mỗi quyết định trong hệ thống phải truy được về một số liệu đo đã kiểm chứng.
- Ưu tiên *đúng* trước *đẹp*: một lời giải thích dài không cứu được một mẫu sai trạng thái.
- Đo trước, tối ưu sau: nếu bộ đo chưa khớp 100,00 với đáp án thì mọi tối ưu đều vô nghĩa.
- Minh bạch khi chưa chắc: những gì chưa kiểm chứng được sẽ được ghi rõ là chưa kiểm chứng, không trình bày như chắc chắn.
