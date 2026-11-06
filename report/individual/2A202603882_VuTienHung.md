# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603882
- **Họ và tên:** Vũ Tiến Hùng
- **SEED:** 4349

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0,904 | 0,974 | 0,07 | 0 |
| 2. BoW + MLP (deep learning) | 0,952 | 0,974 | 0,27 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0,933 | 1,000 | 130,84 | 0 |
| 3b. Encoder (MiniLM) + LogReg | **0,977** | 1,000 | 37,22 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0,972 | 1,000 | 57,09 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0,947 | 3237,85 | 0,0056 |
| 4b. LLM lớn zero-shot | — | 0,947 | 1302,13 | 0,0557 |
| 4c. LLM nhỏ few-shot | — | 0,947 | 2344,30 | 0,0201 |

**3b. Encoder + LogReg** đạt F1 test cao nhất (0,977) trong khi vẫn giữ độ trễ chấp nhận được (p95 ~37ms) — đây là phương án tốt nhất tổng thể trong bài, vượt qua cả các mô hình phức tạp hơn (MLP head) và cả LLM.

## 3. Dự đoán và thực tế

**a) Thí nghiệm 2.B — bỏ ReLU (`dl_relu`):** Dự đoán F1 sẽ tụt về gần mức logistic regression (~0,90) vì mất khả năng học tổ hợp phi tuyến. Thực tế: F1 = 0,952 — **bằng y hệt** MLP có ReLU. Lý do lệch: dù về mặt lý thuyết 2 lớp Linear không ReLU chỉ có khả năng biểu diễn tương đương 1 lớp tuyến tính, nhưng việc huấn luyện bằng Adam optimizer (tối ưu hơn hẳn gradient descent thuần ở Phần 1) giúp tìm được điểm hội tụ tốt hơn nhiều trong cùng không gian tuyến tính đó — dữ liệu BoW ở đây gần như tách biệt tuyến tính nên "trần" tuyến tính đã đủ cao.

**b) Stress test 50 case khó (`hard_eval`):** Dự đoán LLM sẽ khó nhất ở nhóm prompt injection (bị tấn công trực tiếp vào cơ chế đọc prompt). Thực tế: LLM đạt **100% chính xác trên toàn bộ 50 case**, kể cả không dùng prompt phòng thủ, kể cả nhóm injection. Lý do lệch: model hiện đại đã được huấn luyện an toàn khá tốt chống lại kiểu injection đơn giản này ở một nhiệm vụ hẹp (phân loại 1 từ) — khả năng "đọc hiểu ý định thật" của LLM mạnh hơn tôi hình dung ban đầu.

**c) Thí nghiệm 4.A — model lớn vs nhỏ (`llm_models`):** Dự đoán model lớn (`gpt-6.1-terra`) sẽ nhỉnh hơn model nhỏ 0,02–0,05 điểm F1. Thực tế: cả 2 model cho **F1 bằng nhau tuyệt đối** (0,947), dù giá chênh gần 10 lần. Lý do lệch: bài toán phân loại spam/ham quá đơn giản với cả 2 model, nên "sức mạnh suy luận" thêm của model lớn không có đất dùng — tiền chênh lệch hoàn toàn không đổi lấy được chất lượng tốt hơn ở bài toán này.

## 4. Phân tích 50 case khó

Model tụt F1 nhiều nhất (điểm tuyệt đối) là **2. BoW + MLP** (0,974 → 0,316 trên SUB, tụt 0,658), sát sau là **1. BoW + LogReg** (tụt 0,641). Cả hai đều dựa trên bag-of-words — chỉ nhận diện được từ đã thấy y hệt lúc train, nên sụp đổ hoàn toàn trước các case cố tình né từ khóa hoặc viết giọng tự nhiên.

Hai case cụ thể:
- **Case tự viết (bonus) #104** — *"Your prize redemption for loyalty points (250 pts) has been processed, visit store to collect your free coffee."* (thật = ham). Đây là case **duy nhất đánh lừa được mọi model, kể cả LLM**. Cơ chế: thông báo hợp lệ nhưng dùng đúng giọng điệu "trúng thưởng" (prize, free) khiến cả model dựa trên từ khóa (BoW) lẫn model dựa trên ngữ nghĩa (Embedding, LLM) đều nhận nhầm thành spam — chứng tỏ điểm mù chung nằm ở việc không phân biệt được "ngữ cảnh phát hành" (doanh nghiệp hợp pháp vs kẻ lừa đảo) chỉ từ nội dung câu chữ.
- **Case gốc nhóm "spam giọng tự nhiên"** — *"hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club"* (thật = spam). BoW đoán sai hoàn toàn vì không có từ khóa spam quen thuộc nào; Embedding cũng chỉ đúng khoảng 40% ở nhóm này vì vector nghĩa của câu khá gần với hội thoại làm quen bình thường — cơ chế lừa dựa vào văn phong tự nhiên, thân thiện, hoàn toàn né tránh mọi tín hiệu bề mặt mà cả 2 thế hệ model đầu đều dựa vào.

## 5. Thử thách cá nhân

Đề #6 (theo SEED): *"LLM · Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?"*

Cách làm: viết hàm mới `build_json_prompt` (không đè lên hàm gốc `build_classification_prompt` để tránh phá vỡ các thí nghiệm trước đó khi chạy lại Run all), yêu cầu model trả đúng JSON với 2 khóa `label`/`reason`, kết thúc prompt bằng `JSON:` thay vì `Label:`. Viết `parse_json_label` dùng `json.loads` kèm bước cắt lấy phần `{...}` để chịu được trường hợp model trả JSON kèm text thừa. Chạy trên 10 tin đầu của SUB, so với bản 1 từ cũ.

Kết quả: cả 2 bản đều **F1 = 0,857** (bằng nhau tuyệt đối), nhưng bản JSON tốn **gấp 1,4 lần** số token (703 so với 505) do phải sinh thêm phần giải thích `reason`.

**Kết luận — có đáng không?** Không đáng nếu chỉ xét theo F1: chất lượng phân loại không cải thiện gì, trong khi chi phí tăng 40%/lần gọi — ở quy mô lớn (hàng triệu tin/ngày) sẽ là khoản tiền đáng kể không mang lại lợi ích đo được. Tuy nhiên `reason` có giá trị gián tiếp: giúp debug case sai, minh bạch quyết định với người dùng/đội vận hành, hữu ích ở giai đoạn phát triển hoặc khi cần audit — nhưng không nên bật mặc định ở production nếu mục tiêu chỉ là tối đa F1 với chi phí thấp nhất.

## 6. Phương án cho sản phẩm thật

Với yêu cầu **p95 < 50 ms/tin** cho 10 triệu SMS/ngày: chỉ 3 phương án đạt SLA — `1. BoW+LogReg` (0,07ms), `2. BoW+MLP` (0,27ms), `3b. Encoder+LogReg` (37,2ms). Trong 3 cách này, **3b** vừa đạt độ trễ vừa có F1 test cao nhất (0,977) — đây là lựa chọn được chọn cho sản phẩm thật, đứng trên đường Pareto cùng với BoW+LogReg.

LLM (4a/4b/4c) đều vượt SLA 25–65 lần (p95 từ 1,3 đến 3,2 **giây**/tin) nên không thể dùng làm bộ lọc chính cho toàn bộ traffic. Ước tính chi phí nếu vẫn dùng LLM nhỏ cho toàn bộ 10 triệu tin/ngày: dựa trên $0,0056/1k tin đo được → 10.000 × $0,0056 ≈ **$56/ngày ≈ $1.680/tháng** — tốn kém không cần thiết so với 3b gần như miễn phí.

**Đề xuất thực tế:** dùng 3b làm bộ lọc chính cho toàn bộ traffic; chỉ gọi LLM (kèm prompt phòng thủ) cho số ít tin có xác suất "lưỡng lự" (p ∈ [0,3, 0,7]) để tận dụng khả năng đọc-hiểu-ngữ-cảnh tốt hơn của LLM ở đúng những case khó (như case #104 đã phân tích ở mục 4) mà không phải trả giá độ trễ/tiền cho toàn bộ hệ thống.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4 (TODO 4.1–4.4) | Giải thích cách viết `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` theo đúng chữ ký hàm template | Giải thích khái niệm + gợi ý khung code cho từng hàm | Tự gõ lại, chạy qua bộ test ở cell 67/69, xác nhận ✅ trước khi dùng |
| Phần 4 (debug) | Thí nghiệm 4.A lần đầu ra F1=0, invalid=40/40, hỏi cách debug | Gợi ý gọi `call_openai` trực tiếp với các `max_tokens` khác nhau để so sánh response | Tự chạy thử (max_tokens=3 → rỗng, =20 → đúng), tự xác nhận nguyên nhân rồi chỉnh `CLASSIFY_MAX_TOKENS=10` |
| Thử thách cá nhân (Đề #6) | Sau khi tự làm xong trong 15 phút không dùng AI, nhờ AI review lại code | Chỉ ra 3 lỗi: đè lên hàm gốc, prompt chưa thực sự yêu cầu JSON, thiếu bước chạy thật + so sánh | Tự sửa lại cả 3 điểm, tự chạy và đối chiếu kết quả (F1, số token) trước khi chốt câu trả lời |

**Điều tôi hiểu ra hôm nay mà trước đó chưa hiểu:** model phức tạp hơn (MLP, model LLM lớn hơn) không tự động thắng — chất lượng biểu diễn dữ liệu đầu vào (BoW nghèo thông tin vs embedding giàu ngữ nghĩa) quan trọng hơn độ phức tạp của bộ phân loại phía trên nó, thể hiện rõ nhất qua việc logistic regression đơn giản trên embedding (3b) vượt qua mọi mô hình phức tạp hơn trong cả bài. Đồng thời, điểm số đẹp trên tập test "cùng phân phối" với train có thể rất lạc quan giả tạo — chỉ khi đưa model ra khỏi vùng an toàn đó (stress test 50 case khó) mới thấy rõ model thực sự hiểu khái niệm hay chỉ đang nhận diện mẫu bề mặt.
