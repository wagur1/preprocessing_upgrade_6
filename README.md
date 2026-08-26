# VCM Preprocessing Upgrade 5

Bộ công cụ preprocessing và tối ưu hóa cho **Video Coding for Machines (VCM)**. Mục tiêu là giữ chất lượng tác vụ AI sau khi encode/decode, đồng thời giảm bitrate và chi phí tính toán.

## Chạy thử trên Kaggle với Kinetics-400-5%

Tạo một notebook Kaggle bật Internet, sau đó chạy các cell sau:

```bash
!git clone https://github.com/wagur1/preprocessing_upgrade_5.git /kaggle/working/preprocessing_upgrade_5
%cd /kaggle/working/preprocessing_upgrade_5
!pip install -e ".[image]"
!python kaggle/run_kaggle_vcm.py --data /kaggle/input/kinetics400-5per --videos 8
```

Nếu tên dataset của bạn khác, tìm thư mục thực tế bằng:

```bash
!find /kaggle/input -maxdepth 2 -type f -name "*.mp4" | head
```

Script `kaggle/run_kaggle_vcm.py` sẽ:

1. Tìm các video MP4 trong thư mục dataset và lấy tối đa `--videos` video đầu tiên.
2. Encode mỗi video bằng H.264 `libx264` ở QP 22, 27, 32, 37.
3. Tạo nhánh preprocessing mẫu bằng `hqdn3d + eq + unsharp` rồi encode cùng QP.
4. Đo bitrate, PSNR và SSIM xấp xỉ trên các frame mẫu.
5. Tính BD-rate. Giá trị âm nghĩa là tiết kiệm bitrate so với baseline.

Kết quả được ghi vào `/kaggle/working/vcm_results.json`. Đây là **smoke benchmark** để kiểm tra pipeline. PSNR/SSIM chỉ là proxy thị giác; kết quả VCM cần thay hàm `quality()` bằng đầu ra model AI (mAP, mIoU, HOTA...) trên nhãn tương ứng.

## Metric nên tối ưu

| Tác vụ | Metric chính | Metric bổ trợ |
|---|---|---|
| Detection | mAP@0.5, mAP@[.5:.95] | Recall@N, precision |
| Segmentation | mIoU, Dice/F1 | Boundary F-score |
| Tracking | HOTA, IDF1 | Số ID switch, MOTA |
| Re-identification | mAP, Rank-1/5 | CMC curve |
| Pose/keypoint | OKS-mAP, PCK | AP theo từng khớp |
| Perception tổng quát | Cosine embedding/CLIP | Top-k accuracy |
| Chất lượng ảnh | VMAF, MS-SSIM, PSNR-Y | SSIM-Y |

Metric chính nên là metric AI của tác vụ. PSNR hoặc SSIM có thể tăng nhưng mAP giảm nếu preprocessing làm mất cạnh, texture hoặc chi tiết nhỏ quan trọng với model.

## BD-rate và tối ưu đa mục tiêu

Mỗi cấu hình preprocessing phải được đánh giá ở cùng tập QP hoặc bitrate để tạo các điểm `(bitrate_kbps, task_quality)`. `bd_rate(reference, test)` fit đa thức bậc hai trên `ln(bitrate)` theo quality rồi tích phân trên miền giao nhau. Kết quả âm, ví dụ `-18.4%`, nghĩa là cần ít bitrate hơn để đạt cùng chất lượng. `bd_quality` đo mức tăng quality tại cùng bitrate.

Với một operating point, `RandomSearch` tối ưu `w_quality * task_quality - w_rate * log(1 + bitrate_kbps)`. Với toàn bộ đường cong RD, dùng `BDRateSearch` để tối thiểu hóa BD-rate trực tiếp. Khi benchmark thực tế nên theo dõi thêm latency (ms/frame), bộ nhớ, năng lượng và độ ổn định theo video.

## Cài đặt và API

```bash
pip install -e ".[image]"
```

```python
from vcm_preprocess import PreprocessConfig, preprocess_sequence, RandomSearch, BDRateSearch
preprocess_sequence("frames", "frames_pp", PreprocessConfig())

def evaluator(config):
    # preprocess -> encode/decode -> chạy model AI
    return {"task_quality": 0.72, "bitrate_kbps": 500}

result = RandomSearch(evaluator, iterations=100, seed=7).run()
print(result.config, result.metrics)
```

Tối ưu trực tiếp BD-rate:

```python
reference = [(120, .55), (220, .63), (410, .70), (820, .75)]
def rd_evaluator(config):
    return [(110, .55), (205, .63), (390, .70), (790, .75)]
result = BDRateSearch(reference, rd_evaluator, iterations=100).run()
print(result.metrics["bd_rate_percent"])
```

CLI xử lý một thư mục frame: `vcm-preprocess frames/ frames_pp/ --config config.json`.

## Quy trình benchmark đề xuất

1. Chia train/validation/test theo **video**, không trộn frame giữa các tập.
2. Giữ cố định codec, preset, GOP, độ phân giải, chroma format và các QP (thường 22/27/32/37).
3. Đo bitrate trung bình, metric AI trên toàn bộ frame, latency preprocessing + decode + inference.
4. Tính BD-rate trên metric AI; báo cáo thêm BD-rate theo VMAF/PSNR-Y để phát hiện suy giảm thị giác.
5. Chọn các điểm Pareto và chỉ xác nhận một lần trên test set.

## Ghi chú

- Yêu cầu Python 3.10+. Pillow là dependency tùy chọn; metric và optimizer dùng Python standard library.
- `denoise`, `sharpen` nằm trong [0, 1]; `contrast` [0.8, 1.3]; `saturation` [0.7, 1.3]; `luma_gain` [0.85, 1.15].
- Pipeline hiện tại là baseline CPU, không phải learned preprocessor. Có thể thay `preprocess_image` bằng mô hình ONNX/TensorRT mà không đổi evaluator và optimizer.
- BD-rate không có ý nghĩa nếu hai đường RD không có miền quality giao nhau; hàm sẽ báo lỗi để tránh kết quả sai.

## Kiểm thử

```bash
python -m pytest -q
```

## Recipe khuyến nghị để tối ưu transfer

`configs/robust_transfer.yaml` là preset cho đường cong RD ổn định hơn: proxy
block-DCT dùng closed-loop P-frame references, `qp_per_step=3` tối ưu đồng thời
nhiều operating points, và loss giữ task-mask + TV. Sau stage 1, có thể fine-tune
STE với `codec.kind=ste codec.ste_codec=both codec.ste_eval_codec=h265` để lấy mẫu
H.264/H.265 trong forward; khi đánh giá nên chạy riêng `ste_eval_codec=h264` và
`h265` để báo cáo hai anchor cùng protocol.
