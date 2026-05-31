# Phân đoạn Vệt Dầu Trên Ảnh SAR Sử Dụng Deep Learning

## Giới thiệu

Ô nhiễm môi trường biển do các sự cố tràn dầu là một trong những vấn đề nghiêm trọng ảnh hưởng đến hệ sinh thái biển và các hoạt động kinh tế ven biển. Việc phát hiện và xác định chính xác phạm vi dầu loang từ ảnh SAR (Synthetic Aperture Radar) có ý nghĩa quan trọng trong công tác giám sát và ứng phó sự cố môi trường.

Dự án này tập trung xây dựng một mô hình phân đoạn ảnh theo hướng **"from scratch"**, tức là tự cài đặt các thành phần của mạng nơ-ron bằng NumPy thay vì sử dụng các framework Deep Learning như TensorFlow hoặc PyTorch. Mục tiêu chính là tìm hiểu nguyên lý hoạt động của Deep Learning trong bài toán Image Segmentation.

---

## Mục tiêu

* Tìm hiểu bài toán phân đoạn ảnh (Image Segmentation).
* Xây dựng mô hình Encoder–Decoder lấy cảm hứng từ U-Net.
* Tự cài đặt các lớp cơ bản của mạng CNN bằng NumPy.
* Hiểu cơ chế Forward Propagation và Backpropagation.
* Thực hiện phân đoạn vùng dầu loang trên ảnh SAR.
* Đánh giá kết quả bằng các chỉ số phổ biến trong Segmentation.

---

## Chức năng chính

* Đọc và tiền xử lý dữ liệu SAR.
* Xây dựng lớp Conv2D bằng kỹ thuật im2col.
* Cài đặt ReLU và Sigmoid.
* Cài đặt MaxPooling và Upsampling.
* Sử dụng Skip Connection dạng Residual Add.
* Huấn luyện mô hình bằng SGD.
* Lưu và nạp trọng số mô hình.
* Trực quan hóa kết quả phân đoạn.
* Đánh giá bằng Precision, Recall, Dice và IoU.

---

## Bộ dữ liệu

Dự án sử dụng bộ dữ liệu công khai trên Kaggle gồm:

* Ảnh SAR (Synthetic Aperture Radar).
* Ground Truth Mask tương ứng.

Mỗi mẫu dữ liệu bao gồm:

* Ảnh đầu vào dạng grayscale.
* Mặt nạ nhị phân biểu diễn vùng dầu loang.

Thư mục dữ liệu được loại khỏi Git thông qua file `.gitignore`.

---
## Cấu trúc dự án

```text
.
├── .gitignore
├── README.md
├── dataset.py         # Xử lý và chuẩn bị dữ liệu
├── load_data.py       # Đọc dữ liệu SAR và mask
├── function.py        # Các hàm hỗ trợ, layer và phép toán
├── train.py           # Huấn luyện mô hình
├── evaluate.py        # Đánh giá mô hình
└── predict.py         # Dự đoán và trực quan hóa kết quả
```

## Kiến trúc mô hình

Đầu vào:

```text
1 × 128 × 128
```

Kiến trúc:

```text
Encoder
 ├── Conv2D + ReLU
 ├── MaxPool
 ├── Conv2D + ReLU
 └── MaxPool

Bottleneck

Decoder
 ├── Upsample
 ├── Residual Add
 ├── Conv2D + ReLU
 ├── Upsample
 ├── Residual Add
 └── Conv2D Final + Sigmoid
```

Đầu ra:

```text
1 × 128 × 128
```

Mỗi pixel đầu ra biểu diễn xác suất thuộc vùng dầu loang.

---


Quy trình huấn luyện gồm:

1. Đọc dữ liệu.
2. Tiền xử lý ảnh.
3. Forward Propagation.
4. Tính Loss.
5. Backpropagation.
6. Cập nhật trọng số bằng SGD.
7. Đánh giá trên tập Validation.

---

## Chỉ số đánh giá

Mô hình được đánh giá bằng:

* Precision
* Recall
* Dice / F1-score
* Intersection over Union (IoU)

Kết quả tốt nhất:

| Chỉ số        | Giá trị |
| ------------- | ------- |
| Precision     | 47.77%  |
| Recall        | 87.83%  |
| Dice/F1-score | 61.88%  |
| IoU           | 44.80%  |

(Threshold = 0.8)

---

## Kết quả đạt được

* Xây dựng thành công mô hình Segmentation bằng NumPy.
* Tự cài đặt các lớp CNN cơ bản.
* Hoàn thiện cơ chế Forward và Backpropagation.
* Thực hiện phân đoạn vùng dầu loang trên ảnh SAR.
* Trực quan hóa kết quả dự đoán và đánh giá mô hình.

Mặc dù chưa đạt hiệu năng như các mô hình hiện đại, dự án đã hoàn thành mục tiêu quan trọng nhất là giúp hiểu rõ nguyên lý hoạt động của Deep Learning trong bài toán phân đoạn ảnh.

---

## Hướng phát triển

* Tăng độ sâu mạng nơ-ron.
* Bổ sung Batch Normalization.
* Áp dụng Attention Mechanism.
* Sử dụng Transposed Convolution thay cho Upsampling.
* Data Augmentation.
* Tận dụng GPU để tăng tốc huấn luyện.
* Chuyển sang TensorFlow hoặc PyTorch để tối ưu hiệu năng.


