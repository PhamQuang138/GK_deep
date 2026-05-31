import numpy as np


# =====================================================================
# HÀM BỔ TRỢ VECTOR HÓA CẤP CAO: IM2COL VÀ COL2IM (XÓA BỎ VÒNG LẶP PIXEL)
# =====================================================================

def im2col_indices(x, kh, kw, padding=1, stride=1):
    """
    Biến đổi các vùng cửa sổ tích chập trên ảnh thành các cột của ma trận.
    Xử lý song song hàng loạt trên toàn bộ Batch.
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    b, c, h, w = x.shape
    out_h = (h + 2 * p - kh) // stride + 1
    out_w = (w + 2 * p - kw) // stride + 1

    shape_i = (c, kh, kw, out_h, out_w)
    strides_i = (x_padded.strides[1], x_padded.strides[2], x_padded.strides[3],
                 x_padded.strides[2] * stride, x_padded.strides[3] * stride)

    i_all = []
    for img_idx in range(b):
        x_pad_single = x_padded[img_idx]
        sub_matrices = np.lib.stride_tricks.as_strided(x_pad_single, shape=shape_i, strides=strides_i)
        cols = sub_matrices.transpose(3, 4, 0, 1, 2).reshape(out_h * out_w, c * kh * kw).T
        i_all.append(cols)

    return np.array(i_all)  # Shape: (Batch, C*kh*kw, out_h*out_w)


def col2im_indices(cols, x_shape, kh, kw, padding=1, stride=1):
    """
    Biến đổi ngược từ ma trận cột gradient về lại khối tensor 4D ban đầu.
    """
    b, c, h, w = x_shape
    p = padding
    h_padded, w_padded = h + 2 * p, w + 2 * p
    x_padded = np.zeros((b, c, h_padded, w_padded))

    out_h = (h + 2 * p - kh) // stride + 1
    out_w = (w + 2 * p - kw) // stride + 1

    for img_idx in range(b):
        col = cols[img_idx]
        col_reshaped = col.T.reshape(out_h, out_w, c, kh, kw).transpose(2, 3, 4, 0, 1)

        for i in range(kh):
            for j in range(kw):
                x_padded[img_idx, :, i:i + out_h * stride:stride, j:j + out_w * stride:stride] += col_reshaped[
                    :, i, j, :, :]

    if p > 0:
        return x_padded[:, :, p:-p, p:-p]
    return x_padded


# =====================================================================
# THÀNH PHẦN I: CÁC LỚP KÍCH HOẠT (ACTIVATION LAYERS)
# =====================================================================

class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, d_out, lr=None):
        x = self.cache
        return d_out * (x > 0)


class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        x = np.clip(x, -500, 500)
        out = 1.0 / (1.0 + np.exp(-x))
        self.cache = out
        return out

    def backward(self, d_out, lr=None):
        p = self.cache
        return d_out * p * (1.0 - p)


# =====================================================================
# THÀNH PHẦN II: LỚP HÌNH HỌC KHÔNG GIAN (POOLING & UPSAMPLE & RESIDUAL)
# =====================================================================

class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.cache = None

    def forward(self, x):
        """
        Nén kích thước ảnh (Ví dụ: từ 128x128 xuống 64x64)
        """
        self.cache = x
        b, c, h, w = x.shape
        p = self.pool_size
        out = np.zeros((b, c, h // p, w // p))

        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                region = x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p]
                out[:, :, i, j] = np.max(region, axis=(2, 3))
        return out

    def backward(self, d_out):
        x = self.cache
        b, c, h, w = x.shape
        p = self.pool_size
        d_x = np.zeros_like(x)

        for i in range(d_out.shape[2]):
            for j in range(d_out.shape[3]):
                region = x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p]
                max_val = np.max(region, axis=(2, 3), keepdims=True)
                mask = (region == max_val)
                d_x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p] += d_out[:, :, i, j, np.newaxis, np.newaxis] * mask
        return d_x


class Upsample2D:
    def __init__(self, factor=2):
        self.factor = factor
        self.input_shape = None

    def forward(self, x):
        """
        Giải nén tăng kích thước ảnh (Ví dụ: từ 32x32 lên 64x64)
        """
        self.input_shape = x.shape
        return np.repeat(np.repeat(x, self.factor, axis=-2), self.factor, axis=-1)

    def backward(self, d_out):
        b, c, h, w = self.input_shape
        f = self.factor
        d_x = np.zeros(self.input_shape)

        for i in range(h):
            for j in range(w):
                d_x[:, :, i, j] = np.sum(d_out[:, :, i * f:(i + 1) * f, j * f:(j + 1) * f], axis=(2, 3))
        return d_x


class ResidualAdd:
    def __init__(self):
        self.encoder_cache = None

    def save_identity(self, x):
        self.encoder_cache = x

    def forward(self, decoder_x):
        return decoder_x + self.encoder_cache

    def backward(self, d_out):
        return d_out, d_out


# =====================================================================
# THÀNH PHẦN III: LỚP TÍCH CHẬP CONV2D ĐÃ ĐƯỢC TỐI ƯU BẰNG IM2COL
# =====================================================================

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.padding = padding

        # Khởi tạo trọng số He chuẩn cho mạng tích chập sâu
        limit = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.b = np.zeros((out_channels, 1, 1))

        self.x_shape = None
        self.x_cols = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Forward siêu tốc bằng phép Nhân ma trận cục bộ (np.dot) thay cho vòng lặp pixel.
        """
        self.x_shape = x.shape
        b, c, h, w = x.shape

        # Chuyển đổi khối ảnh sang cấu trúc ma trận cột hàng loạt
        self.x_cols = im2col_indices(x, self.k, self.k, padding=self.padding)

        # Duỗi phẳng trọng số của bộ lọc (out_channels, C*k*k)
        W_row = self.W.reshape(self.out_channels, -1)
        out = np.zeros((b, self.out_channels, h, w))

        # Nhân ma trận đồng thời trên chiều Batch
        for img_idx in range(b):
            out_col = np.dot(W_row, self.x_cols[img_idx]) + self.b.reshape(-1, 1)
            out[img_idx] = out_col.reshape(self.out_channels, h, w)

        return out

    def backward(self, d_out, lr):
        """
        Backward vector hóa tuyệt đối, giải quyết lỗi shape bias và cập nhật SGD.
        """
        b, c, h, w = self.x_shape
        W_row = self.W.reshape(self.out_channels, -1)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        d_cols_all = []

        for img_idx in range(b):
            d_out_flat = d_out[img_idx].reshape(self.out_channels, -1)

            # Tính đạo hàm dW và db tích lũy từ các cụm ma trận phẳng
            self.dW += np.dot(d_out_flat, self.x_cols[img_idx].T).reshape(self.W.shape)
            self.db += np.sum(d_out_flat, axis=1, keepdims=True).reshape(self.b.shape)

            # Đẩy ngược gradient lỗi qua ma trận bộ lọc lật
            d_cols = np.dot(W_row.T, d_out_flat)
            d_cols_all.append(d_cols)

        # Lấy trung bình cộng gradient của cả Batch
        self.dW /= b
        self.db /= b

        # Biến đổi ma trận cột lỗi về lại hình khối tensor 4D ban đầu
        d_cols_all = np.array(d_cols_all)
        d_x = col2im_indices(d_cols_all, self.x_shape, self.k, self.k, padding=self.padding)

        # Cập nhật trực tiếp bộ tham số (Thuật toán SGD)
        self.W -= lr * self.dW
        self.b -= lr * self.db

        return d_x


# =====================================================================
# THÀNH PHẦN IV: CẤU TRÚC MẠNG MÔ HÌNH (128X128)
# =====================================================================

class OilSpillSegmentationNet:
    def __init__(self):
        # Nhánh nén đặc trưng (Encoder)
        self.conv1 = Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2)  # 128x128 -> 64x64

        self.conv2 = Conv2D(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2)  # 64x64 -> 32x32

        # Vùng đáy mạng (Bottleneck)
        self.conv_bottom = Conv2D(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.relu_bottom = ReLU()

        # Mạch lưu trữ phần dư (Skip Connections)
        self.res1 = ResidualAdd()
        self.res2 = ResidualAdd()

        # Nhánh giải nén đặc trưng (Decoder)
        self.up1 = Upsample2D(factor=2)  # 32x32 -> 64x64
        self.conv3 = Conv2D(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.relu3 = ReLU()

        self.up2 = Upsample2D(factor=2)  # 64x64 -> 128x128
        self.conv4 = Conv2D(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.relu4 = ReLU()

        # Khối phân loại Pixel đầu ra (128x128)
        self.conv_final = Conv2D(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Block 1 (Encoder) - Lưu phần dư 1 ở kích thước 128x128
        out_c1 = self.relu1.forward(self.conv1.forward(x))
        self.res1.save_identity(out_c1)
        out_p1 = self.pool1.forward(out_c1)

        # Block 2 (Encoder) - Lưu phần dư 2 ở kích thước 64x64
        out_c2 = self.relu2.forward(self.conv2.forward(out_p1))
        self.res2.save_identity(out_c2)
        out_p2 = self.pool2.forward(out_c2)

        # Vùng Đáy Mạng (Bottleneck - 32x32)
        out_bottom = self.relu_bottom.forward(self.conv_bottom.forward(out_p2))

        # Block 3 (Decoder) - Khôi phục kích thước 64x64 và kết hợp phần dư 2
        out_u1 = self.up1.forward(out_bottom)
        out_c3 = self.relu3.forward(self.conv3.forward(out_u1))
        out_res2 = self.res2.forward(out_c3)

        # Block 4 (Decoder) - Khôi phục kích thước 128x128 và kết hợp phần dư 1
        out_u2 = self.up2.forward(out_res2)
        out_c4 = self.relu4.forward(self.conv4.forward(out_u2))
        out_res1 = self.res1.forward(out_c4)

        # Đầu ra dự đoán dạng xác suất cho mỗi pixel (128x128)
        out_final = self.conv_final.forward(out_res1)
        p_mask = self.sigmoid.forward(out_final)

        return p_mask

    def backward(self, loss_grad, lr):
        grad = self.sigmoid.backward(loss_grad)
        grad = self.conv_final.backward(grad, lr)

        grad_decoder, grad_encoder_c1 = self.res1.backward(grad)

        grad = self.relu4.backward(grad_decoder)
        grad = self.conv4.backward(grad, lr)
        grad = self.up2.backward(grad)

        grad_decoder, grad_encoder_c2 = self.res2.backward(grad)

        grad = self.relu3.backward(grad_decoder)
        grad = self.conv3.backward(grad, lr)
        grad = self.up1.backward(grad)

        grad = self.relu_bottom.backward(grad)
        grad = self.conv_bottom.backward(grad, lr)

        grad = self.pool2.backward(grad)
        grad = grad + grad_encoder_c2
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)

        grad = self.pool1.backward(grad)
        grad = grad + grad_encoder_c1
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad, lr)

        return None


# =====================================================================
# THÀNH PHẦN V: CÁC HÀM LOSS & ĐÁNH GIÁ ĐƯỢC TỐI ƯU CHO VẾT TRÀN DẦU SAR
# =====================================================================

def weighted_binary_cross_entropy_loss(y_pred, y_true, weight_class1=15.0):
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    pixel_losses = - (weight_class1 * y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(pixel_losses)


def compute_weighted_bce_gradient(y_pred, y_true, weight_class1=15.0):
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    num_elements = y_pred.size
    grad = (y_pred * (1.0 + y_true * (weight_class1 - 1.0)) - y_true * weight_class1) / num_elements
    return grad


def calculate_soft_iou(y_pred, y_true):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection
    if union == 0:
        return 1.0
    return intersection / union