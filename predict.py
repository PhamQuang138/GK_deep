import numpy as np
import cv2
import matplotlib.pyplot as plt
import load_data as ld
import function as func

def predict_new_image(image_path, model_path="oil_spill_model.npy", threshold=0.65):
    # 1. Khởi tạo cấu trúc mạng trống
    model = func.OilSpillSegmentationNet()

    # 2. Nạp trọng số đã huấn luyện từ file vào mạng
    print("[HỆ THỐNG] Đang tải trọng số mô hình...")
    weights = np.load(model_path, allow_pickle=True).item()

    model.conv1.W = weights["conv1_W"]; model.conv1.b = weights["conv1_b"]
    model.conv2.W = weights["conv2_W"]; model.conv2.b = weights["conv2_b"]
    model.conv_bottom.W = weights["conv_bottom_W"]; model.conv_bottom.b = weights["conv_bottom_b"]
    model.conv3.W = weights["conv3_W"]; model.conv3.b = weights["conv3_b"]
    model.conv4.W = weights["conv4_W"]; model.conv4.b = weights["conv4_b"]
    model.conv_final.W = weights["conv_final_W"]; model.conv_final.b = weights["conv_final_b"]

    # 3. Đọc và xử lý ảnh SAR mới
    raw_image = ld.load_image(image_path)  # Shape: (1, 128, 128)
    batch_image = np.expand_dims(raw_image, axis=0)  # Ép thành 4D: (1, 1, 128, 128)

    # 4. Dự đoán (Forward)
    print("[HỆ THỐNG] Đang tiến hành phân đoạn vết dầu...")
    pred_mask = model.forward(batch_image)  # Đầu ra dạng xác suất (0 đến 1)

    # Lấy ảnh 2D từ tensor ra để vẽ
    pred_mask_2d = pred_mask[0, 0]

    # =========================================================
    # SỬA LOGIC: ÉP NGƯỠNG ĐẢO NGƯỢC (VẾT DẦU MÀU ĐEN, NỀN TRẮNG)
    # =========================================================
    # Cách xử lý: Vùng nào XÁC SUẤT THẤP (< threshold) thì cho thành 255 (Trắng)
    # Vùng nào XÁC SUẤT CAO (> threshold - tức là vệt dầu) sẽ thành 0 (Đen)
    binary_mask = (pred_mask_2d < threshold).astype(np.uint8) * 255

    # 5. Trực quan hóa kết quả (Visualize)
    plt.figure(figsize=(12, 4))

    # Tấm hình 1: Ảnh SAR gốc
    plt.subplot(1, 3, 1)
    plt.imshow(raw_image[0], cmap='gray')
    plt.title("Ảnh SAR mới")
    plt.axis("off")

    # Tấm hình 2: Bản đồ nhiệt xác suất vết dầu (Vẫn giữ Jet để nhìn lõi đỏ cho trực quan)
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_2d, cmap='jet', vmin=0, vmax=1)
    plt.title("Bản đồ xác suất (Heatmap)")
    plt.colorbar()
    plt.axis("off")

    # Tấm hình 3: Mặt nạ đầu ra theo đúng ý bạn (Dầu đen - Nền trắng)
    plt.subplot(1, 3, 3)
    # Sử dụng vmin=0 và vmax=255 để đảm bảo matplotlib hiển thị đúng hệ màu tương phản tuyệt đối
    plt.imshow(binary_mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Mặt nạ kết quả (Ngưỡng {threshold})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    NEW_SAR_IMAGE_PATH = "C:/Users/duyqu/Downloads/A_zoom_of_the_17_Nov._ASAR_image_article.jpg"
    predict_new_image(NEW_SAR_IMAGE_PATH)