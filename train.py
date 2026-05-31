import numpy as np
from tqdm import tqdm  # Import thư viện thanh trạng thái
import load_data as ld
import function as func

# Cấu hình siêu tham số
EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 0.001

if __name__ == "__main__":
    print("=== KẾT NỐI DATA VÀ KHỞI TẠO MẠNG TRÊN CPU ===")

    # Khởi tạo mô hình từ file function.py
    model = func.OilSpillSegmentationNet()

    # Tính toán tổng số batch để tqdm hiển thị % chính xác
    total_train_batches = int(np.ceil(len(ld.train_files) / BATCH_SIZE))
    total_val_batches = int(np.ceil(len(ld.val_files) / BATCH_SIZE))
    history_train_loss = []
    history_train_iou = []
    history_val_iou = []
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        # =========================================================
        # 1. GIAI ĐOẠN HUẤN LUYỆN (TRAINING)
        # =========================================================
        train_gen = ld.data_generator(ld.train_image_dir, ld.train_mask_dir, batch_size=BATCH_SIZE)
        train_losses = []
        train_ious = []

        # Sử dụng tqdm để tạo thanh trạng thái cho vòng lặp train
        train_bar = tqdm(train_gen, total=total_train_batches, desc="[Train]", unit="batch")

        for batch_x, batch_y in train_bar:
            # Lan truyền tiến (Forward)
            pred_y = model.forward(batch_x)

            # Tính toán tổn thất (Weighted BCE Loss)
            loss = func.weighted_binary_cross_entropy_loss(pred_y, batch_y)
            train_losses.append(loss)

            # Đánh giá IoU vệt dầu
            iou = func.calculate_soft_iou(pred_y, batch_y)
            train_ious.append(iou)

            # Lan truyền ngược và cập nhật tham số (Backward + SGD)
            loss_grad = func.compute_weighted_bce_gradient(pred_y, batch_y)
            model.backward(loss_grad, LEARNING_RATE)

            # Cập nhật thông tin Loss và IoU thời gian thực ngay trên thanh trạng thái
            train_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "soft_iou": f"{iou:.4f}"
            })

        # Tính toán kết quả trung bình sau khi kết thúc 1 epoch train
        avg_train_loss = np.mean(train_losses)
        avg_train_iou = np.mean(train_ious)
        print(f"-> Kết quả [Train]: Loss TB = {avg_train_loss:.4f} | Soft IoU TB = {avg_train_iou:.4f}")

        # =========================================================
        # 2. GIAI ĐOẠN KIỂM ĐỊNH (VALIDATION)
        # =========================================================
        val_gen = ld.data_generator(ld.val_image_dir, ld.val_mask_dir, batch_size=BATCH_SIZE)
        val_ious = []

        # Sử dụng tqdm cho vòng lặp validation
        val_bar = tqdm(val_gen, total=total_val_batches, desc="[Valid]", unit="batch")

        for val_x, val_y in val_bar:
            val_pred = model.forward(val_x)
            v_iou = func.calculate_soft_iou(val_pred, val_y)
            val_ious.append(v_iou)

            # Cập nhật thông tin IoU kiểm định thời gian thực
            val_bar.set_postfix({"soft_iou": f"{v_iou:.4f}"})

        avg_val_iou = np.mean(val_ious)
        print(f"-> Kết quả [Validation]: Soft IoU kiểm định TB = {avg_val_iou:.4f}")

    print("\n=== QUÁ TRÌNH HUẤN LUYỆN HOÀN THÀNH ===")
    # =========================================================
    # 3. LƯU MÔ HÌNH VÀ LOG HUẤN LUYỆN
    # =========================================================
    print("\n[HỆ THỐNG] Đang lưu trọng số mô hình...")

    # Gom tất cả trọng số W và b của các lớp tích chập lại
    model_weights = {
        "conv1_W": model.conv1.W, "conv1_b": model.conv1.b,
        "conv2_W": model.conv2.W, "conv2_b": model.conv2.b,
        "conv_bottom_W": model.conv_bottom.W, "conv_bottom_b": model.conv_bottom.b,
        "conv3_W": model.conv3.W, "conv3_b": model.conv3.b,
        "conv4_W": model.conv4.W, "conv4_b": model.conv4.b,
        "conv_final_W": model.conv_final.W, "conv_final_b": model.conv_final.b
    }
    # Lưu file trọng số
    np.save("oil_spill_model.npy", model_weights)
    print("-> Đã lưu trọng số vào file: oil_spill_model.npy")

    # Gom log lịch sử huấn luyện để sau này tiện visualize đường cong học tập
    training_logs = {
        "train_loss": history_train_loss,  # Mẹo: Bạn nhớ tạo 2 list rỗng ở đầu file
        "train_iou": history_train_iou,  # để append kết quả trung bình của mỗi epoch vào nhé
        "val_iou": history_val_iou
    }
    np.save("training_logs.npy", training_logs)
    print("-> Đã lưu log lịch sử vào file: training_logs.npy")
    import numpy as np
    import matplotlib.pyplot as plt

    logs = np.load("training_logs.npy", allow_pickle=True).item()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(logs["train_loss"], label="Train Loss")
    plt.title("Đồ thị Loss qua các Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(logs["train_iou"], label="Train Soft IoU")
    plt.plot(logs["val_iou"], label="Validation Soft IoU")
    plt.title("Đồ thị IoU đánh giá")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.show()