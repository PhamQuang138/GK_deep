import numpy as np
import os
import matplotlib.pyplot as plt
import load_data as ld
import function as func

def evaluate_model(model_path="oil_spill_model.npy", threshold=0.8, num_visualize=2):
    # 1. Khởi tạo mạng và nạp trọng số đã train thành công
    model = func.OilSpillSegmentationNet()
    weights = np.load(model_path, allow_pickle=True).item()
    model.conv1.W = weights["conv1_W"]; model.conv1.b = weights["conv1_b"]
    model.conv2.W = weights["conv2_W"]; model.conv2.b = weights["conv2_b"]
    model.conv_bottom.W = weights["conv_bottom_W"]; model.conv_bottom.b = weights["conv_bottom_b"]
    model.conv3.W = weights["conv3_W"]; model.conv3.b = weights["conv3_b"]
    model.conv4.W = weights["conv4_W"]; model.conv4.b = weights["conv4_b"]
    model.conv_final.W = weights["conv_final_W"]; model.conv_final.b = weights["conv_final_b"]

    # 2. Gọi dữ liệu validation (Batch_size=1 để đánh giá chi tiết từng ảnh)
    val_gen = ld.data_generator(ld.val_image_dir, ld.val_mask_dir, batch_size=1)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    # Danh sách lưu trữ thông tin phục vụ vẽ trực quan mẫu ảnh
    vis_samples = []

    print("[HỆ THỐNG] Đang quét toàn bộ tập dữ liệu kiểm định...")
    for idx, (val_x, val_y) in enumerate(val_gen):
        # Dự đoán xác suất
        pred_y = model.forward(val_x)

        # Ép ngưỡng chuẩn cho việc tính toán thống kê (Vệt dầu = 1)
        pred_binary = (pred_y[0, 0] > threshold).astype(np.uint8)
        true_binary = (val_y[0, 0] > 0.5).astype(np.uint8)

        # Tính toán các pixel TP, FP, FN, TN cho ảnh này
        tp = np.sum((pred_binary == 1) & (true_binary == 1))
        fp = np.sum((pred_binary == 1) & (true_binary == 0))
        fn = np.sum((pred_binary == 0) & (true_binary == 1))
        tn = np.sum((pred_binary == 0) & (true_binary == 0))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # Lưu lại mẫu phục vụ trực quan hóa
        if idx < num_visualize:
            # =========================================================
            # ĐỒNG BỘ LOGIC: TẠO MẶT NẠ ĐẢO NGƯỢC (VẾT DẦU ĐEN - NỀN TRẮNG)
            # =========================================================
            vis_pred_black_oil = (pred_y[0, 0] < threshold).astype(np.uint8) * 255
            vis_true_black_oil = (val_y[0, 0] < 0.5).astype(np.uint8) * 255

            vis_samples.append({
                "raw_img": val_x[0, 0],
                "true_mask_vis": vis_true_black_oil,
                "pred_mask_vis": vis_pred_black_oil,
                "prob_map": pred_y[0, 0]
            })

    # 3. Tính toán các chỉ số thống kê tổng thể
    eps = 1e-8
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)

    # 4. In bảng báo cáo kết quả ra terminal
    print("\n=============================================")
    print("      BÁO CÁO ĐÁNH GIÁ MÔ HÌNH VẾT DẦU       ")
    print("=============================================")
    print(f"Tổng số pixel Vệt Dầu đoán trúng (TP): {total_tp}")
    print(f"Tổng số pixel Báo Động Nhầm      (FP): {total_fp}")
    print(f"Tổng số pixel Bị Bỏ Sót          (FN): {total_fn}")
    print(f"Tổng số pixel Nước Biển chuẩn    (TN): {total_tn}")
    print("---------------------------------------------")
    print(f"--> ĐỘ CHÍNH XÁC (Precision) : {precision * 100:.2f}%")
    print(f"--> ĐỘ THU HỒI   (Recall)    : {recall * 100:.2f}%")
    print(f"--> ĐIỂM DICE    (F1-Score)  : {f1_dice * 100:.2f}%")
    print(f"--> CHỈ SỐ IoU   (Jaccard)   : {iou * 100:.2f}%")
    print("=============================================")

    # =========================================================
    # BIỂU ĐỒ 1: VẼ LỚP MA TRẬN NHẦM LẪN (CONFUSION MATRIX HEATMAP)
    # =========================================================
    cm = np.array([[total_tn, total_fp],
                   [total_fn, total_tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im)

    classes = ["Nước biển (0)", "Vệt dầu (1)"]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Nhãn Mô Hình Đoán (Predicted)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight='bold')
    ax.set_title("Ma Trận Nhầm Lẫn Trên Tập Pixel Kiểm Định", fontsize=12, fontweight='bold', pad=12)

    for i in range(2):
        for j in range(2):
            color_text = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\npixel", ha="center", va="center", color=color_text, fontweight='bold')

    plt.tight_layout()

    # =========================================================
    # BIỂU ĐỒ 2: SO SÁNH TRỰC QUAN ĐỒNG BỘ MÀU (VẾT DẦU ĐEN - NỀN TRẮNG)
    # =========================================================
    plt.figure(figsize=(14, 3.5 * num_visualize))

    for i, sample in enumerate(vis_samples):
        # Cột 1: Ảnh SAR gốc
        plt.subplot(num_visualize, 4, i * 4 + 1)
        plt.imshow(sample["raw_img"], cmap="gray")
        plt.title(f"Mẫu {i + 1}: Ảnh SAR Gốc")
        plt.axis("off")

        # Cột 2: Bản đồ nhiệt xác suất vết dầu (Giữ hệ Jet để thấy lõi xác suất)
        plt.subplot(num_visualize, 4, i * 4 + 2)
        plt.imshow(sample["prob_map"], cmap="jet", vmin=0, vmax=1)
        plt.title(f"Mẫu {i + 1}: Bản Đồ Xác Suất")
        plt.axis("off")

        # Cột 3: Nhãn thực tế đảo ngược màu (Vết dầu đen - Nền trắng)
        plt.subplot(num_visualize, 4, i * 4 + 3)
        plt.imshow(sample["true_mask_vis"], cmap="gray", vmin=0, vmax=255)
        plt.title(f"Mẫu {i + 1}: Nhãn Thật (Ground Truth)")
        plt.axis("off")

        # Cột 4: Mặt nạ mạng tự phân đoạn đảo ngược màu (Vết dầu đen - Nền trắng)
        plt.subplot(num_visualize, 4, i * 4 + 4)
        plt.imshow(sample["pred_mask_vis"], cmap="gray", vmin=0, vmax=255)
        plt.title(f"Mẫu {i + 1}: Kết Quả Dự Đoán (Threshold={threshold})")
        plt.axis("off")

    plt.suptitle("Trực Quan Hóa Kết Quả Phân Đoạn Vệt Tràn Dầu Trên Tập Validation", fontsize=14, fontweight='bold',
                 y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()