import kagglehub
import shutil

# download (to default cache)
path = kagglehub.dataset_download("bakhtiyar2222/deep-sar-oil-spill-segmentation-refined")

# move to your desired folder
shutil.move(path, "D:/Deep_SAR/dataset")

print("Moved to:", "D:/Deep_SAR/dataset")