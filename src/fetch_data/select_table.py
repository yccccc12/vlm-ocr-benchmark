'''
This script organizes the selected table images and their corresponding JSON ground truth files 
into separate folders based on predefined levels. Each level contains a specific set of tables, 
and the script copies the relevant images and JSON files from the source directories to the appropriate destination folders under "table_by_level".
'''
import os
import shutil

# Source folders
img_src_folder = r"data\raw\table\images"   # raw images
gt_src_folder = r"data\raw\table\gt"        # JSON ground truth

# Destination root
dst_root = r"data\raw\table_by_level"

# Define selected tables for each level
levels = {
    "level_1": ["0016", "0096", "0101", "0135", "0138"],
    "level_2": ["0079", "0155", "0203", "0281", "0704"],
    "level_3": ["0143", "0331", "0534", "0584", "0804"],
    "level_4": ["0047", "0329", "0464", "0687", "0773"]
}

# levels = {
#     "level_1": ["0016", "0096", "0101", "0135", "0138", "0161"],
#     "level_2": ["0056", "0079", "0155", "0170", "0203", "0281",
#                 "0445", "0458", "0516", "0704", "0851"],
#     "level_3": ["0094", "0116", "0143", "0163", "0177", "0193",
#                 "0230", "0331", "0534", "0584", "0754", "0804"],
#     "level_4": ["0047", "0087", "0156", "0329", "0464", "0687",
#                 "0721", "0737", "0773"]
# }

# Copy images and JSON ground truth to table_by_level
for level_name, image_numbers in levels.items():
    img_folder = os.path.join(dst_root, level_name, "img")
    gt_folder = os.path.join(dst_root, level_name, "gt")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)

    for num in image_numbers:
        img_filename = f"table_{num}.jpg"
        gt_filename = f"table_{num}.json"

        img_src_path = os.path.join(img_src_folder, img_filename)
        gt_src_path = os.path.join(gt_src_folder, gt_filename)

        if os.path.exists(img_src_path):
            shutil.copy2(img_src_path, os.path.join(img_folder, img_filename))
        else:
            print(f"Warning: Image {img_filename} not found")

        if os.path.exists(gt_src_path):
            shutil.copy2(gt_src_path, os.path.join(gt_folder, gt_filename))
        else:
            print(f"Warning: Ground truth {gt_filename} not found")

        print(f"Copied {img_filename} and {gt_filename} to {level_name}")