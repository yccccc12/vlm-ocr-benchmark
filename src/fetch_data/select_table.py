'''
This script fetches the specific tables used for the "table_by_level" difficulty
split directly from the Hugging Face Datasets API.
'''
import requests
import os
import json
import time

# Parameters (same dataset/split as table.py)
dataset_name = "docling-project/PubTabNet_OTSL"
split = "val"

# Destination root
dst_root = r"data\raw\table_by_level"

# Define selected tables for each level (manually chosen indices)
levels = {
    "level_1": ["0016", "0096", "0101", "0135", "0138"],
    "level_2": ["0079", "0155", "0203", "0281", "0704"],
    "level_3": ["0143", "0331", "0534", "0584", "0804"],
    "level_4": ["0047", "0329", "0464", "0687", "0773"]
}

url = "https://datasets-server.huggingface.co/rows"

for level_name, image_numbers in levels.items():
    img_folder = os.path.join(dst_root, level_name, "img")
    gt_folder = os.path.join(dst_root, level_name, "gt")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)

    for table_id in image_numbers:
        offset = int(table_id)
        file_id = f"table_{table_id}"

        params = {
            "dataset": dataset_name,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": 1
        }

        print(f"Fetching {file_id} ({level_name})...")

        response = requests.get(url, params=params)

        if response.status_code == 200:
            rows = response.json().get("rows", [])

            if not rows:
                print(f"Warning: no row returned for {file_id}")
                continue

            row_data = rows[0]["row"]
            image_url = row_data["image"]["src"]

            # Save image
            img_resp = requests.get(image_url)
            img_path = os.path.join(img_folder, f"{file_id}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_resp.content)

            # Save GT text
            gt_data = {
                "filename": row_data.get("filename"),
                "imgid": row_data.get("imgid"),
                "rows": row_data.get("rows"),
                "cols": row_data.get("cols"),
                "html_restored": row_data.get("html_restored"),
                "html": row_data.get("html"),
                "otsl": row_data.get("otsl"),
                "cells": row_data.get("cells")
            }

            gt_path = os.path.join(gt_folder, f"{file_id}.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_data, f, ensure_ascii=False, indent=2)

            print(f"Saved {file_id} to {level_name}")
            time.sleep(0.3)
        else:
            print(f"Error fetching {file_id}: {response.status_code} - {response.text}")

print("Done fetching table_by_level samples.")
