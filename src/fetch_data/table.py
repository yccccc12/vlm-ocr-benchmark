'''
This script fetches the first 1000 samples from the "val" split 
of the "docling-project/PubTabNet_OTSL" dataset using the Hugging Face Datasets API.
'''
import requests
import os
import json
import time

# Save data to data/raw/table
save_dir = "data/raw/table"
images_dir = os.path.join(save_dir, "images")
gt_dir = os.path.join(save_dir, "gt")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# Parameters to fetch first 1000 rows from the test split
dataset_name = "docling-project/PubTabNet_OTSL"
split = "val"
total_samples = 1000
batch_size = 100

count = 0

# URL and parameters
url = "https://datasets-server.huggingface.co/rows"

for offset in range(0, total_samples, batch_size):

    params = {
        "dataset": dataset_name,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": batch_size
    }

    print(f"Fetching offset {offset}...")

    # Fetch data
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        rows = data.get("rows", [])
        
        for i, item in enumerate(rows):
            
            row_data = item['row']

            # Stop exactly at total_samples
            if count >= total_samples:
                break
            
            image_url = row_data['image']['src']

            file_id = f"table_{i + offset:04d}"


            # Save image
            img_resp = requests.get(image_url)
            img_path = os.path.join(images_dir, f"{file_id}.jpg")
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

            gt_path = os.path.join(gt_dir, f"{file_id}.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_data, f, ensure_ascii=False, indent=2)

            count += 1

        print(f"Downloaded so far: {count}")
        time.sleep(0.5)

        if count >= total_samples:
            break 
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
print(f"Successfully fetched {count} samples from {dataset_name}.")
