import os
import json
import shutil
from collections import defaultdict

# Define directories
ROOT_DIR = "."
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
STYLES_DIR = os.path.join(ROOT_DIR, "styles")
OUTPUT_DIR = os.path.join(ROOT_DIR, "preprocessed_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility function to create safe folder/file names
def safe_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )

# Extract style key from JSON metadata
def extract_style_key(style_json_path: str):
    with open(style_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    data = obj.get("data", {})

    # Extract relevant fields
    article_type_obj = data.get("articleType", {})
    article_type = article_type_obj.get("typeName", None)
    base_colour = data.get("baseColour", None)
    usage = data.get("usage", None)

    if not article_type or not base_colour or not usage:
        return None

    article_type = safe_name(article_type)
    base_colour = safe_name(base_colour)
    usage = safe_name(usage)

    return (article_type, base_colour, usage)

def main():
    # Group item IDs by style key
    style_groups = defaultdict(list)

    # Iterate over style JSON files
    for fname in os.listdir(STYLES_DIR):
        if not fname.endswith(".json"):
            continue

        style_path = os.path.join(STYLES_DIR, fname)
        style_key = extract_style_key(style_path)
        if style_key is None:
            continue

        item_id = os.path.splitext(fname)[0]
        style_groups[style_key].append(item_id)

    # Filter groups with 4 to 20 images
    filtered_groups = {}

    for style_key, id_list in style_groups.items():
        valid_ids = [
            iid for iid in id_list
            if os.path.exists(os.path.join(IMAGES_DIR, f"{iid}.jpg"))
        ]

        if 4 <= len(valid_ids) <= 20:
            filtered_groups[style_key] = valid_ids

    print(f"전체 스타일 그룹 수: {len(style_groups)}")
    print(f"4개 이상 20개 이하 이미지가 있는 스타일 그룹 수: {len(filtered_groups)}")

    # Copy images to output directory
    for style_key, id_list in filtered_groups.items():
        article_type, base_colour, usage = style_key
        group_folder_name = f"{article_type}_{base_colour}_{usage}"
        group_folder_path = os.path.join(OUTPUT_DIR, group_folder_name)
        os.makedirs(group_folder_path, exist_ok=True)

        for item_id in id_list:
            src_img_path = os.path.join(IMAGES_DIR, f"{item_id}.jpg")
            if not os.path.exists(src_img_path):
                continue

            dst_img_path = os.path.join(group_folder_path, f"{item_id}.jpg")
            shutil.copy2(src_img_path, dst_img_path)

if __name__ == "__main__":
    main()
