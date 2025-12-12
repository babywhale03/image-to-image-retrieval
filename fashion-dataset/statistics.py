import os
import matplotlib.pyplot as plt
from collections import Counter

ROOT_DIR = "."
GROUPED_DIR = os.path.join(ROOT_DIR, "preprocessed_data")

# Output directory for statistics
OUTPUT_DIR = os.path.join(ROOT_DIR, "stats_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

style_counts = []
total_images = 0

# Count images per style category
for style_name in os.listdir(GROUPED_DIR):
    style_path = os.path.join(GROUPED_DIR, style_name)
    if not os.path.isdir(style_path):
        continue

    # Only count image files    
    image_files = [
        f for f in os.listdir(style_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(image_files)
    style_counts.append(count)
    total_images += count

# Compute statistics
if not style_counts:
    print("스타일 폴더를 찾지 못했습니다.")
else:
    num_categories = len(style_counts)
    avg_size = sum(style_counts) / num_categories
    min_size = min(style_counts)
    max_size = max(style_counts)

    print("===== Style Group Statistics =====")
    print(f"Total images: {total_images}")
    print(f"Number of style categories: {num_categories}")
    print(f"Average images per style: {avg_size:.2f}")
    print(f"Minimum images per style: {min_size}")
    print(f"Maximum images per style: {max_size}")

    # Save to text file
    with open(os.path.join(OUTPUT_DIR, "style_counts.txt"), "w") as f:
        f.write("===== Style Group Statistics =====\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Number of style categories: {num_categories}\n")
        f.write(f"Average images per style: {avg_size:.2f}\n")
        f.write(f"Minimum images per style: {min_size}\n")
        f.write(f"Maximum images per style: {max_size}\n")
    # Distribution output
    dist = Counter(style_counts)
    print("\n===== Distribution of Image Counts =====")
    for img_count, num_styles in sorted(dist.items()):
        print(f"{img_count} images: {num_styles} styles")

    # Save distribution to text file
    with open(os.path.join(OUTPUT_DIR, "style_counts.txt"), "a") as f:
        f.write("\n===== Distribution of Image Counts =====\n")
        for img_count, num_styles in sorted(dist.items()):
            f.write(f"{img_count} images: {num_styles} styles\n")

    # Generate and save plots
    # 1) Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(style_counts, bins=range(1, max_size + 2), edgecolor='black', alpha=0.7)
    plt.title("Distribution of Image Count per Style Category")
    plt.xlabel("Images per Category")
    plt.ylabel("Number of Categories")
    plt.xticks(range(1, max_size + 1))
    plt.grid(axis="y", alpha=0.3)
    
    hist_path = os.path.join(OUTPUT_DIR, "histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(style_counts, vert=True, patch_artist=True)
    plt.title("Boxplot of Category Image Counts")
    plt.ylabel("Number of Images")
    
    boxplot_path = os.path.join(OUTPUT_DIR, "boxplot.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()
