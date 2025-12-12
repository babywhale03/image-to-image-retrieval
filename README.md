# Image-to-Image Retrieval for Fashion Items 

This project implements a simple image-to-image retrieval pipeline for fashion items. The goal is to compare a **naïve heuristic-based baseline** with a more powerful **embedding-based AI pipeline** using a pre-trained vision model.

## 📌 Task Description

Given a **query image** of a fashion item, the model retrieves the **top-K most visually similar images** from a database. Images belonging to the same "style group" as the query are treated as relevant results.

This task simulates real-world applications such as:
* Visual search in e-commerce platforms.
* Finding visually similar or alternative products.
* Fashion item recommendation systems.

## 📂 Dataset

### Source
* **Original Data:** [Fashion Product Images Dataset (Kaggle)](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)
* **Content:** Fashion product images with JSON metadata (article type, base color, usage, etc.).

### Style Group Construction
Since the dataset does not provide ground-truth labels specifically for image retrieval, images were reorganized into **style categories** defined by a combination of attributes:
`ArticleType` + `BaseColour` + `Usage`

### Filtering & Statistics
Only style groups containing **4 to 20 images** were retained to ensure meaningful evaluation.

* **Total Images:** 6,217
* **Style Categories:** 697
* **Average Images per Style:** 8.92

## ⚙️ Methods

### 1. Naïve Baseline: Color Histogram Retrieval
* **Representation:** Each image is represented by a 3D RGB color histogram.
* **Similarity:** Computed using **Cosine Similarity** between histograms.

### 2. AI Pipeline: ResNet-18 Embedding Retrieval
* **Model:** Pre-trained **ResNet-18** (on ImageNet) used as a feature extractor.
* **Representation:** The final global average pooled feature is used as the image embedding.
* **Processing:** Embeddings are L2-normalized.
* **Similarity:** Computed using **Cosine Similarity** (Nearest-Neighbor ranking).

## 📊 Evaluation

For each query image, the retrieval results are evaluated against the ground truth (images in the same style category). The query image itself is excluded from the results.

**Metrics:**
* **Precision@5**
* **Recall@5**
* **Hit@5**

Both pipelines are evaluated on the identical dataset to ensure a fair comparison.

## 📁 Repository Structure

```bash
.
├── fashion-dataset/             # Fashion dataset
│   └── categorize.py            # Categorize based on original dataset
│   └── statistics.py            # Compute statistics of the dataset
├── notebooks/                   # Main codes
│   ├── ai_pipeline.py           # AI pipeline code
│   └── baseline_pipeline.py     # Baseline pipeline code
│   └── pipeline_demo.ipynb      # Evaluation and Analysis
│   └── ai_pipeline.py           # Utils for pipeline execution
├── viz/                         # Result visualizations
├── download.py                  # Dataset download 
├── requirements.txt
└── README.md