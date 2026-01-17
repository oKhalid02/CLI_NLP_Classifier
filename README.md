# 🌟 Arabic NLP Classification CLI Tool 🌟

<div align="center">

![NLP](https://img.shields.io/badge/NLP-Arabic%20Text-blue?style=for-the-badge&logo=language)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-orange?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### 🚀 A Comprehensive Arabic Natural Language Processing Pipeline

*From Raw Text to Smart Classification - All in One Powerful CLI Tool!*

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [📁 Project Structure](#-project-structure)
- [🔧 Prerequisites & Installation](#-prerequisites--installation)
- [🎮 Quick Start Guide](#-quick-start-guide)
- [📚 Detailed Command Guide](#-detailed-command-guide)
- [💡 Complete Workflow Example](#-complete-workflow-example)
- [📊 Output Files & Visualizations](#-output-files--visualizations)
- [🐛 Troubleshooting](#-troubleshooting)
- [👨‍💻 Project Architecture](#-project-architecture)

---

## 🎯 Project Overview

This project is a **comprehensive Arabic NLP classification pipeline** that takes raw Arabic text data and transforms it into trained machine learning models. It's designed to handle the unique challenges of Arabic language processing, including:

✅ **Diacritics removal** (Tashkeel)
✅ **Stop word removal** with Arabic-specific stopwords
✅ **URL and special character filtering**
✅ **Text normalization** and cleaning
✅ **Multiple embedding strategies** (TF-IDF, Model2Vec)
✅ **Machine learning classification** with multiple algorithms
✅ **Comprehensive reporting** and visualizations

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Exploratory Data Analysis (EDA)** | Detailed statistical analysis and visualizations of your dataset |
| 🧹 **Text Preprocessing** | Arabic-specific text cleaning and normalization |
| 🎯 **Text Embedding** | Convert text to numerical vectors using TF-IDF or Model2Vec |
| 🤖 **Model Training** | Train multiple ML models (Logistic Regression, SVM, Random Forest, etc.) |
| 📈 **Performance Reports** | Detailed metrics, confusion matrices, and ROC curves |
| 📊 **Visualizations** | Beautiful charts and graphs for data insights |

---

## 📁 Project Structure

```
Project/
│
├── 📄 main.py                          # Entry point for the CLI
├── 📄 README.md                        # This file
│
├── 📂 commands/                        # Command modules
│   ├── __init__.py
│   ├── eda.py                          # Exploratory Data Analysis
│   ├── preprocess.py                   # Text preprocessing commands
│   ├── embed.py                        # Text embedding commands
│   └── train.py                        # Model training commands
│
├── 📂 data/                            # Input data files
│   ├── CompanyReviews.csv              # Original dataset
│   ├── cleaned.csv                     # After preprocessing
│   ├── final.csv                       # Final cleaned dataset
│   ├── normalized.csv                  # Normalized text
│   └── nostopwords.csv                 # Without stopwords
│
├── 📂 outputs/                         # Generated outputs
│   ├── embeddings/                     # Text embeddings (vectors)
│   ├── models/                         # Trained ML models
│   ├── reports/                        # Performance reports (JSON)
│   └── visualizations/                 # Generated charts & graphs
│
└── 📂 utils/                           # Utility modules
    ├── data_handler.py                 # CSV loading & processing
    ├── arabic_text.py                  # Arabic-specific functions
    ├── metrics.py                      # Evaluation metrics
    └── visualization.py                # Chart generation
```

---

## 🔧 Prerequisites & Installation

### Step 1️⃣: System Requirements

Ensure you have the following installed on your machine:

```bash
# Check Python version (3.8 or higher required)
python --version

# Should output: Python 3.8.x or higher
```

### Step 2️⃣: Clone or Download the Project

```bash
# If using git
cd /Users/khaledalamro/Desktop/NLP_Project/Project

# Or navigate to the project directory
cd Project
```

### Step 3️⃣: Install Python Dependencies

```bash
# Method 1: Using pip (recommended)
pip install -r requirements.txt

# Method 2: Using uv (faster, if installed)
uv sync

# Method 3: Manual installation (if requirements.txt unavailable)
pip install click pandas numpy scikit-learn scipy joblib matplotlib seaborn
```

**Expected packages installed:**
- `click` - CLI framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `scipy` - Scientific computing
- `joblib` - Model serialization
- `matplotlib` - Plotting
- `seaborn` - Advanced visualizations

### Step 4️⃣: Verify Installation

```bash
# Navigate to project directory
cd /Users/khaledalamro/Desktop/NLP_Project/Project

# List all available commands
python main.py --help

# You should see output like:
# Usage: main.py [OPTIONS] COMMAND [ARGS]...
#
#   Arabic NLP Classification CLI Tool
#
# Options:
#   --help  Show this message and exit.
#
# Commands:
#   eda          Exploratory Data Analysis commands
#   preprocess   Text preprocessing commands
#   embed        Text embedding commands
#   train        Train ML models on embeddings
```

---

## 🎮 Quick Start Guide

### 🏃 30-Second Quick Start

```bash
# 1. Explore your data
python main.py eda distribution --csv_path data/CompanyReviews.csv --label_col rating

# 2. Clean and preprocess the text
python main.py preprocess all --csv_path data/CompanyReviews.csv --text_col review_description --output final.csv

# 3. Create text embeddings (choose one)
python main.py embed tfidf --csv_path data/final.csv --text_col review_description --max_features 5000 --output tfidf_vectors.pkl
# OR
python main.py embed model2vec --csv_path data/final.csv --text_col review_description --output model2vec_vectors.pkl

# 4. Train machine learning models
python main.py train --csv_path data/final.csv --input_col outputs/embeddings/model2vec_vectors.pkl --output_col rating --models all

# 5. Check your results!
ls outputs/models/              # Trained models
ls outputs/reports/             # Performance reports
ls outputs/visualizations/      # Charts and confusion matrices
```

---

## 📚 Detailed Command Guide

### 📊 1. Exploratory Data Analysis (EDA)

#### Purpose
Understand your dataset structure, distribution, and characteristics before processing.

#### Command Structure
```bash
python main.py eda <subcommand> [OPTIONS]
```

#### Available Subcommands

##### **1a. View Label Distribution (Pie Chart)**
```bash
python main.py eda distribution \
    --csv_path data/CompanyReviews.csv \
    --label_col rating
```

Creates a **pie chart** showing the distribution of labels in your dataset.

**Parameters:**
| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `--csv_path` | ✅ Yes | string | Path to your CSV file |
| `--label_col` | ✅ Yes | string | Name of the label/target column |
| `--plot_type` | ❌ Optional | string | Chart type: `pie` (default) or `bar` |

**Output:**
- 📊 PNG chart saved to `outputs/visualizations/label_distribution_<label>_pie.png`
- 🎯 Summary statistics printed to console

##### **1b. View Label Distribution (Bar Chart)**
```bash
python main.py eda distribution \
    --csv_path data/CompanyReviews.csv \
    --label_col rating \
    --plot_type bar
```

Creates a **bar chart** showing the distribution of labels.

**Output:**
- 📊 PNG chart saved to `outputs/visualizations/label_distribution_<label>_bar.png`

##### **1c. Text Length Analysis (Word Count)**
```bash
python main.py eda histogram \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --unit words
```

Analyzes text length distribution in **word count** and creates a histogram.

**Parameters:**
| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `--csv_path` | ✅ Yes | string | Path to your CSV file |
| `--text_col` | ✅ Yes | string | Name of the text column |
| `--unit` | ❌ Optional | string | `words` (default) or `chars` for character count |

**Output:**
- 📈 Histogram saved to `outputs/visualizations/text_length_<unit>_hist.png`
- Statistics (mean, median, std dev) printed to console

##### **1d. Text Length Analysis (Character Count)**
```bash
python main.py eda histogram \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --unit chars
```

Analyzes text length distribution in **character count**.

**Output:**
- 📈 Histogram saved to `outputs/visualizations/text_length_chars_hist.png`

**Example Output:**
```
Total samples: 37885
Mean words: 7.47
Median words: 6
Std dev: 5.23
Min words: 1
Max words: 89
```

##### **1e. Remove Statistical Outliers** ⭐ NEW
```bash
python main.py eda remove-outliers \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --method iqr \
    --output clean_data.csv
```

Detects and removes statistical outliers based on text length using either **IQR** or **Z-Score** method.

**Parameters:**
| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `--csv_path` | ✅ Yes | string | - | Path to your CSV file |
| `--text_col` | ✅ Yes | string | - | Name of the text column |
| `--method` | ❌ Optional | string | `iqr` | Detection method: `iqr` or `zscore` |
| `--output` | ✅ Yes | string | - | Output CSV filename (saved to `data/`) |

**IQR Method (Default - Recommended):**
- Calculates Q1 (25th percentile) and Q3 (75th percentile)
- Removes texts outside: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Good for skewed distributions
- More robust to extreme outliers

**Z-Score Method:**
- Calculates mean and standard deviation
- Removes texts with |Z-Score| > 3
- Good for normally distributed data
- Removes extreme values (3+ std devs from mean)

**Output:**
```
Processing: data/CompanyReviews.csv
Text column: review_description
Method: IQR
---
Q1 (25th percentile): 5.0 words
Q3 (75th percentile): 10.0 words
IQR: 5.0 words
Lower bound: 1.0 words
Upper bound: 17.5 words
---
Original rows: 40046
Outliers detected: 1161
Rows kept: 38885
Outliers removed: 2.9%
Saved → data/clean_data.csv
```

**When to Use:**
- 🎯 Before preprocessing: Clean extreme outliers first
- 📊 After EDA: Identify suspicious data points
- 🧹 Before embedding: Ensure consistent text length
- ⚡ For better model training: Remove noise from data

---

### 🧹 2. Text Preprocessing

#### Purpose
Clean and normalize Arabic text by removing diacritics, stopwords, URLs, and special characters.

#### Command Structure
```bash
python main.py preprocess <subcommand> [OPTIONS]
```

#### Available Subcommands

##### **2a. Remove Special Characters, URLs, Numbers, and Diacritics**
```bash
python main.py preprocess remove \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --output cleaned.csv
```

Removes:
- 🔤 Arabic diacritics (Tashkeel: ً ٌ ٍ َ ُ ِ ّ ْ)
- 🔗 URLs and links
- 🔢 Numbers and digits
- ✨ Special characters (keeping only Arabic letters)

**Parameters:**
| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `--csv_path` | ✅ Yes | string | Path to your CSV file |
| `--text_col` | ✅ Yes | string | Name of the text column |
| `--output` | ✅ Yes | string | Output CSV filename |

**Output:**
- 📄 Cleaned CSV file (saved to `data/<output>`)
- 📊 Console report with before/after statistics

##### **2b. Remove Stopwords**
```bash
python main.py preprocess stopwords \
    --csv_path data/cleaned.csv \
    --text_col review_description \
    --output nostopwords.csv
```

Removes common Arabic stopwords like: من، في، هذا، هو، ليس، إلى، و، أو، أن، etc.

**Output:**
- 📄 CSV without stopwords
- 📊 Word count statistics before/after

##### **2c. Normalize Arabic Text**
```bash
python main.py preprocess replace \
    --csv_path data/nostopwords.csv \
    --text_col review_description \
    --output normalized.csv
```

Normalizes Arabic text by:
- 🔤 Converting hamza variants (أ، إ، ؤ) → ا
- 🔤 Converting ة (taa marboota) → ه
- 🔤 Converting ى (alef maksura) → ي

**Output:**
- 📄 Normalized CSV file

##### **2d. Run All Steps at Once (Recommended!)**
```bash
python main.py preprocess all \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --output final.csv
```

**The recommended approach!** Runs the complete preprocessing pipeline in optimal order:

1. Remove special characters and URLs
2. Remove numbers and digits
3. Remove diacritics (tashkeel)
4. Remove stopwords
5. Normalize Arabic characters
6. Clean whitespace

**Output:**
```
Rows before: 40046
Rows after : 37885
Avg words before: 9.37
Avg words after : 7.47
Saved → data/final.csv
```

---

### 🎯 3. Text Embedding

#### Purpose
Convert text into numerical vectors that machine learning models can understand.

#### Command Structure
```bash
python main.py embed <subcommand> [OPTIONS]
```

#### Available Subcommands

##### **3a. TF-IDF Embedding**
```bash
python main.py embed tfidf \
    --csv_path data/final.csv \
    --text_col review_description \
    --max_features 5000 \
    --output tfidf_vectors.pkl
```

Creates **TF-IDF vectors** (Term Frequency-Inverse Document Frequency) using scikit-learn.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | Required | Path to cleaned CSV file |
| `--text_col` | Required | Name of text column to vectorize |
| `--max_features` | 5000 | Maximum number of features to extract |
| `--ngram_range` | "1 2" | Unigrams and bigrams (1-2 word phrases) |
| `--min_df` | 2 | Minimum document frequency |
| `--max_df` | 0.8 | Maximum document frequency |
| `--output` | Required | Output filename (saved to `outputs/embeddings/`) |

**Output:**
- ✅ Trained vectorizer saved
- 📊 Embedding statistics (shape, sparsity, memory usage)
- 💾 Vectors saved as sparse matrix (.pkl file)

**Example Output:**
```
TF-IDF vectors shape: (37885, 5000) | nnz=236867 | approx_mem=2.86 MB
Saved → outputs/embeddings/tfidf_vectors.pkl
```

##### **3b. Model2Vec Embedding**
```bash
python main.py embed model2vec \
    --csv_path data/final.csv \
    --text_col review_description \
    --output model2vec_vectors.pkl
```

Uses **pre-trained Arabic word embeddings** from HuggingFace (Model2Vec-ARBERTv2).

**Output:**
```
Model2Vec vectors shape: (37885, 128) | dtype=float64 | approx_mem=37.00 MB
Saved → outputs/embeddings/model2vec_vectors.pkl
```

**Why Model2Vec?**
- ✨ Pre-trained on large Arabic corpus
- 📉 Lower memory footprint than TF-IDF (128 dims vs 5000)
- 🎯 Semantic similarity captured
- 🚀 Better for small datasets

---

### 🤖 4. Model Training

#### Purpose
Train machine learning models on the embedded text to classify documents.

#### Command Structure
```bash
python main.py train [OPTIONS]
```

#### Main Command
```bash
python main.py train \
    --csv_path data/final.csv \
    --input_col outputs/embeddings/model2vec_vectors.pkl \
    --output_col rating \
    --test_size 0.2 \
    --models knn lr rf
```

**Main Parameters:**
| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `--csv_path` | ✅ | string | - | Path to CSV with data and labels |
| `--input_col` | ✅ | string | - | Path to embeddings file (.pkl) |
| `--output_col` | ✅ | string | - | Target column to predict (label column) |
| `--test_size` | ❌ | float | 0.2 | Test set percentage (20% = 0.2) |
| `--models` | ❌ | string | "knn lr rf" | Models to train (space-separated) |
| `--random_state` | ❌ | int | 42 | Random seed for reproducibility |

#### Single Model Training
```bash
python main.py train \
    --csv_path data/final.csv \
    --input_col outputs/embeddings/tfidf_vectors.pkl \
    --output_col rating \
    --models lr
```

**Available Models:**
- `knn` - K-Nearest Neighbors (fast, simple)
- `lr` - Logistic Regression (fast, interpretable)
- `rf` - Random Forest (ensemble, powerful)
- `svm` - Support Vector Machine (slower, accurate)

- `all` - Train all available models

#### Multiple Models Training
```bash
python main.py train \
    --csv_path data/final.csv \
    --input_col outputs/embeddings/model2vec_vectors.pkl \
    --output_col rating \
    --models "knn lr rf svm"
```

#### Train All Models at Once
```bash
python main.py train \
    --csv_path data/final.csv \
    --input_col outputs/embeddings/model2vec_vectors.pkl \
    --output_col rating \
    --models all
```

#### Training Output

**Console Output:**
```
[train] CSV: data/final.csv
[train] Embeddings source: outputs/embeddings/model2vec_vectors.pkl
[train] Label column: rating
[train] Loaded CSV rows=37885
[train] Loading embeddings from file: outputs/embeddings/model2vec_vectors.pkl
[train] Embeddings shape: (37885, 128)
[train] Detected classes (3): ['-1', '0', '1']
[train] Split: train=30296 test=7574 (test_size=0.2)
[train] Training model: knn
[train] Done: knn -> acc=0.7678 prec=0.5665 rec=0.5441 f1=0.5401
[train] Training model: lr
[train] Done: lr -> acc=0.7811 prec=0.5142 rec=0.5385 f1=0.5260
[train] Training model: rf
[train] Done: rf -> acc=0.7827 prec=0.6158 rec=0.5408 f1=0.5349
[train] Saved best model: outputs/models/best_model_20260117_045535.pkl
[train] Saved report: outputs/reports/training_report_20260117_045535.md

✅ Report saved → outputs/reports/training_report_20260117_045535.md
✅ Best model saved → outputs/models/best_model_20260117_045535.pkl
```

**Generated Files:**
- 📦 `outputs/models/best_model_<timestamp>.pkl` - Best trained model
- 📄 `outputs/reports/training_report_<timestamp>.md` - Full performance report
- 📊 `outputs/visualizations/cm_<model>_<timestamp>.png` - Confusion matrices

**Report Contents:**
```markdown
# Training Report - 2026-01-17 04:55:35

## Dataset Info
- Total samples: 37,885
- Train/Test split: 30,296/7,574 (80/20)
- Classes: 3
- Features: 128

## Model Performance

### K-Nearest Neighbors
- Accuracy:  76.78%
- Precision: 56.65%
- Recall:    54.41%
- F1-Score:  54.01%

### Logistic Regression  
- Accuracy:  78.11%
- Precision: 51.42%
- Recall:    53.85%
- F1-Score:  52.60%

### Random Forest ⭐ (Best)
- Accuracy:  78.27%
- Precision: 61.58%
- Recall:    54.08%
- F1-Score:  53.49%

## Confusion Matrices
[PNG visualizations saved]
```

---

## 💡 Complete Workflow Example

### 🔄 Step-by-Step Tutorial: From Raw Data to Trained Models

#### **Step 1: Prepare Your Data** 
Ensure you have a CSV file with at least a text column and optional label column:

```csv
review_description,rating,company
هذا منتج رائع وممتاز,1,CompanyA
جودة سيئة جداً,-1,CompanyB
خدمة العملاء ممتازة,1,CompanyC
```

**Column names in example:**
- Text column: `review_description`
- Label column: `rating`
- Other columns: `company` (optional)

#### **Step 2: Explore the Data with EDA**

View label distribution:
```bash
python main.py eda distribution \
    --csv_path data/CompanyReviews.csv \
    --label_col rating
```

View text length statistics (word count):
```bash
python main.py eda histogram \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --unit words
```

View text length statistics (character count):
```bash
python main.py eda histogram \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --unit chars
```

*(Optional) Remove statistical outliers:*
```bash
python main.py eda remove-outliers \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --method iqr \
    --output clean_data.csv
```

📊 **Check outputs:**
- `outputs/visualizations/label_distribution_rating_pie.png`
- `outputs/visualizations/text_length_words_hist.png`
- `outputs/visualizations/text_length_chars_hist.png`
- `data/clean_data.csv` (if using outlier removal)

#### **Step 3: Clean the Text (Preprocessing)**

**Optional Step 3a: Remove Outliers**
```bash
python main.py eda remove-outliers \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --method iqr \
    --output outliers_removed.csv
```

**Main Step 3b: Use all preprocessing steps at once (recommended):**
```bash
python main.py preprocess all \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --output final.csv
```

Or if you removed outliers first:
```bash
python main.py preprocess all \
    --csv_path data/outliers_removed.csv \
    --text_col review_description \
    --output final.csv
```

**Detailed Step-by-Step Option (Alternative):**
```bash
# Remove special chars, URLs, numbers, diacritics
python main.py preprocess remove \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --output cleaned.csv

# Remove stopwords
python main.py preprocess stopwords \
    --csv_path data/cleaned.csv \
    --text_col review_description \
    --output nostopwords.csv

# Normalize Arabic characters
python main.py preprocess replace \
    --csv_path data/nostopwords.csv \
    --text_col review_description \
    --output normalized.csv

# Run all preprocessing at once
python main.py preprocess all \
    --csv_path data/normalized.csv \
    --text_col review_description \
    --output final.csv
```

✅ **Result:** Clean, normalized Arabic text ready for embedding

#### **Step 4: Create Embeddings**

**Option A: TF-IDF Embedding (faster, uses more dimensions)**
```bash
python main.py embed tfidf \
    --csv_path data/final.csv \
    --text_col review_description \
    --max_features 5000 \
    --output tfidf_vectors.pkl
```

**Option B: Model2Vec Embedding (pre-trained, semantic)**
```bash
python main.py embed model2vec \
    --csv_path data/final.csv \
    --text_col review_description \
    --output model2vec_vectors.pkl
```

🎯 **Result:** Text converted to numerical vectors

#### **Step 5: Train Models**

**Option A: Train specific models (KNN, Logistic Regression, Random Forest)**
```bash
python main.py train \
    --csv_path data/final.csv \
    --input_col outputs/embeddings/model2vec_vectors.pkl \
    --output_col rating \
    --models "knn lr rf" \
    --test_size 0.2
```

**Option B: Train all available models**
```bash
python main.py train \
    --csv_path data/final.csv \
    --input_col outputs/embeddings/model2vec_vectors.pkl \
    --output_col rating \
    --models all \
    --test_size 0.2
```

🤖 **Result:** Trained models + performance report + confusion matrices

#### **Step 6: Review Results**

List all trained models:
```bash
ls outputs/models/
```

List all reports:
```bash
ls outputs/reports/
```

View the latest report:
```bash
cat outputs/reports/training_report_*.md | tail -100
```

View confusion matrices:
```bash
ls outputs/visualizations/cm_*.png
```

---

## 📊 Output Files & Visualizations

### 📁 Directory Structure After Running

```
outputs/
├── embeddings/
│   ├── tfidf.joblib                    # TF-IDF vectorizer
│   ├── tfidf_vectors.joblib            # TF-IDF sparse matrix
│   └── model2vec_vectors.pkl           # Model2Vec embeddings
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── logistic_regression_vectorizer.pkl
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   └── ... (more models)
│
├── reports/
│   ├── training_report_20260117_045535.md  # Comprehensive report
│   ├── training_report_20260117_045625.md  # Another report
│   ├── eda_report_<timestamp>.json
│   └── embedding_report_<timestamp>.json
│
└── visualizations/
    ├── label_distribution.png           # Label frequency chart
    ├── text_length_distribution.png     # Text length histogram
    ├── confusion_matrix.png             # Confusion matrix heatmap
    ├── roc_curve.png                    # ROC curve
    ├── feature_importance.png           # Top features
    └── model_comparison.png             # Performance comparison
```

### 📈 Sample Report Content

```markdown
# Training Report - 2026-01-17 04:55:35

## Configuration
- Dataset: data/final.csv
- Input: outputs/embeddings/tfidf.joblib
- Target: rating
- Models: logistic-regression
- Test Size: 20%

## Results Summary
- Accuracy: 84.7%
- Precision: 84.2%
- Recall: 84.0%
- F1-Score: 84.1%

## Confusion Matrix
```
             Negative    Positive
Negative        800          50
Positive         60         290
```

## ROC-AUC Score: 0.912

## Top 10 Features (Words)
1. رائع - 0.523
2. ممتاز - 0.498
3. جودة - 0.456
...
```

---

## 🎁 Bonus Features (Available Now! ✨)

### 🎯 **Outlier Detection & Removal** ⭐

Remove statistical outliers from your dataset to improve data quality before training.

```bash
# Using IQR method (recommended)
python main.py eda remove-outliers \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --method iqr \
    --output clean_data.csv

# Using Z-Score method
python main.py eda remove-outliers \
    --csv_path data/CompanyReviews.csv \
    --text_col review_description \
    --method zscore \
    --output clean_data.csv
```

**Benefits:**
- ✅ Removes extremely short/long reviews
- ✅ Improves model training by removing noise
- ✅ Makes data more consistent
- ✅ Two robust statistical methods

**Example Workflow with Outlier Removal:**
```bash
# 1. Explore raw data
python main.py eda histogram --csv_path data/CompanyReviews.csv --text_col review_description --unit words

# 2. Remove outliers
python main.py eda remove-outliers --csv_path data/CompanyReviews.csv --text_col review_description --method iqr --output clean_data.csv

# 3. Preprocess cleaned data
python main.py preprocess all --csv_path data/clean_data.csv --text_col review_description --output final.csv

# 4. Continue with embedding & training
python main.py embed model2vec --csv_path data/final.csv --text_col review_description --output model2vec_vectors.pkl
python main.py train --csv_path data/final.csv --input_col outputs/embeddings/model2vec_vectors.pkl --output_col rating --models all
```

---

### ❌ Problem: "Command not found" or "python: command not found"

**Solution:**
```bash
# Use full path to Python
/usr/bin/python3 main.py --help

# Or verify Python is installed
which python3
```

### ❌ Problem: "ModuleNotFoundError: No module named 'click'"

**Solution:**
```bash
# Install missing dependencies
pip install click pandas numpy scikit-learn scipy

# Or reinstall all
pip install -r requirements.txt
```

### ❌ Problem: "FileNotFoundError: data/CompanyReviews.csv"

**Solution:**
```bash
# Check file exists
ls -la data/

# Use correct path (case-sensitive on Mac)
python main.py preprocess all --csv_path data/CompanyReviews.csv --text_col text --output_path data/cleaned.csv
```

### ❌ Problem: "MemoryError" with large datasets

**Solution:**
```bash
# Reduce max_features for embedding
python main.py embed tfidf \
    --csv_path data/cleaned.csv \
    --text_col text \
    --max_features 2000 \
    --output_path outputs/embeddings/tfidf.joblib
```

### ❌ Problem: "No such file or directory: outputs/embeddings/..."

**Solution:**
```bash
# Directories are created automatically, but ensure embeddings exist
python main.py embed tfidf \
    --csv_path data/cleaned.csv \
    --text_col text \
    --output_path outputs/embeddings/tfidf.joblib
```

### ❌ Problem: Model training is very slow

**Solution:**
```bash
# Use faster models first
python main.py train \
    --csv_path data/cleaned.csv \
    --input_col outputs/embeddings/tfidf.joblib \
    --output_col rating \
    --models logistic-regression  # Fast!

# Avoid these for large datasets:
# - mlp (Neural network - slowest)
# - svm (Support Vector Machine - slow)
```

---

## 👨‍💻 Project Architecture

### 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **CLI Framework** | Click | Command-line interface |
| **Data Processing** | Pandas, NumPy | Manipulation & analysis |
| **ML Algorithms** | Scikit-learn | Model training & evaluation |
| **Text Processing** | Scikit-learn | TF-IDF vectorization |
| **Visualization** | Matplotlib, Seaborn | Charts & graphs |
| **Serialization** | Joblib, Pickle | Model persistence |

### 📦 Module Breakdown

#### **commands/eda.py**
- Statistical analysis of datasets
- Label distribution analysis
- Text length statistics
- Visualization generation

#### **commands/preprocess.py**
- Diacritics removal (Tashkeel)
- Stopword removal (Arabic-specific)
- URL and digit removal
- Text normalization
- Character filtering

#### **commands/embed.py**
- TF-IDF vectorization
- Model2Vec embeddings
- Sparse matrix generation
- Vector serialization

#### **commands/train.py**
- Multiple ML algorithms
- Train/test splitting
- Model evaluation
- Metrics calculation
- Report generation

#### **utils/data_handler.py**
- CSV loading and validation
- Column existence checking
- Data type conversion

#### **utils/visualization.py**
- Pie charts for labels
- Histograms for text length
- Confusion matrices
- ROC curves
- Feature importance plots

#### **utils/metrics.py**
- Accuracy, Precision, Recall, F1
- Confusion matrix generation
- ROC curve calculation
- Performance comparison

---

## � Learning Tips

### 📚 Understanding the Workflow

1. **EDA First** → Always explore your data before processing
2. **Preprocess Second** → Clean Arabic text properly
3. **Embed Third** → Convert text to numbers
4. **Train Last** → Build and evaluate models

### 🔍 Best Practices

✅ Always save your reports
✅ Start with small datasets to test
✅ Compare multiple models
✅ Keep track of your preprocessing steps
✅ Save your best models for later use

### 📖 Arabic NLP Challenges

- **Diacritics**: Same words can look different with marks
- **Stopwords**: Common words that don't add meaning
- **Morphology**: Rich word formations
- **Short vowels**: Often omitted in text

This project handles all these challenges! 🎉

---
