# NLP_Research_Assistant
Official Repository of the Unsupervised Project

*A balanced scientific dataset for NLP vs. Non-NLP document classification*

This repository contains the complete pipeline used to build a high-quality, balanced dataset for the **Doculizer** research assistant.  
The goal of this stage is to automatically gather, clean, filter, and structure scientific papers in order to train a robust classifier that detects whether an uploaded document is **NLP-related** or **Non-NLP**.

---



## 🗂 Dataset Composition

| Category | Count | Description |
|----------|-------|-------------|
| **NLP-related** | 1200 | Scientific papers from arXiv (cs.CL) + Semantic Scholar queries |
| **Non-NLP** | 1200 | Scientific papers from non-NLP domains (Physics, Math, Biology, CV, Robotics…) |

Total documents: **2400**

Each document includes:
- `paper_id`
- `title`
- `abstract`
- `label` (nlp_related / not_nlp)
- `subcategory` (for NLP papers)
- `source`
- `filepath` (local .txt file containing the cleaned text)

---

## 📚 Data Sources

The dataset is collected from two major academic repositories:

### **1. arXiv**
Used for both:
- NLP papers: `cs.CL`
- Non-NLP papers:  
  `physics`, `math.AG`, `math.AP`, `q-bio`, `cs.CV`, `cs.LG`, `cs.RO`, `stat.ML`

### **2. Semantic Scholar**
Used to enrich the NLP category with keyword-based queries:
- "NLP"
- "natural language processing"
- "machine translation"
- "summarization"
- "question answering"

> Note: API-based collection is restricted; we fallback to arXiv when rate limits occur.

---

## 🧹 Preprocessing & Cleaning Pipeline

Each paper undergoes several preprocessing steps:

1. **Whitespace normalization**
2. **Control character cleaning**
3. **Abstract length filtering (≥ 50 tokens)**
4. **Keyword-based exclusion to avoid NLP contamination in Non-NLP papers**
5. **Subcategory assignment for NLP papers**  
   (e.g., MT, QA, Summarization, NER, Sentiment, Dialogue, LLM)

---

## 🔎 Duplicate Removal (Fuzzy Matching)

To guarantee dataset quality, a fuzzy deduplication step is applied using:

- **SHA1 fingerprinting** (exact duplicates)
- **FuzzyWuzzy – token_sort_ratio ≥ 90** (semantic duplicates)

This ensures:
- No repeated papers  
- No near-duplicates  
- No noisy titles/abstracts

---

## ⚖️ Dataset Balancing

After collection and deduplication:

- The NLP and Non-NLP sets are randomly shuffled  
- Each category is trimmed to the target size:  
  **1200 + 1200 documents**

This avoids class imbalance and prevents overfitting during classifier training.

---

---

# 📘 **SciBERT Classification Module**

This module implements the **core binary classifier** used in *Doculizer* to determine whether an academic paper is **NLP-related** or **not NLP-related**.
The classifier acts as the **first logical gate** of the system: only papers classified as NLP-related proceed to the downstream pipeline.

---

## 🔬 **Model Choice: Why SciBERT?**

We selected **SciBERT (allenai/scibert_scivocab_uncased)** for fine-tuning because:

* It is **pretrained on 1.14M scientific papers**, making it inherently aligned with our domain.
* It performs significantly better than general-domain BERT models on scientific tasks.
* It requires **minimal compute** to achieve high accuracy on short-text classification (title + abstract).

SciBERT offers the best balance between:

* **Performance**
* **Training efficiency**
* **Low carbon footprint**
* **Domain adaptation**

---

## ⚙️ **Training Setup**

| Component                   | Configuration                           |
| --------------------------- | --------------------------------------- |
| Base model                  | SciBERT (uncased scientific vocabulary) |
| Task                        | Binary classification (NLP vs. Non-NLP) |
| Input                       | Title + Abstract concatenated           |
| Max sequence lengths tested | 16, 32, 64                              |
| Optimizer                   | AdamW                                   |
| Loss function               | Cross-entropy                           |
| Batch size                  | 8                                       |
| Epochs tested               | 3, 5, 10                                |
| Validation split            | 15%                                     |
| Test split                  | 20%                                     |

We trained multiple configurations to find the optimal trade-off between accuracy and energy consumption.

---

## 📊 **Evaluation Metrics**

We report:

* **Precision**
* **Recall**
* **F1-score**
* **Accuracy**

And additionally, we measured **CO₂ emissions** using the `codecarbon` library to ensure environmental transparency.

---

## 🧪 **Ablation Study Results**

### **Final Comparison of Experiments**

| Experiment             | Epochs | Max Tokens | Train F1 | Val F1 | Test F1  | Carbon (g CO₂eq) |
| ---------------------- | ------ | ---------- | -------- | ------ | -------- | ---------------- |
| **Best (recommended)** | **3**  | **64**     | 1.00     | 0.99   | **0.99** | **1.965 g**      |
| Exp 2                  | 5      | 32         | 1.00     | 0.99   | 0.99     | 2.199 g          |
| Exp 3                  | 5      | 16         | 1.00     | 0.98   | 0.97     | 4.958 g          |
| Exp 4                  | 10     | 32         | 1.00     | 0.99   | 0.98     | 10.730 g         |

---

## 🏆 **Chosen Model**

We selected the configuration:

### **SciBERT + 3 Epochs + 64 Max Tokens**

Because it provides:

* **State-of-the-art performance** on our dataset
* **High generalization** without overfitting
* **Best environmental footprint**
* **Fast inference suitable for production**

This model achieved:

```
Test Precision: 0.99
Test Recall:    0.99
Test F1-score:  0.99
```

---

## 🌱 **Carbon-Aware Machine Learning**

Training experiments were monitored using **CodeCarbon**, ensuring:

* Transparent reporting of energy consumption
* Reduced environmental impact
* Responsible ML practices aligned with modern research standards

---



لو بدك، فيني أكتب لك **قسم الـ API**،
**قسم الـ evaluation visualizations**،
أو **قسم الarchitecture** للـ README أيضاً.

