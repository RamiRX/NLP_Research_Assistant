# NLP_Research_Assistant
Official Repository of the Unsupervised Project
#ي
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



