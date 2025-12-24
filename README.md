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
Model URL on Google Drive:
https://drive.google.com/file/d/1-uf26t8kUTh60O16q8eVGdaeD9oOHlzY/view?usp=sharing
---

## 🌱 **Carbon-Aware Machine Learning**

Training experiments were monitored using **CodeCarbon**, ensuring:

* Transparent reporting of energy consumption
* Reduced environmental impact
* Responsible ML practices aligned with modern research standards

---

## 🔎 Transition to Semantic Retrieval

While **SciBERT** is highly effective for the initial binary classification (filtering NLP vs. non-NLP papers), it is not optimized for ranking documents based on query relevance.

The next stage of the project introduces **Sentence-BERT (SBERT)** models to handle semantic retrieval and ranking, replacing traditional TF-IDF-based approaches.

### Semantic Retrieval with SBERT

#### Objective

After filtering papers with SciBERT, we rank the accepted papers by semantic relevance to a user's query. Unlike classical bag-of-words approaches (e.g., TF-IDF), our objective is to capture **semantic similarity** beyond exact lexical overlap. This is crucial in scientific literature where similar concepts are often described using different terminology.

#### Models Compared

We evaluated three Sentence-BERT (SBERT) retrievers:

* **MiniLM** (`all-MiniLM-L6-v2`): A lightweight and efficient model, commonly used as a baseline for semantic search.
* **MPNet** (`all-mpnet-base-v2`): A higher-capacity model designed to produce more expressive sentence embeddings, often yielding stronger ranking performance.
* **DistilRoBERTa**: A compressed transformer model that trades some accuracy for reduced computational cost.

All models utilize the same retrieval paradigm (dense embeddings + cosine similarity) but differ in architecture and pretraining objectives.

### Evaluation Protocol

To evaluate semantic retrieval in a controlled and reproducible manner, we use the **SciFact** dataset.

**Why SciFact?**

* **Scientific Domain:** Composed of real scientific abstracts, aligning perfectly with our academic paper retrieval use case.
* **Ground-Truth Relevance:** Each query (claim) is paired with verified supporting/refuting documents, enabling objective automatic evaluation.
* **Closed-Corpus Setting:** Matches our system design where retrieval is performed over a specific set of user-provided papers.

#### Metrics

We report standard ranking-based retrieval metrics at cutoff :

* **nDCG@10:** Evaluates ranking quality by rewarding relevant documents appearing at higher ranks.
* **MAP@10:** Measures average precision across all queries, capturing ranking consistency.
* **Recall@10:** Indicates whether relevant documents are retrieved within the top 10 results.
* **Precision@10 (P@10):** Measures the proportion of relevant documents in the top 10. *Note: This is typically low in scientific retrieval as usually only 1-2 documents are relevant per query.*

### 📊 Results

| Model | nDCG@10 | MAP@10 | Recall@10 | P@10 |
| --- | --- | --- | --- | --- |
| **MPNet** | **0.65570** | **0.60827** | **0.79011** | **0.08933** |
| MiniLM | 0.64508 | 0.59593 | 0.78333 | 0.08833 |
| DistilRoBERTa | 0.63146 | 0.57580 | 0.78789 | 0.08900 |

> **Note on Metrics:** It is crucial to distinguish between classification and retrieval metrics. While classification F1-scores often exceed 0.95, retrieval tasks involve ranking a large set of candidates for ambiguous queries. Consequently, nDCG and MAP values in the range of **0.60–0.70** represent strong performance in scientific retrieval.

### ✅ Model Selection

Based on the evaluation results, we selected **MPNet** (`all-mpnet-base-v2`) as the default semantic retriever. It achieves the highest nDCG@10 and MAP@10, indicating superior ranking quality and robustness. MiniLM remains available as a fallback for resource-constrained environments.
---



Retrieval-Augmented Generation (RAG) Pipeline

### 🎯 Objective

The primary goal of this system is to provide evidence-grounded answers to research questions. By combining dense retrieval with a Large Language Model (LLM), we implement a pipeline that filters and ranks scientific documents to ensure generated answers are strictly derived from retrieved scientific evidence, rather than external world knowledge.

### 🏗️ Overall Architecture

The RAG pipeline reflects a closed-domain scientific QA scenario and consists of four main stages:

Document Chunking: Intelligent segmentation of text.

Semantic Retrieval: Using SBERT for dense vector ranking.

Prompt Construction: Evidence selection and strict constraint injection.

Answer Generation: Producing answers via a Large Language Model.

### ⚙️ Pipeline Components

1. Document Chunking Strategy

Scientific documents are typically long and structured around coherent argumentative units. Feeding entire documents to the retriever or LLM introduces semantic noise and computational inefficiency.

We employ a Sentence-Based Chunking Strategy. Unlike fixed-size word chunking, this approach preserves sentence boundaries to maintain the integrity of theoretical explanations.

FeatureConfigurationRationaleStrategySentence-basedPrevents fragmenting definitions or math explanations across chunks.Target Size~200 wordsBalances context window usage with semantic precision.OverlapNoneSemantic continuity is preserved via sentence grouping; redundancy is unnecessary here.

2. Semantic Retrieval with SBERT

Each text chunk is encoded into a dense vector using Sentence-BERT, and semantic relevance is computed using cosine similarity.

Model: all-mpnet-base-v2

Selected for strong performance on information retrieval benchmarks.

Captures sentence-level semantics, paraphrasing, and conceptual similarity (unlike TF-IDF).

Diversified Retrieval: We enforce a maximum number of retrieved chunks per document to ensure broader coverage and reduce redundancy in the evidence set.

3. Evidence-Based Prompt Construction

Retrieved chunks are injected into a structured prompt that strictly constrains the LLM behavior:

### ✅ Constraint: Use only the provided evidence.

### 🚫 Constraint: Avoid hallucination or reliance on prior external knowledge.

### 📝 Constraint: Explicitly cite sources using evidence identifiers.

Each evidence block passed to the model includes the Document ID, Chunk Index, and Truncated Text.

4. Answer Generation (LLM)

We utilize Mistral-7B-Instruct-v0.2 as the generative engine.

Why Mistral-7B?

Instruction Following: Strong adherence to complex grounding constraints.

Reasoning: Competitive capabilities for scientific text.

Transparency: Open-weight architecture enabling reproducible research.

Efficiency: Favorable trade-off between performance and cost.

Note: The system uses token streaming to generate answers incrementally, mirroring the interactive experience of an academic assistant.

### 🔍 Evidence Grounding & Explainability

Transparency is a core design principle. For every query, the system outputs:

The final generated answer.

The list of evidence chunks used.

Similarity scores for each retrieved chunk.

This allows users to trace every claim back to its specific source document, which is critical for scientific validity.

### 🆚 Relation to Prior Work

Unlike commercial systems such as NotebookLM or Semantic Scholar AI, our approach offers:

User Control: Operates on a fully user-controlled corpus.

Full Visibility: Exposes the complete retrieval and generation pipeline.

Research Platform: Enables systematic evaluation of individual components (Chunking vs. Retrieval vs. Generation).

### ⚠️ Current Limitations

Focus: The pipeline currently prioritizes correctness, grounding, and interpretability over aggressive speed optimization.

Evaluation: End-to-end quality is assessed separately using retrieval metrics and LLM-based judges.


