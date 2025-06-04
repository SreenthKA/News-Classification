# 📰 News Categorization using Traditional & Deep Learning NLP Models

This repository contains a complete pipeline for news article classification using various NLP models and embedding techniques. It explores classical feature representations like Word2Vec and TF-IDF, as well as deep learning models including fine-tuned BERT and BERT-BiLSTM. The project also incorporates evaluation using real-world datasets such as the 20 Newsgroups collection.

---

## 📂 Project Structure

```bash
├── max_length.ipynb             # Analysis of optimal sequence lengths for transformers
├── newsword2vec.ipynb           # Word2Vec embedding and classifier implementation
├── BERTfinetuning.ipynb         # Fine-tuning a pre-trained BERT model
├── BERT-BiLSTM.ipynb            # BERT embeddings with a BiLSTM classification head
├── implementation.ipynb         # Combined pipeline implementation
├── imp2.ipynb                   # Experimental trials and model performance comparison
├── news-article-categories.csv  # Main labeled dataset
├── 20_newsgroups/               # 20 text files representing raw Usenet newsgroups
└── requirements.txt             # Dependencies
````

---

## 📌 Objectives

* Evaluate different feature representations (TF-IDF, Word2Vec, BERT)
* Build and compare classifiers (SVM, Logistic Regression, BiLSTM, Fine-tuned BERT)
* Measure performance using accuracy, F1 score, and confusion matrices
* Understand real-world challenges using raw newsgroup data

---

##  Datasets

### 1. `news-article-categories.csv`
* Download it from https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories
* Contains labeled news headlines/articles
* Columns:

  * `Text` – raw news content
  * `Category` – assigned label for classification

### 2. `20 Newsgroups (Raw)`

Located in `/20_newsgroups/` (20 `.txt` files). Each file corresponds to a different Usenet newsgroup, e.g.:

* `alt.atheism.txt`
* `sci.space.txt`
* `soc.religion.christian.txt`
* `rec.sport.baseball.txt`

These files contain unstructured posts that can be used for:

* Topic modeling
* Multi-class text classification
* Pretraining or unsupervised clustering

---

##  How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Jupyter Notebooks

Each notebook is modular. Run in the following order for full pipeline:

1. `max_length.ipynb` – Token length analysis
2. `newsword2vec.ipynb` – Word2Vec + classifiers
3. `BERTfinetuning.ipynb` – Fine-tune base BERT
4. `BERT-BiLSTM.ipynb` – BERT embeddings + BiLSTM head
5. `implementation.ipynb` & `imp2.ipynb` – Integrated pipelines and final testing

---

## Results & Evaluation

Evaluation metrics include:

* Accuracy
* Precision, Recall, F1-score (macro & micro)
* Confusion matrices
* Token length distribution visualizations

Key findings are discussed within each notebook.

---

##  Models Used

| Embedding | Classifier               |
| --------- | ------------------------ |
| TF-IDF    | Logistic Regression      |
| Word2Vec  | SVM, Logistic Regression |
| BERT      | Fine-tuned classifier    |
| BERT      | BiLSTM                   |

---

##  Dependencies

```text
gensim
matplotlib
nltk
numpy
pandas
seaborn
sentence_transformers
sklearn
spacy
torch
tqdm
transformers
```

Install with:

```bash
pip install -r requirements.txt
```

