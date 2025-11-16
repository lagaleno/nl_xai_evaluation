# XAI Dataset Generation & Evaluation

This repository contains the full pipeline used to create, validate, and prepare a benchmark dataset of natural-language explanations for RAG (Retrieval-Augmented Generation) systems.  
The project includes:

- automatic extraction of HotpotQA data  
- generation of a dataset containing examples of three explanation types (*correct*, *incomplete*, *incorrect*) using an LLM  
- preliminary validation using sentence-wise embedding similarity  
- TODO: add description of experiment

---

## 📦 1. Prerequisites

- **Python 3.9+**  
- macOS or Linux (recommended)  
- `pip` or `pip3` installed  
- (Optional) A local LLaMA runtime (Ollama / llama.cpp)

---

## 🚀 2. Installation

All dependencies are listed in `requirements.txt`.

Simply run:

```bash
./install.sh
```

The script automatically detects whether your system uses  
`pip3` (macOS) or `pip` (Linux/WSL) and installs all dependencies accordingly.

If you prefer manual installation:

```bash
pip3 install -r requirements.txt
```

---

## 📁 3. Project Structure

```
project_root/
│
├── 0-utils/
│   ├── get_hotpotqa.py        # Downloads HotpotQA and converts to CSV
│   └── hotpotqa_train.csv/    # CSV containing hotpotqa dataset
│
├── 1-creating_dataset/
│   └── create_dataset.py     # Generates 3 explanation types per Q/A
│
├── 2-validating_dataset/
│   ├── validate_dataset.py       # Sentence-wise cosine similarity validation
│   ├── figures/                  # Evaluation plots (precision, recall, F1)
│   └── metrics/                  # Evaluation metrics (precision, recall, F1)
│
├── 3-experiments/
│   ├── upcoming_metrics/            # Placeholder for KG, Jaccard, FOL experiments
│   └── ...
│
├── requirements.txt
├── install.sh
└── README.md
```

---

## 📝 4. Step-by-Step Usage

### **Step 1 — Prepare HotpotQA**

Download and convert the dataset:

```bash
python 0-utils/prepare_hotpot_csv.py
```

Output:

```
0-utils/hotpotqa_train.csv
```

---

### **Step 2 — Generate the Explanation Dataset**

```bash
python 1-creating_dataset/generate_explanations.py
```

Output:

```
1-creating_dataset/explainrag_hotpot_llama.jsonl
```

---

### **Step 3 — Validate the Dataset (Sanity Check)**

```bash
python 2-validating_dataset/evaluate_embeddings.py
```

Outputs:

- explanations_sentencewise_metrics.csv  
- explanations_summary_by_label.csv  
- f1_by_label_boxplot.png  
- precision_by_label_boxplot.png  
- recall_by_label_boxplot.png  

---

### **Step 4 — Experiments **
TODO: Finish experiments

---

## ❗ Troubleshooting

### Pip issues  
```bash
pip3 install -r requirements.txt
```

### Matplotlib errors on macOS

```bash
brew install freetype pkg-config libpng
```

### LLaMA inference errors  
Ensure your local model server is running.

---

## 💬 Contact

For questions or contributions, please open an issue in the repository.
