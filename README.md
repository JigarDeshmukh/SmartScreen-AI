# 🚀 HireSense AI — ML-Powered Resume Screening System

> Stop keyword-based hiring. Start semantic ranking.

---

## 🔍 Overview

Most resume screening systems rely on keyword matching.

This leads to:

* ❌ Strong candidates getting rejected due to wording differences
* ❌ Poor resume–JD alignment detection
* ❌ Rigid filtering that ignores borderline candidates

**HireSense AI** solves this using semantic understanding and ML-based ranking.

---

## 🎯 What This Project Does

* Parses resumes (PDF → structured data)
* Extracts skills using a normalized taxonomy
* Computes semantic similarity (SBERT) between resume & job description
* Matches skills using both keyword + semantic understanding
* Analyzes career trajectory (growth, stability, role progression)
* Detects domain alignment (Data, DevOps, Product, etc.)
* Ranks candidates (instead of simple shortlist/reject)

---

## 🧠 Key Idea

Instead of hard filters → everything becomes a feature.

The system evaluates:

* Skill match
* Semantic similarity
* Experience quality
* Career progression

Then ranks candidates based on overall relevance.

---

## 🏗️ Architecture

Resume (PDF)
↓
Text Extraction
↓
Skill + Feature Extraction
↓
Semantic Matching (SBERT)
↓
ML Model (Gradient Boosting)
↓
Ranking Engine
↓
Output (Top / Borderline / Low)

---

## 📊 Model & Evaluation

* Gradient Boosting Classifier
* Cross-validation (Precision, Recall, F1)
* Realistic class imbalance (~15% shortlist rate)
* Overfitting detection (train vs validation gap)

---

## 📈 Output

Candidates are categorized into:

* 🟢 Top Shortlist
* 🟡 Borderline
* 🔴 Low Relevance

Each candidate includes:

* Ranking score
* Match breakdown
* Explanation (strengths + concerns)

---

## 🖼️ Demo

### Input:

* Resume (PDF)
* Job Description

### Output:

* Ranked candidates with explanations

📌 
<img width="770" height="113" alt="image" src="https://github.com/user-attachments/assets/a0d29fd0-1369-48bf-b5b5-6947aecc9794" />![Uploading image.png…]()



---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Sentence Transformers (SBERT)
* Pandas / NumPy
* OpenPyXL

---

## 🚀 How to Run

### 1. Clone repo

git clone https://github.com/yourusername/hiresense-ai.git
cd hiresense-ai

### 2. Install dependencies

pip install -r requirements.txt

### 3. Train model

python HR_RESUME_PIPELINE_V2.py --train

### 4. Run screening

python HR_RESUME_PIPELINE_V2.py --screen

---

## 📁 Project Structure

HireSense-AI/
│
├── data/
├── models/
├── reports/
├── resume test/   (Add Resume in this of ur own)
├── assets/
│   ├── output.png
│   ├── architecture.png
│
├── HR_RESUME_PIPELINE_V2.py
├── requirements.txt
├── README.md

---

## ⚠️ Limitations

* Trained on synthetic data (next: real hiring data)
* Resume parsing may fail on inconsistent formats
* Domain detection partially keyword-based

---

## 🔮 Future Improvements

* Train on real-world hiring data
* Improve parsing using layout-aware models
* Replace keyword domain detection with embeddings
* Build web interface / API

---

## 💡 Key Learnings

* Feature engineering > model complexity
* Keyword matching alone is unreliable
* Semantic understanding improves hiring decisions
* Evaluation must reflect real-world imbalance

---

## 🤝 Contributing

Open to feedback and improvements.
If you're working in HR Tech or ML, let’s connect.

---

## ⭐ If you found this useful, consider giving it a star!
