# 📘 Synthetic Career Suggestions Dataset

## 📊 Overview  
This dataset contains **640 career suggestion samples** designed to study and detect **gender bias** in career recommendations. Each entry consists of a text snippet (biased or neutral) along with metadata such as the associated skill, stereotype category, and gender reference. The dataset is structured in **pairs**, where each biased suggestion is matched with a neutral counterpart, enabling fine-grained comparisons.  

The goal of this dataset is to provide high-quality, labeled text examples that allow researchers and practitioners to train, evaluate, and benchmark models for **bias detection and mitigation** in natural language processing.  

---

## 📂 Dataset Structure  

| Column                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `id`                  | Unique identifier for each row                                              |
| `pair_id`             | Identifier linking biased and neutral text pairs                            |
| `stereotype_category` | One of three categories: Nurturing / Compassion, Leadership / Decisiveness, Technical / Analytical |
| `skill`               | The underlying skill the career suggestion is based on                      |
| `bias_type`           | Type of bias expressed (if applicable)                                      |
| `gender`              | Gender explicitly referenced in biased examples (`man` / `woman`)           |
| `text`                | The career suggestion text                                                  |
| `bias_label`          | Label (`1 = biased`, `0 = neutral`)                                         |

---

## ⚖️ Label Distribution  

- **Biased (1):** 320 samples  
- **Neutral (0):** 320 samples  

This ensures **perfect class balance** for training bias detection models.  

---

## 🧩 Category Distribution  

| Stereotype Category          | Count |
|-------------------------------|-------|
| Technical / Analytical        | 220   |
| Nurturing / Compassion        | 218   |
| Leadership / Decisiveness     | 202   |

The dataset evenly covers all three categories, preventing overrepresentation of any single stereotype.  

---

## 🚻 Gender Representation  

| Gender | Count |
|--------|-------|
| Man    | 160   |
| Woman  | 160   |

Both genders are equally represented, ensuring balanced stereotype coverage.  

---

## 📈 Text Length Statistics  

| Bias Label | Count | Mean Tokens | Std Dev | Min | 25% | 50% | 75% | Max |
|------------|-------|-------------|---------|-----|-----|-----|-----|-----|
| Neutral (0)| 320   | 47.3        | 5.6     | 34  | 43  | 47  | 51  | 64  |
| Biased (1) | 320   | 50.8        | 6.4     | 36  | 46  | 51  | 55  | 67  |

Texts fall within a natural range, with biased suggestions slightly longer on average due to explicit gender references.  

---

## ✅ Key Features  

- Balanced across **bias labels** (neutral vs biased).  
- Balanced across **genders** (male vs female).  
- Balanced across **stereotype categories** (nurturing, leadership, technical).  
- Designed as **paired data** (biased + neutral per skill).  
- Cleaned and deduplicated for training readiness.  

---

## 🚀 Potential Use Cases  

- **Bias Detection**: Training NLP models to classify biased vs neutral career suggestions.  
- **Fairness Benchmarking**: Evaluating bias mitigation methods in LLMs.  
- **Explainable AI**: Using interpretability tools (e.g., SHAP, LIME) to understand model reasoning on biased text.  
- **Data Augmentation**: Expanding with real-world LinkedIn or career counseling datasets for domain adaptation.  

---

## 📌 Citation  

If you use this dataset in your work, please cite it as:  

Synthetic Career Suggestions Dataset v1.0
Generated and curated by Dyuti Dasmahapatra, 2025