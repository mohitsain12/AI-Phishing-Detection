# 🛡️ AI Phishing Detection

An intelligent phishing URL detection system powered by Machine Learning. Analyzes 33 lexical features of any URL to predict whether it's a legitimate website or a phishing attempt — achieving **~99.5% accuracy** with a Random Forest classifier.

---

## 🚀 Features

- **33 URL Feature Extraction** — Analyzes URL length, domain structure, character patterns, entropy, suspicious keywords, and more
- **4 ML Models Compared** — Logistic Regression, KNN, Decision Tree, and Random Forest
- **Hyperparameter Tuning** — GridSearchCV optimization for the best model
- **CLI Prediction** — Predict phishing URLs from the command line
- **Streamlit Web App** — Premium dark-themed UI with real-time URL analysis

---

## 📁 Project Structure

```
AI-Phishing-Detection/
├── app/
│   └── app.py                    # Streamlit web application
├── data/
│   ├── phishing.csv              # Original dataset
│   ├── raw_urls.csv              # Preprocessed (URL + label only)
│   └── features.csv              # Extracted feature dataset
├── graphs/
│   ├── confusion_matrix.png      # Model confusion matrix
│   ├── feature_importance.png    # Feature importance chart
│   └── model_comparison.png      # Model accuracy comparison
├── models/
│   ├── logistic_model.pkl        # Logistic Regression
│   ├── knn_model.pkl             # K-Nearest Neighbors
│   ├── decision_tree_model.pkl   # Decision Tree
│   ├── random_forest_model.pkl   # Random Forest (best)
│   ├── best_model.pkl            # Copy of best model
│   └── phishing_feature_names.pkl# Feature names list
├── src/
│   ├── preprocess.py             # Step 1: Data preprocessing
│   ├── feature_extractor.py      # Step 2: URL feature extraction
│   ├── train_model.py            # Step 3: Model training & comparison
│   ├── tune_model.py             # Step 4: Hyperparameter tuning
│   ├── predict.py                # Step 5: URL prediction
│   └── main.py                   # Testing script
├── requirments.txt
└── README.md
```

---

## 🔄 Pipeline Workflow

```
phishing.csv  →  preprocess.py  →  raw_urls.csv  →  feature_extractor.py  →  features.csv
                                                                                    │
                                                                            train_model.py
                                                                                    │
                                                                    ┌───────────────┼───────────────┐
                                                              Logistic Reg       KNN        Decision Tree
                                                                    │               │               │
                                                                    └───────┬───────┘               │
                                                                            │                       │
                                                                     Random Forest ◄────────────────┘
                                                                      (Best Model)
                                                                            │
                                                                     tune_model.py
                                                                            │
                                                                     predict.py / app.py
```

---

## 🧠 Extracted Features (33)

| Category | Features |
|----------|----------|
| **Basic** | url_length, https, dots, many_dots, parameter_count, long_url, contains_email, starts_with_digit |
| **Domain** | domain_length, long_domain, subdomain_count, suspicious_tld, tld_length, domain_has_digits, ip_address, has_port |
| **Security** | hyphen, at_symbol, url_shortener, double_hyphen, multiple_special, repeated_chars, keyword_count, has_suspicious_extension, brand_count |
| **Statistics** | digits, special_characters, digit_ratio, special_character_ratio, slashes, question_marks, equal_signs, ampersands, underscores, entropy, directory_depth |

---

## ⚙️ Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirments.txt
```

### 2. Run the Full Pipeline (Optional — pre-built models included)

```bash
cd src

# Step 1 & 2: Preprocess data and extract features
python preprocess.py

# Step 3: Train and compare all 4 models
python train_model.py

# Step 4: Tune the best model (Random Forest)
python tune_model.py
```

### 3. Predict from CLI

```bash
cd src
python predict.py
```

### 4. Launch Web App

```bash
streamlit run app/app.py
```

---

## 📊 Model Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~88% |
| KNN | ~95% |
| Decision Tree | ~98% |
| **Random Forest** | **~99.5%** |

---

## 🛠️ Technologies

- **Python 3**
- **Pandas** — Data manipulation
- **Scikit-learn** — Machine Learning models
- **Matplotlib** — Visualization
- **Streamlit** — Web application
- **Joblib** — Model serialization

---

## 👤 Author

**Mohit Sain**

---

## 📄 License

This project is licensed under the MIT License.