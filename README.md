# 🛡️ AI Phishing Detection

An intelligent phishing URL detection system powered by machine learning. This project extracts 33 URL features from a website URL and predicts whether it is phishing or legitimate using a Random Forest model.

---

## 🚀 Features

- **33 URL Feature Extraction** — Analyzes URL length, domain structure, character patterns, entropy, suspicious keywords, and more
- **Multi-model comparison** — Supports Logistic Regression, KNN, Decision Tree, and Random Forest
- **Hyperparameter tuning** — Uses GridSearchCV for optimal model performance
- **CLI prediction mode** — Run predictions directly from the command line
- **Streamlit web application** — Interactive UI with prediction results, explanations, and history visualization

---

## 📁 Project Structure

```
AI-Phishing-Detection/
├── app/
│   ├── app.py                    # Streamlit web application entrypoint
│   ├── explainer.py              # Prediction explanation logic
│   ├── history.py                # Prediction history persistence
│   └── utils/                    # Streamlit UI helper modules
├── config.py                    # Application constants and paths
├── data/                        # Dataset and generated feature history
├── graphs/                      # Visual reports and analysis charts
├── models/                      # Trained model artifacts and feature mappings
├── scripts/                     # Utility scripts and automation helpers
├── src/                         # Model training, preprocessing, and prediction logic
├── tests/                       # Unit tests
├── requirments.txt              # Project dependencies
└── README.md                    # Project overview and usage
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

### 2. Run the Full Pipeline (Optional)

```bash
cd src

# Preprocess raw data and generate features
python preprocess.py

# Train and compare multiple models
python train_model.py

# Tune the best model with grid search
python tune_model.py
```

### 3. Predict from CLI

```bash
cd src
python predict.py
```

### 4. Launch the Streamlit Web App

```bash
streamlit run app/app.py
```

### Notes

- The `models/` directory should contain pre-trained model artifacts.
- The `data/` directory stores the dataset and prediction history.
- If you use the Streamlit app, open the browser URL shown after launch.

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