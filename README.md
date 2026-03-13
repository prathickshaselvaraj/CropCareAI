# 🌾 CropCareAI — AI-Powered Crop Recommendation & Care System

> An end-to-end machine learning system that recommends the optimal crop and predicts yield based on soil, weather, and regional conditions — deployed as a REST API using Flask.

---

## 📌 What This Project Does

Farmers and agronomists often rely on experience to decide which crop to grow. CropCareAI replaces guesswork with data — it takes soil nutrients, climate parameters, and geographic features as input, and outputs:

- ✅ The **best crop to grow** for the given conditions
- 📈 The **expected yield** prediction
- 🌡️ Actionable recommendations via a **real-time REST API**

---

## 🧠 ML Pipeline Overview

```
Raw Data (5,000+ records)
        ↓
  Data Preprocessing
  (handling nulls, encoding categoricals)
        ↓
  Feature Engineering
  (soil ratios, climate indices)
        ↓
  Class Imbalance Handling
  (SMOTE / resampling)
        ↓
  Model Training + GridSearchCV
  (RandomForest, DecisionTree, SVM)
        ↓
  Best Model: 84.5% Accuracy
        ↓
  Flask REST API → /predict endpoint
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| ML | Scikit-learn, Pandas, NumPy |
| API | Flask |
| Frontend | HTML, CSS (Jinja2 templates) |
| Deployment | Procfile (Heroku-compatible) |

---

## 📁 Project Structure

```
CropCareAI/
├── app.py                  # Flask app — API routes + UI rendering
├── src/                    # ML pipeline scripts
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── train.py            # Model training and serialization
│   └── predict.py          # Inference logic
├── data/                   # Dataset files
├── frontend/               # HTML/CSS templates
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment config
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/prathickshaselvaraj/CropCareAI.git
cd CropCareAI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Access the app
Open `http://localhost:5000` in your browser, or hit the API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202}'
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **84.5%** |
| Validation | 5-Fold Cross-Validation |
| Imbalance Handling | SMOTE (Synthetic Minority Oversampling) |
| Tuning | GridSearchCV |

---

## 🔑 Key Features

- **Handles class imbalance** — rare crops are not ignored; SMOTE generates synthetic samples so the model learns all classes equally
- **Hyperparameter tuning** — GridSearchCV systematically finds the best model configuration
- **REST API deployment** — any external app can call `/predict` with crop parameters and get a real-time recommendation
- **Structured ML pipeline** — clean separation of preprocessing, training, and inference code

---

## 🔮 Future Improvements

- [ ] Add image-based disease detection using CNN
- [ ] Integrate real-time weather API (OpenWeatherMap)
- [ ] Build a farmer-facing mobile interface
- [ ] Expand dataset to cover more regional crops

---

## 👩‍💻 Author

**Prathicksha S**  
M.Sc. Decision and Computing Sciences, CIT Coimbatore  
📧 prathicksha.selvaraj@gmail.com  
🔗 [GitHub](https://github.com/prathickshaselvaraj) · [LinkedIn](https://linkedin.com/in/prathickshaselvaraj)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
