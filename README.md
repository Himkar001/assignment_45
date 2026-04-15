# Week 08 – Tuesday Assignment  
---
# 📌 Overview
This assignment covers:
- Data Cleaning (Real-world hospital dataset)
- Neural Network from scratch (NumPy)
- Model Evaluation & Business Decision
- Handling class imbalance & feature learning

---

# 🟢 EASY PART

## Sub-step 1: Data Quality Audit

### Code
```python
df = pd.read_csv('hospital_records.csv')
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Data Quality Audit Write-up
- Missing values found in BMI and smoking_status  
- Invalid values: negative age, extreme BMI  
- Outliers present in BMI and glucose  
- Categorical inconsistencies  
- Dataset not ready for modeling  

---

## Sub-step 2: Data Cleaning

### Code
```python
df['bmi'].fillna(df['bmi'].median(), inplace=True)
df['smoking_status'].fillna(df['smoking_status'].mode()[0], inplace=True)
df = df[df['age'] >= 0]
df['bmi'] = df['bmi'].clip(10,50)
df['avg_glucose_level'] = df['avg_glucose_level'].clip(50,300)
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df = pd.get_dummies(df, columns=['smoking_status'], drop_first=True)
```

### Data Cleaning Decisions
- Median used for BMI (robust to outliers)  
- Mode used for categorical  
- Removed invalid ages  
- Capped outliers  
- Encoded categorical variables  

---

# 🟡 MEDIUM PART

## Sub-step 3: Neural Network

### Code (Core)
```python
def sigmoid(x):
    return 1/(1+np.exp(-x))
```

Neural network built with:
- Input layer
- Hidden layer (16 neurons)
- Output layer

---

## Sub-step 4: Training & Evaluation

### Output
- Loss decreases from ~0.69 → ~0.45  
- Model learns meaningful patterns  

### Evaluation
- Used F1-score instead of accuracy  
- Compared with Logistic Regression  

### Insights
- Accuracy alone is misleading  
- F1-score better for imbalanced data  

---

## Sub-step 5: Business Decision

### Cost Setup
- False Negative = 10  
- False Positive = 1  

### Insight
- Lower threshold improves recall  
- Best threshold chosen based on minimum cost  

### Final Recommendation
Prioritize recall to avoid missing high-risk patients.

---

# 🔴 HARD PART

## Sub-step 6: Misleading Accuracy

### Observation
- Accuracy ≈ 94%  

### Problem
- Model predicts majority class  
- Fails on minority (high-risk patients)  

### Fix
```python
LogisticRegression(class_weight='balanced')
```

### Insight
Accuracy is misleading in imbalanced datasets.

---

## Sub-step 7: Feature Extraction

### Approach
- Used hidden layer as embeddings  
- Trained classifier on embeddings  

### Result
- Improved F1-score  
- Better class separation  

### Insight
Neural networks can act as feature extractors.

---

# 📊 FINAL INSIGHTS

- Data quality directly impacts model performance  
- Neural networks can learn complex patterns  
- Class imbalance requires careful metric selection  
- Business decisions must consider cost, not just accuracy  

---

# 🚀 How to Run

1. Load dataset  
2. Run Easy → Medium → Hard sections sequentially  
3. Ensure NumPy, pandas, sklearn installed  

---

# 📦 Requirements
- Python 3.x  
- pandas  
- numpy  
- sklearn  
- matplotlib  

---

# ✅ Conclusion
This assignment demonstrates:
- End-to-end ML pipeline  
- Importance of data cleaning  
- Neural network fundamentals  
- Real-world decision making