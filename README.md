# 🎓 UCLA Admission Predictor

This is a Streamlit-based web application that predicts a student's chance of admission to a university using a machine learning model. It uses a Multi-Layer Perceptron (MLP) neural network regressor trained on historical admission data.

---

## 📊 Dataset

**File**: `Admission.csv`  
**Source**: Provided as part of course materials

### 🔍 Features Used:
| Feature             | Description                                               |
|---------------------|-----------------------------------------------------------|
| `GRE Score`         | Graduate Record Exam score (out of 340)                   |
| `TOEFL Score`       | Test of English as a Foreign Language score (out of 120)  |
| `University Rating` | Subjective ranking from 1 to 5                            |
| `SOP`               | Statement of Purpose strength (1–5)                        |
| `LOR`               | Letter of Recommendation strength (1–5)                   |
| `CGPA`              | Undergraduate GPA (out of 10)                              |
| `Research`          | 1 if student has research experience, 0 otherwise          |

**Target Variable**: `Admit_Chance` (probability between 0 and 1)

---

## 🧠 Model

- **MLPRegressor** from `sklearn.neural_network`
- Scaled input features using `StandardScaler`
- Evaluated using:
  - R² Score
  - Mean Squared Error (MSE)

---

## 🚀 Features

- Load and preview the admission dataset
- Preprocess features and train a neural network regression model
- Visualize model performance with actual vs predicted scores
- Enter custom values to predict an individual student's admission chance

---
## 🌐 Live App

You can try the deployed Streamlit app here:  
🔗 [UCLA Admission Predictor](https://uclaadmissionpredictor-russel.streamlit.app/)


## 🛠 How to Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/UCLA_Admission_Predictor.git
cd UCLA_Admission_Predictor
