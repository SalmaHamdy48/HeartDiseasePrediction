🫀 Heart Disease Prediction using Manual Linear Regression

A complete Machine Learning pipeline built from scratch (without sklearn) to predict heart disease presence based on clinical features.


---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Attributes**: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, etc.
- **Target**: Presence of heart disease (0 = No, 1 = Yes)

---

## 🔍 Objectives

- Clean and preprocess the dataset
- Select the best predictor variable
- Implement six linear regression models manually (from scratch)
- Tune hyperparameters: learning rate & number of epochs
- Evaluate each model using **Mean Squared Error (MSE)**
- Plot:
  - MSE vs Learning Rate
  - MSE vs Epochs
- Write a conclusion based on the results

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🧠 Models Implemented

All models were implemented manually (not using sklearn):

1. Model 1: Predictor - `age`
2. Model 2: Predictor - `chol`
3. Model 3: Predictor - `thalach`
4. Model 4: Predictor - `oldpeak`
5. Model 5: Predictor - `cp`
6. Model 6: Predictor - `trestbps`

Each model was evaluated with multiple learning rates and epoch values.

---

## 📉 Visualizations

- ✅ MSE vs Learning Rate (six models in one plot or subplot)
- ✅ MSE vs Epochs (six models)
- ✅ Correlation heatmap
- ✅ Target variable distribution
- ✅ Feature histograms

---

## 📄 Report (report.pdf)

Includes:

- Student name, ID, department, UMS level
- Table summarizing MSEs
- All plots
- Final conclusion on the best predictor

---

## ✅ Results & Conclusion

- The best predictor variable was: **`thalach`** (or your actual result)
- The optimal learning rate: **e.g., 0.01**
- The optimal number of epochs: **e.g., 1000**
- MSE was lowest for this configuration compared to others

---

👩‍💻 Author
Salma Hamdy Ahmed
Faculty of Computers & Information – Ain Shams University
Email: salmahamdy1066@gmail.com

📌 Notes

This project was completed as part of a Machine Learning course assignment

All code was implemented manually without using sklearn for training

The report and script were submitted via the required Google Form


