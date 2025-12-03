# 🏃‍♂️ Smart Calorie Estimator: AI vs. Medical Science

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Project Overview
This project is an advanced regression analysis aimed at predicting **calories burned** during exercise. Unlike standard fitness trackers that rely on static mathematical formulas (like the Keytel or Harris-Benedict equations), this project utilizes **Machine Learning (Polynomial Regression)** to learn the non-linear physiological patterns of the human body.

The core objective is to **benchmark AI models against established medical standards** to prove that Machine Learning provides significantly higher accuracy in physiological estimations.

---

## 🧠 The Problem Statement
Standard calorie calculators often fail because they assume a **linear relationship** between heart rate and energy expenditure. However, human physiology is complex:
* As heart rate increases, energy burn does not increase in a perfectly straight line.
* Factors like body temperature and weight have multicollinear relationships.

**The Solution:** By implementing **Polynomial Regression (Degree 2)** and **Ridge Regression**, this model captures these complex curves and interactions, outperforming traditional medical equations.

---

## ⚙️ Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear, Ridge, Polynomial Regression)
* **Visualization:** Matplotlib, Seaborn
* **UI/Terminal:** Rich (for beautiful CLI output)

---

## 📂 Repository Structure
Here is the "pin-to-pin" description of every file in this project:

```text
Smart-Calorie-Estimator/
│
├── data/
│   ├── calories.csv       # Target variable data (Calories burned)
│   └── exercise.csv       # Feature data (Heart rate, duration, temp, etc.)
│
├── main.py                # 🚀 The BRAIN of the project. Contains:
│                          #    - Data Preprocessing & Merging
│                          #    - Medical Equation Logic (Keytel Formula)
│                          #    - Model Training (Linear, Ridge, Poly)
│                          #    - Evaluation & Plotting
│
├── requirements.txt       # List of libraries needed to run the project
├── README.md              # Project documentation (You are reading this)
└── results_plot.png       # Generated comparison graph (AI vs Medical)

## 📊 Key Results & Performance

The models were evaluated using **RMSE (Root Mean Squared Error)** and **R² Score**.

| Model / Approach | R² Score | RMSE | Performance Verdict |
| :--- | :--- | :--- | :--- |
| **Medical Eq (Keytel)** | ~0.86 | High | Good baseline, but fails at high intensity. |
| **Linear Regression** | ~0.96 | Moderate | Better, but misses the physiological curve. |
| **Polynomial Regression** | **~0.99** | **Lowest** | **Matches human physiology perfectly.** |

> **Observation:** The Polynomial Regression model reduced the error rate significantly compared to the medical equation, proving that AI can personalize fitness tracking better than static formulas.

---

## 🚀 How to Run the Project

### Prerequisites
Make sure you have **Python** installed.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/ayush007-lio/Smart-Calorie-Estimator.git](https://github.com/ayush007-lio/Smart-Calorie-Estimator.git)
cd Smart-Calorie-Estimator
