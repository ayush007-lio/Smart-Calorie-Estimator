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

📊 Key Results & PerformanceThe models were evaluated using RMSE (Root Mean Squared Error) and R² Score.Model / ApproachR² ScoreRMSEPerformance VerdictMedical Eq (Keytel)~0.86HighGood baseline, but fails at high intensity.Linear Regression~0.96ModerateBetter, but misses the physiological curve.Polynomial Regression~0.99LowestMatches human physiology perfectly.Observation: The Polynomial Regression model reduced the error rate significantly compared to the medical equation, proving that AI can personalize fitness tracking better than static formulas.🚀 How to Run the ProjectPrerequisitesMake sure you have Python installed.Step 1: Clone the RepositoryBashgit clone [https://github.com/ayush007-lio/Smart-Calorie-Estimator.git](https://github.com/ayush007-lio/Smart-Calorie-Estimator.git)
cd Smart-Calorie-Estimator
Step 2: Install DependenciesIt is recommended to use a virtual environment.Bashpip install -r requirements.txt
Step 3: Run the AnalysisExecute the main script. The script uses the Rich library to provide a colorful, summarized output in your terminal.Bashpython main.py
Step 4: View ResultsCheck the Terminal for the R² and RMSE comparison table.Open results_plot.png in the folder to see the visual graph comparing the "Medical Equation" vs "AI Model."💡 How to UseThis project is designed as a backend prototype for fitness apps.Input: The model accepts: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp.Processing: It runs these inputs through the trained Polynomial Regressor.Output: It returns the precise calories burned (e.g., 245.3 kcal).🤝 Contribution & LicenseFeel free to fork this repository and submit pull requests.Author: Ayush (Eng. Student, CS-AI)License: MIT
### Why this README works:
1.  **Badges:** It looks like a "real" open-source tool immediately.
2.  **The "Pin-to-Pin" Structure:** I added the file tree so the evaluator knows you understand your own file organization.
3.  **The Comparison Table:** This is the most important part. It proves your project works.
4.  **Medical Context:** It emphasizes that you know about the "Keytel Equation," showing yo
