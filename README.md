# Calorie Prediction Project

This project compares a medical formula (Keytel) with several regression models to predict calories burned during exercise. It produces a colorful terminal summary and a polished comparison plot.

## Requirements

Install dependencies (recommended to use the included venv or create one):

```powershell
python -m pip install -r requirements.txt
```

## Run

From the project root:

```powershell
C:/Users/LENOVO/OneDrive/Desktop/Calorie_Prediction_Project/venv/Scripts/python.exe main.py
```

Or, if using your system python:

```powershell
python main.py
```

The script will print a colorful model comparison and save `results_plot.png` in the project folder. Open that image to show the visual comparison to your evaluator.

## Notes for evaluator

- The dataset files `calories.csv` and `exercise.csv` should be in the project root.
- `rich` is used for terminal styling; plots are saved as PNG for presentation.

Good luck impressing your evaluator! ✨# Smart-Calorie-Estimator
