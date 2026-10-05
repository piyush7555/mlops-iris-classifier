# Hyperparameter Tuning Analysis

## 1. Baseline Model

Model: DecisionTreeClassifier

CV F1 Macro: 0.9663
Test Accuracy: 0.9000

## 2. Grid Search

Model: RandomForestClassifier
Total combinations: 72
Cross-validation: 5-fold
Total fits: 360
Best CV F1 Macro: 0.9663
Test Accuracy: 0.9667

## 3. Random Search

Model: RandomForestClassifier
Number of iterations: 30
Cross-validation: 5-fold
Total fits: 150
Best CV F1 Macro: 0.9663
Test Accuracy: 0.9667

## 4. Comparison

Grid Search exhaustively evaluates all combinations in the defined hyperparameter space.
Random Search samples a fixed number of candidate configurations from the same space.
In this experiment, Grid Search required 360 model fits while Random Search required only 150.
The tuned Random Forest models achieved the same 5-fold CV F1 score as the baseline Decision Tree, but improved the held-out test accuracy to 0.9667. Random Search reached essentially the same result as Grid Search using less than half the total fits.
