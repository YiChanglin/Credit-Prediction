# Credit-Prediction

This Jupyter Notebook, "Predictive Modeling for Credit Risk.ipynb," presents a comprehensive machine learning workflow for predicting serious financial delinquency.

**Key Features:**

* **Data Preprocessing:** Handles missing values through median imputation to ensure data quality and model robustness.
* **Exploratory Data Analysis (EDA):** Includes correlation analysis to understand feature relationships, noting the robustness of tree-based models to multicollinearity.
* **Comparative Model Analysis:** Evaluates a diverse set of classifiers, including `AdaBoostClassifier`, `HistGradientBoostingClassifier`, `ExtraTreesClassifier`, `RandomForestClassifier`, `LogisticRegression`, `GaussianNB`, `BernoulliNB`, `MultinomialNB`, `DecisionTreeClassifier`, `XGBClassifier`, and `LGBMClassifier`.
* **Class Imbalance Handling:** Investigates the impact of Synthetic Minority Over-sampling Technique (SMOTE) on model performance, revealing that for this dataset, gradient boosting models perform well even without oversampling.
* **Hyperparameter Optimization:** Utilizes `BayesSearchCV` for efficient hyperparameter tuning of the best-performing model (LGBMClassifier), optimizing for ROC AUC.
* **Model Interpretability:** Employs SHAP (SHapley Additive exPlanations) to provide insights into feature importance and their impact on model predictions, crucial for transparent credit risk assessment.
* **Prediction Export:** Generates and exports predictions on a hold-out dataset to a CSV file.

**Usage:**

1.  Ensure all necessary libraries are installed (`pandas`, `numpy`, `scikit-learn`, `shap`, `scikit-optimize`, `imblearn`, `xgboost`, `lightgbm`).
2.  Place your `Credit_data.zip` file in the specified `DATA_PATH` or update the path accordingly.
3.  Run the notebook cells sequentially to execute the workflow.

This script provides a solid foundation for credit risk prediction, emphasizing robust methodology and model interpretability.
