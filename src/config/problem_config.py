# src/config/problem_config.py

# ==============================================================================
# 1. IMPORT BACKEND LOGIC & MODELS
# ==============================================================================

# Import the pipeline functions that orchestrate the ML process
from src.pipelines import classification_pipeline, regression_pipeline

# Import the actual model classes from scikit-learn and xgboost
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# ==============================================================================
# 2. DEFINE THE MASTER CONFIGURATION DICTIONARY
# ==============================================================================

PROBLEM_CONFIG = {
    "Classification": {
        "display_name": "Classification",
        "icon": "🎯",
        "pipeline": classification_pipeline.run_classification_pipeline,
        "metrics_display_name": "Classification Metrics",
        "supported_plots": ["Confusion Matrix", "ROC Curve", "Feature Importance"],
        "models": {
            "Logistic Regression": LogisticRegression,
            "K-Nearest Neighbors": KNeighborsClassifier,
            "Random Forest": RandomForestClassifier,
            "XGBoost": XGBClassifier,
        },
        "hyperparameters": {
            "Logistic Regression": {
                "C": {
                    "widget": "slider",
                    "label": "Regularization (C)",
                    "min_value": 0.01,
                    "max_value": 10.0,
                    "value": 1.0,
                    "step": 0.01,
                    "help": "Inverse of regularization strength. Smaller values specify stronger regularization.",
                },
                "solver": {
                    "widget": "selectbox",
                    "label": "Solver",
                    "options": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
                    "value": "liblinear",
                },
            },
            "K-Nearest Neighbors": {
                "n_neighbors": {
                    "widget": "slider",
                    "label": "Number of Neighbors (k)",
                    "min_value": 1,
                    "max_value": 30,
                    "value": 5,
                    "step": 1,
                }
            },
            "Random Forest": {
                "n_estimators": {
                    "widget": "slider",
                    "label": "Number of Trees",
                    "min_value": 10,
                    "max_value": 500,
                    "value": 100,
                    "step": 10,
                },
                "max_depth": {
                    "widget": "number_input",
                    "label": "Max Depth",
                    "min_value": 1,
                    "max_value": 50,
                    "value": 10,
                    "step": 1,
                    "help": "Enter 0 for no limit.",
                },
                "class_weight": {
                    "widget": "selectbox",
                    "label": "Class Weight",
                    "options": [None, "balanced"],  # Allow user to choose
                    "index": 1,  # Default to 'balanced'
                    "help": "Set to 'balanced' to automatically adjust weights inversely proportional to class frequencies. Crucial for imbalanced datasets.",
                },
            },
            "XGBoost": {
                "n_estimators": {
                    "widget": "slider",
                    "label": "Number of Trees",
                    "min_value": 10,
                    "max_value": 500,
                    "value": 100,
                    "step": 10,
                },
                "learning_rate": {
                    "widget": "slider",
                    "label": "Learning Rate",
                    "min_value": 0.01,
                    "max_value": 0.5,
                    "value": 0.1,
                    "step": 0.01,
                },
                "max_depth": {
                    "widget": "slider",
                    "label": "Max Depth",
                    "min_value": 1,
                    "max_value": 15,
                    "value": 3,
                    "step": 1,
                },
            },
        },
    },
    # "Regression": {
    #     "display_name": "Regression",
    #     "icon": "📈",
    #     "pipeline": regression_pipeline.run_regression_pipeline,
    #     "metrics_display_name": "Regression Metrics",
    #     "supported_plots": ["Actual vs. Predicted", "Feature Importance"],
    #     "models": {
    #         "Linear Regression": LinearRegression,
    #         "Ridge Regression": Ridge,
    #         "Random Forest Regressor": RandomForestRegressor,
    #         "XGBoost Regressor": XGBRegressor,
    #     },
    #     "hyperparameters": {
    #         "Ridge Regression": {
    #             "alpha": {
    #                 "widget": "slider",
    #                 "label": "Regularization (alpha)",
    #                 "min_value": 0.01,
    #                 "max_value": 10.0,
    #                 "value": 1.0,
    #                 "step": 0.01,
    #                 "help": "Regularization strength. Larger values specify stronger regularization.",
    #             }
    #         },
    #         "Random Forest Regressor": {
    #             "n_estimators": {
    #                 "widget": "slider",
    #                 "label": "Number of Trees",
    #                 "min_value": 10,
    #                 "max_value": 500,
    #                 "value": 100,
    #                 "step": 10,
    #             },
    #             "max_depth": {
    #                 "widget": "number_input",
    #                 "label": "Max Depth",
    #                 "min_value": 1,
    #                 "max_value": 50,
    #                 "value": 10,
    #                 "step": 1,
    #                 "help": "Enter 0 for no limit.",
    #             },
    #         },
    #         "XGBoost Regressor": {
    #             "n_estimators": {
    #                 "widget": "slider",
    #                 "label": "Number of Trees",
    #                 "min_value": 10,
    #                 "max_value": 500,
    #                 "value": 100,
    #                 "step": 10,
    #             },
    #             "learning_rate": {
    #                 "widget": "slider",
    #                 "label": "Learning Rate",
    #                 "min_value": 0.01,
    #                 "max_value": 0.5,
    #                 "value": 0.1,
    #                 "step": 0.01,
    #             },
    #             "max_depth": {
    #                 "widget": "slider",
    #                 "label": "Max Depth",
    #                 "min_value": 1,
    #                 "max_value": 15,
    #                 "value": 3,
    #                 "step": 1,
    #             },
    #         },
    #     },
    # },
    # "Clustering": { ... } # A placeholder for when you're ready to add it
}
