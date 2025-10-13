# src/evaluation/specific_plots/regression.py
import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Generates a scatter plot of actual vs. predicted values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_pred, alpha=0.5)

    # Add a diagonal line representing a perfect prediction
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Perfect Prediction")

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs. Predicted Values")
    ax.legend()
    ax.grid(True)

    return fig


def plot_residuals(y_test, y_pred):
    """
    Generates a scatter plot of residuals vs. predicted values.
    """
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_pred, residuals, alpha=0.5)

    # Add a horizontal line at y=0, where residuals should be centered
    ax.axhline(y=0, color="r", linestyle="--", label="Zero Error")

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residuals vs. Predicted Values")
    ax.legend()
    ax.grid(True)

    return fig
