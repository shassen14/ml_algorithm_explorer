# src/analysis/model_analyzer.py


def get_error_analysis(X_test, y_test, y_pred, y_pred_proba, class_names):
    """
    Identifies the most incorrect predictions (false positives and false negatives).
    """
    if y_test.nunique() != 2:
        return None, None  # This analysis is primarily for binary classification

    # Assuming positive class is the second one
    positive_class = class_names[1]
    negative_class = class_names[0]
    positive_class_encoded = 1

    # Combine data for easy filtering
    analysis_df = X_test.copy()
    analysis_df["true_label"] = y_test
    analysis_df["predicted_label"] = y_pred
    analysis_df["positive_class_prob"] = y_pred_proba[:, 1]

    # --- False Positives: Predicted Positive, but was Negative ---
    fp_mask = (analysis_df["predicted_label"] == positive_class_encoded) & (
        analysis_df["true_label"] != positive_class
    )
    false_positives = analysis_df[fp_mask]
    # Sort by the model's confidence in its wrong prediction
    false_positives = false_positives.sort_values(
        by="positive_class_prob", ascending=False
    )

    # --- False Negatives: Predicted Negative, but was Positive ---
    fn_mask = (analysis_df["predicted_label"] != positive_class_encoded) & (
        analysis_df["true_label"] == positive_class
    )
    false_negatives = analysis_df[fn_mask]
    # Sort by how confident the model was in its wrong "negative" prediction (low positive prob)
    false_negatives = false_negatives.sort_values(
        by="positive_class_prob", ascending=True
    )

    return false_positives.head(), false_negatives.head()
