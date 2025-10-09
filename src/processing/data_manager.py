# src/processing/data_manager.py
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# This sets up a basic logger that prints messages to your console.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get a logger instance for this specific module
logger = logging.getLogger(__name__)


def split_data(df, target_column, test_size):
    """Splits a DataFrame into train/test sets."""
    try:
        logger.info(
            f"Starting data split. Target column: '{target_column}', Test size: {test_size}"
        )
        logger.info(f"Original DataFrame shape: {df.shape}")

        # Sanity check
        if target_column not in df.columns:
            logger.error(
                f"Target column '{target_column}' not found in DataFrame columns: {df.columns.tolist()}"
            )
            return None, None, None, None

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(f"Features (X) shape: {X.shape}")
        logger.info(f"Target (y) shape: {y.shape}")

        # Check if stratification is possible (at least 2 members of each class)
        if y.nunique() > 1 and all(y.value_counts() > 1):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info("Successfully performed stratified split.")
        else:
            # Fallback to a non-stratified split if stratification is not possible
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logger.warning(
                "Could not perform stratified split (target might have only one class or single-member classes). Using regular split."
            )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        # This will log the full error traceback to the console
        logger.error(
            f"An unexpected error occurred during data splitting: {e}", exc_info=True
        )
        return None, None, None, None
