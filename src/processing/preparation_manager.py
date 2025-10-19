# src/processing/preparation_manager.py
import pandas as pd
from src.schemas import TransformationRecipe, TransformationStep
from .recipe_executor import apply_recipe


class DataPreparationManager:
    """
    Encapsulates the state and logic for the data preparation workflow.
    This class is now completely decoupled from Streamlit.
    """

    def __init__(self, raw_df: pd.DataFrame, recipe: TransformationRecipe):
        """
        Initializes the manager with its dependencies: the raw data and the recipe object.

        Args:
            raw_df (pd.DataFrame): The original, unmodified dataframe.
            recipe (TransformationRecipe): The Pydantic object that holds the transformation steps.
        """
        self.raw_df = raw_df
        self.recipe = recipe

    def get_working_df(self) -> pd.DataFrame:
        """Applies the current recipe to the raw data."""
        return apply_recipe(self.raw_df, self.recipe)

    # --- The methods now modify the recipe object they were given ---

    def add_step(self, step: TransformationStep):
        self.recipe.steps.append(step)

    def remove_step(self, index: int):
        if 0 <= index < len(self.recipe.steps):
            self.recipe.steps.pop(index)

    def move_step(self, index: int, direction: str):
        if direction == "up" and index > 0:
            self.recipe.steps.insert(index - 1, self.recipe.steps.pop(index))
        elif direction == "down" and index < len(self.recipe.steps) - 1:
            self.recipe.steps.insert(index + 1, self.recipe.steps.pop(index))

    def reset_recipe(self):
        self.recipe.steps = []
