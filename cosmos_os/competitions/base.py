from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from cosmos_os.experiment import ExperimentManager

LOGGER = logging.getLogger(__name__)


class BaseCompetitionAdapter(ABC):
    """Abstract Base Class for Competition Adapters.
    
    This class defines the standard pipeline (Template Method) for solving
    machine learning competitions using Cosmos OS.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the adapter.
        
        Args:
            config (dict | None): Optional configuration dictionary.
        """
        self.config = config or {}
        self.experiment_manager = ExperimentManager(
            root_dir=self.config.get("experiment", {}).get("root_dir", "experiments"),
            reports_dir=self.config.get("experiment", {}).get("reports_dir", "reports"),
        )
        self.train_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.sample_submission: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.test_features: pd.DataFrame | None = None
        self.predictions: pd.DataFrame | None = None
        self.metrics: dict[str, float] = {}

    @abstractmethod
    def load_dataset(self) -> None:
        """Load train, test, and sample_submission datasets."""
        pass

    @abstractmethod
    def validate_schema(self) -> None:
        """Validate the loaded datasets (columns, dtypes, missing values)."""
        pass

    @abstractmethod
    def run_eda(self) -> None:
        """Run Exploratory Data Analysis and generate a report."""
        pass

    @abstractmethod
    def prepare_features(self) -> None:
        """Apply feature engineering to the datasets."""
        pass

    @abstractmethod
    def build_validation(self) -> None:
        """Build the validation strategy (e.g., temporal split)."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the model and compute validation metrics."""
        pass

    @abstractmethod
    def predict(self) -> None:
        """Generate predictions for the test dataset."""
        pass

    @abstractmethod
    def create_submission(self) -> None:
        """Create the final submission file mapping predictions to sample_submission."""
        pass

    def run(self) -> None:
        """Execute the entire competition pipeline sequentially."""
        LOGGER.info("Starting Competition Pipeline...")
        
        LOGGER.info("[1/8] Loading Dataset...")
        self.load_dataset()
        
        LOGGER.info("[2/8] Validating Schema...")
        self.validate_schema()
        
        LOGGER.info("[3/8] Running EDA...")
        self.run_eda()
        
        LOGGER.info("[4/8] Preparing Features...")
        self.prepare_features()
        
        LOGGER.info("[5/8] Building Validation...")
        self.build_validation()
        
        LOGGER.info("[6/8] Training Model...")
        self.train()
        
        LOGGER.info("[7/8] Generating Predictions...")
        self.predict()
        
        LOGGER.info("[8/8] Creating Submission...")
        self.create_submission()
        
        LOGGER.info("Competition Pipeline Completed Successfully.")
