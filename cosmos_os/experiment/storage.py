from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from .models import ExperimentRecord

LOGGER = logging.getLogger(__name__)
_VALID_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class ExperimentStorage:
    """Manages file-system storage and in-memory caching for experiments.
    
    This class handles saving and loading metadata JSON files, as well as 
    generating the leaderboard CSV. It uses an in-memory cache to prevent 
    O(N) file I/O operations on multiple accesses.
    """
    
    def __init__(self, root: str | Path = "experiments") -> None:
        """Initialize the storage manager.
        
        Args:
            root (str | Path): Root directory for saving experiments.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ExperimentRecord] = {}
        self._cache_valid: bool = False

    def _validate_id(self, experiment_id: str) -> None:
        if not _VALID_ID_PATTERN.match(experiment_id):
            raise ValueError(f"Invalid experiment_id format: {experiment_id}")

    def experiment_dir(self, experiment_id: str) -> Path:
        """Get the directory path for a specific experiment.
        
        Args:
            experiment_id (str): The unique ID of the experiment.
            
        Returns:
            Path: The directory path for the experiment.
            
        Raises:
            ValueError: If experiment_id contains invalid characters.
        """
        self._validate_id(experiment_id)
        return self.root / experiment_id

    def metadata_path(self, experiment_id: str) -> Path:
        """Get the file path for an experiment's metadata.json.
        
        Args:
            experiment_id (str): The unique ID of the experiment.
            
        Returns:
            Path: The file path to metadata.json.
        """
        return self.experiment_dir(experiment_id) / "metadata.json"

    def save(self, record: ExperimentRecord) -> Path:
        """Save an ExperimentRecord to disk and update the cache.
        
        Args:
            record (ExperimentRecord): The experiment record to save.
            
        Returns:
            Path: The file path where metadata.json was saved.
        """
        exp_dir = self.experiment_dir(record.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        path = self.metadata_path(record.experiment_id)
        payload = record.to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        
        self._cache[record.experiment_id] = record
        return path

    def load(self, experiment_id: str) -> ExperimentRecord:
        """Load an ExperimentRecord by ID, using cache if available.
        
        Args:
            experiment_id (str): The unique ID of the experiment.
            
        Returns:
            ExperimentRecord: The loaded record.
        """
        self._validate_id(experiment_id)
        if experiment_id in self._cache:
            return self._cache[experiment_id]
            
        path = self.metadata_path(experiment_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = ExperimentRecord.from_dict(payload)
        self._cache[experiment_id] = record
        return record

    def all_records(self) -> list[ExperimentRecord]:
        """Retrieve all experiment records. Uses in-memory cache if valid.
        
        Returns:
            list[ExperimentRecord]: A list of all recorded experiments.
        """
        if self._cache_valid:
            return list(self._cache.values())
            
        records: list[ExperimentRecord] = []
        if not self.root.exists():
            return records
            
        self._cache.clear()
        for exp_dir in sorted(self.root.iterdir()):
            if not exp_dir.is_dir():
                continue
            path = exp_dir / "metadata.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rec = ExperimentRecord.from_dict(payload)
                records.append(rec)
                self._cache[rec.experiment_id] = rec
            except Exception as exc:
                LOGGER.warning("metadata 읽기 실패: %s (%s)", str(path), exc)
                
        self._cache_valid = True
        return records

    def save_leaderboard(self, rows: list[dict], filename: str = "leaderboard.csv") -> Path:
        """Save a leaderboard in CSV format.
        
        Args:
            rows (list[dict]): A list of dictionary objects representing the rows.
            filename (str): The output filename. Defaults to "leaderboard.csv".
            
        Returns:
            Path: The file path to the saved CSV leaderboard.
        """
        path = self.root / filename
        cols = ["Experiment", "Metric", "Date", "Model", "Preset", "Rank"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path
