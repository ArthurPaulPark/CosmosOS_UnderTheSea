"""모달리티 플러그인 — 데이터 종류별 피처 추출과 누수 단위 선언."""
from .anomaly import AnomalyFeaturePlugin, PRESETS as ANOMALY_PRESETS
from .base import FeatureSpec, ModalityPlugin, detect_leakage_unit_columns

__all__ = ["ModalityPlugin", "FeatureSpec", "detect_leakage_unit_columns",
           "AnomalyFeaturePlugin", "ANOMALY_PRESETS"]
