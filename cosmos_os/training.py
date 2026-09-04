"""LightGBM 학습 — 도메인 무관.

이 프로젝트에서 실측으로 확정된 기본 하이퍼파라미터(TUNED_A)를 기본값으로 쓴다.
2026-07-14 Optuna 재탐색(13 트라이얼)에서도 이 조합을 이기는 설정을 찾지 못했으므로,
새 데이터셋에서도 일단 이 값으로 시작하고 필요할 때만 재탐색하기를 권한다.

지원 기능:
  - 클래스 불균형 자동 보정(balanced sample weight)
  - 신뢰도 낮은 행 down-weight(conflict down-weight) — 라벨 노이즈가 있는 데이터에서 유효
  - OOF(out-of-fold) 확률 생성 — 스태킹용
  - 시드 배깅
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

# 이 프로젝트에서 실측 확정된 기본값. Optuna 재탐색으로도 개선 실패(2026-07-14).
TUNED_A: Dict = dict(
    objective="multiclass", boosting_type="gbdt", learning_rate=0.03,
    num_leaves=63, max_depth=-1, feature_fraction=0.6, bagging_fraction=0.8,
    bagging_freq=1, min_child_samples=40, lambda_l1=0.0, lambda_l2=1.0,
    n_jobs=-1, verbosity=-1, seed=42,
)

REGRESSION_PARAMS: Dict = dict(
    objective="regression", metric="rmse", boosting_type="gbdt", learning_rate=0.03,
    num_leaves=63, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, min_child_samples=20, lambda_l1=0.0, lambda_l2=1.0,
    n_jobs=-1, verbosity=-1,
)


@dataclass
class TrainedModel:
    boosters: List
    labels: List[str]
    feature_cols: List[str]
    n_rounds: int

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """시드 배깅된 확률 예측(부스터가 1개면 그대로)."""
        missing = [c for c in self.feature_cols if c not in x.columns]
        if missing:
            raise ValueError(f"학습 때 쓰던 피처가 없다: {missing[:5]}")
        x = x.loc[:, self.feature_cols]
        proba = None
        for b in self.boosters:
            p = b.predict(x)
            proba = p if proba is None else proba + p
        return proba / len(self.boosters)

    def save(self, directory: str) -> None:
        """부스터·피처 스키마·해시를 한 묶음으로 저장한다."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        names = []
        for i, booster in enumerate(self.boosters):
            name = f"booster_{i}.txt"
            booster.save_model(str(out / name))
            names.append(name)
        schema = {"labels": self.labels, "feature_cols": self.feature_cols,
                  "n_rounds": self.n_rounds, "boosters": names}
        (out / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False,
                                                               indent=2), encoding="utf-8")
        files = names + ["feature_schema.json"]
        manifest = {name: _sha256(out / name) for name in files}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "TrainedModel":
        """해시 검증 후 저장된 LightGBM 모델을 복원한다."""
        import lightgbm as lgb
        out = Path(directory)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if any(_sha256(out / name) != digest for name, digest in manifest.items()):
            raise ValueError("모델 아티팩트 무결성 검사 실패")
        schema = json.loads((out / "feature_schema.json").read_text(encoding="utf-8"))
        return cls([lgb.Booster(model_file=str(out / name)) for name in schema["boosters"]],
                   schema["labels"], schema["feature_cols"], schema["n_rounds"])


@dataclass
class TrainedRegressor:
    """LightGBM 회귀 모델 묶음과 학습 당시 피처 스키마."""

    boosters: List
    feature_cols: List[str]
    n_rounds: int

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in x.columns]
        if missing:
            raise ValueError(f"학습 때 쓰던 피처가 없다: {missing[:5]}")
        x = x.loc[:, self.feature_cols]
        return np.mean([booster.predict(x) for booster in self.boosters], axis=0)

    def save(self, directory: str) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        names = []
        for i, booster in enumerate(self.boosters):
            name = f"booster_{i}.txt"
            booster.save_model(str(out / name))
            names.append(name)
        schema = {"task_type": "regression", "feature_cols": self.feature_cols,
                  "n_rounds": self.n_rounds, "boosters": names}
        (out / "feature_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        files = names + ["feature_schema.json"]
        manifest = {name: _sha256(out / name) for name in files}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "TrainedRegressor":
        import lightgbm as lgb
        out = Path(directory)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if any(_sha256(out / name) != digest for name, digest in manifest.items()):
            raise ValueError("모델 아티팩트 무결성 검사 실패")
        schema = json.loads((out / "feature_schema.json").read_text(encoding="utf-8"))
        if schema.get("task_type") != "regression":
            raise ValueError("회귀 모델 아티팩트가 아니다")
        boosters = [lgb.Booster(model_file=str(out / name)) for name in schema["boosters"]]
        return cls(boosters, schema["feature_cols"], schema["n_rounds"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sample_weights(
    y: Sequence[int],
    balanced: bool = True,
    low_confidence_mask: np.ndarray | None = None,
    low_confidence_factor: float = 0.35,
) -> np.ndarray:
    """샘플 가중치 생성.

    Args:
        balanced: 클래스 불균형 보정(권장 — 이 프로젝트 데이터는 8.78:1 불균형이었다).
        low_confidence_mask: 라벨을 신뢰하기 어려운 행 마스크(예: 다른 모델과 불일치).
        low_confidence_factor: 그 행들에 곱할 가중치. 0.35가 실측 최적이었다.
            **0.0(완전 제거)은 오히려 급락(-2.6pp)했다 — 제거하지 말고 낮추기만 할 것.**
    """
    y = np.asarray(y)
    w = compute_sample_weight("balanced", y) if balanced else np.ones(len(y), dtype=float)
    if low_confidence_mask is not None:
        w = w.copy()
        w[np.asarray(low_confidence_mask, dtype=bool)] *= low_confidence_factor
    return w


def train_lgbm(
    x_train: pd.DataFrame,
    y_train: Sequence[int],
    labels: Sequence[str],
    params: Dict | None = None,
    n_rounds: int = 400,
    sample_weight: np.ndarray | None = None,
    seeds: Sequence[int] = (42,),
) -> TrainedModel:
    """LightGBM 학습(시드 배깅 지원).

    seeds를 여러 개 주면 같은 레시피를 시드만 바꿔 학습 후 평균 — 순수 분산감소.
    (이 프로젝트 실측 기여도는 +0.01pp로 미미했으니, 시간이 아깝다면 1개만 써도 된다.)
    """
    import lightgbm as lgb

    p = {**TUNED_A, **(params or {})}
    p["num_class"] = len(labels)
    y = np.asarray(y_train)
    boosters = []
    for seed in seeds:
        ds = lgb.Dataset(x_train, label=y, weight=sample_weight)
        boosters.append(lgb.train({**p, "seed": seed}, ds, num_boost_round=n_rounds))
    return TrainedModel(boosters=boosters, labels=list(labels),
                        feature_cols=list(x_train.columns), n_rounds=n_rounds)


def train_lgbm_regressor(
    x_train: pd.DataFrame,
    y_train: Sequence[float],
    params: Dict | None = None,
    n_rounds: int = 400,
    sample_weight: np.ndarray | None = None,
    seeds: Sequence[int] = (42,),
) -> TrainedRegressor:
    """LightGBM 회귀 학습. 여러 seed를 주면 예측 평균으로 분산을 낮춘다."""
    import lightgbm as lgb

    p = {**REGRESSION_PARAMS, **(params or {})}
    y = np.asarray(y_train, dtype=float)
    if len(x_train) != len(y):
        raise ValueError("x_train과 y_train의 행 수가 다르다")
    if not np.isfinite(y).all():
        raise ValueError("회귀 타깃에 NaN 또는 무한대가 있다")

    boosters = []
    for seed in seeds:
        estimator = lgb.LGBMRegressor(**p, n_estimators=n_rounds, random_state=seed)
        estimator.fit(x_train, y, sample_weight=sample_weight)
        boosters.append(estimator.booster_)
    return TrainedRegressor(boosters, list(x_train.columns), n_rounds)


def train_oof(
    x: pd.DataFrame,
    y: Sequence[int],
    labels: Sequence[str],
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Dict | None = None,
    n_rounds: int = 400,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """그룹 K-fold OOF 확률 생성 — 누수 없는 스태킹 피처를 만들 때 쓴다.

    Returns:
        (N, C) 확률 행렬. 각 행은 그 행을 학습에 쓰지 않은 모델의 예측이다.
    """
    import lightgbm as lgb

    p = {**TUNED_A, **(params or {})}
    p["num_class"] = len(labels)
    y = np.asarray(y)
    oof = np.zeros((len(y), len(labels)))
    for fold, (tr_idx, te_idx) in enumerate(fold_indices):
        w = None if sample_weight is None else sample_weight[tr_idx]
        ds = lgb.Dataset(x.iloc[tr_idx], label=y[tr_idx], weight=w)
        booster = lgb.train(p, ds, num_boost_round=n_rounds)
        oof[te_idx] = booster.predict(x.iloc[te_idx])
    return oof
