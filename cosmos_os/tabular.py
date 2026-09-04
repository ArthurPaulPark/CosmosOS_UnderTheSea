"""표형(tabular) 피처 준비 — 도메인 무관 자립 모듈.

Cosmos 엔진(`Cosmos/feature_engineering.py` + `utils.reduce_mem_usage`)에서 추출.
원본은 `from utils import reduce_mem_usage` 외부 의존이 있었으나 여기서는 자립형으로
내장 — numpy/pandas/scikit-learn 외 어떤 의존성도 없다.

제공 기능(전부 데이터셋 무관):
  - 결측 지시자 자동 생성 + 결측 대치(median/mean)
  - 범주형 자동 감지 및 category dtype 캐스팅
  - group-by 통계 피처(범주×수치 평균/표준편차)
  - 수치 상호작용 피처(곱/나눗셈)
  - 메모리 다운캐스팅
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureBundle:
    """준비된 학습/평가 행렬과 메타데이터."""

    x_train: pd.DataFrame
    y_train: pd.Series | None
    x_test: pd.DataFrame | None
    categorical_cols: List[str]
    feature_cols: List[str]
    train_ids: pd.Series | None = None
    test_ids: pd.Series | None = None


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """수치 컬럼 dtype을 축소해 메모리 사용량을 줄인다(값 보존)."""
    for col in df.columns:
        col_type = df[col].dtype
        # 최신 pandas의 문자열 dtype(StrDType)은 object와 다르게 취급되므로
        # dtype 비교 대신 is_numeric_dtype으로 판정한다(문자열/범주형 안전).
        if pd.api.types.is_numeric_dtype(df[col]) and not isinstance(col_type, pd.CategoricalDtype):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            if str(col_type)[:3] == "int":
                for dt in (np.int8, np.int16, np.int32, np.int64):
                    if c_min > np.iinfo(dt).min and c_max < np.iinfo(dt).max:
                        df[col] = df[col].astype(dt)
                        break
            else:
                # float16은 수치 불안정으로 의도적으로 건너뜀(원본 엔진과 동일 정책)
                df[col] = df[col].astype(
                    np.float32
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max
                    else np.float64
                )
    return df


def detect_categorical_features(df: pd.DataFrame, configured: List[str] | None = None) -> List[str]:
    """범주형 컬럼 감지 — 명시 설정이 있으면 그것을 우선한다.

    dtype 문자열 비교 대신 pandas API 사용(최신 pandas의 `str` dtype 대응).
    """
    if configured is not None:
        return [c for c in configured if c in df.columns]
    return [c for c in df.columns
            if pd.api.types.is_bool_dtype(df[c]) or not pd.api.types.is_numeric_dtype(df[c])]


def auto_feature_engineering(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame | None,
    categorical_cols: List[str],
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """결측 지시자·대치·스케일링·group-by 통계·수치 상호작용을 설정에 따라 자동 생성.

    **누수 방지**: group-by 통계와 스케일러는 train에서만 적합(fit)하고 test에는
    매핑/변환만 적용한다.
    """
    x_train = x_train.copy()
    if x_test is not None:
        x_test = x_test.copy()

    numeric_cols = [
        c for c in x_train.columns
        if c not in categorical_cols and pd.api.types.is_numeric_dtype(x_train[c])
    ]

    # 1) 결측 지시자
    for col in x_train.columns:
        if x_train[col].isnull().any():
            flag = f"{col}_isnull"
            x_train[flag] = x_train[col].isnull().astype(int)
            if x_test is not None:
                x_test[flag] = x_test[col].isnull().astype(int)

    # 2) 결측 대치(train 통계만 사용)
    strategy = config.get("missing_value_strategy", "median")
    if strategy != "none":
        for col in numeric_cols:
            if x_train[col].isnull().any():
                fill = x_train[col].median() if strategy == "median" else x_train[col].mean()
                x_train[col] = x_train[col].fillna(fill)
                if x_test is not None:
                    x_test[col] = x_test[col].fillna(fill)
        for col in categorical_cols:
            if col in x_train.columns and x_train[col].isnull().any():
                mode = x_train[col].mode()
                fill = mode.iloc[0] if not mode.empty else "missing"
                x_train[col] = x_train[col].fillna(fill)
                if x_test is not None:
                    x_test[col] = x_test[col].fillna(fill)

    # 3) 스케일링(선택)
    if config.get("scaling", "none") == "standard" and numeric_cols:
        scaler = StandardScaler()
        x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
        if x_test is not None:
            x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    # 4) 상호작용 피처
    if config.get("interaction_features", True):
        max_cats = config.get("max_groupby_cats", 3)
        max_nums = config.get("max_groupby_nums", 5)
        valid_cats = [c for c in categorical_cols
                      if c in x_train.columns and 2 <= x_train[c].nunique() <= 100][:max_cats]
        valid_nums = numeric_cols[:max_nums]
        for cat_col in valid_cats:
            for num_col in valid_nums:
                mean_map = x_train.groupby(cat_col, observed=True)[num_col].mean()
                std_map = x_train.groupby(cat_col, observed=True)[num_col].std()
                x_train[f"{num_col}_by_{cat_col}_mean"] = x_train[cat_col].map(mean_map)
                x_train[f"{num_col}_by_{cat_col}_std"] = x_train[cat_col].map(std_map)
                if x_test is not None:
                    x_test[f"{num_col}_by_{cat_col}_mean"] = x_test[cat_col].map(mean_map)
                    x_test[f"{num_col}_by_{cat_col}_std"] = x_test[cat_col].map(std_map)

        interact = numeric_cols[:config.get("max_interaction_cols", 4)]
        eps = 1e-5
        for i in range(len(interact)):
            for j in range(i + 1, len(interact)):
                a, b = interact[i], interact[j]
                x_train[f"{a}_x_{b}"] = x_train[a] * x_train[b]
                x_train[f"{a}_div_{b}"] = x_train[a] / (x_train[b] + eps)
                if x_test is not None:
                    x_test[f"{a}_x_{b}"] = x_test[a] * x_test[b]
                    x_test[f"{a}_div_{b}"] = x_test[a] / (x_test[b] + eps)

    return x_train, x_test


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    categorical_cols: List[str],
    numerical_cols: List[str] | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    preprocessing_config: Dict[str, Any] | None = None,
) -> FeatureBundle:
    """타깃/ID 분리 → 화이트리스트 → 자동 피처엔지니어링 → dtype 정리.

    Args:
        train_df/test_df: 원본 데이터프레임(둘 다 같은 컬럼 구성이어야 함).
        categorical_cols/numerical_cols: 사용할 피처 화이트리스트.
        target_col/id_col: 있으면 분리 보관.
        preprocessing_config: auto_feature_engineering 설정.
    """
    cats = list(categorical_cols)
    nums = list(numerical_cols or [])
    config = preprocessing_config or {}

    y_train = train_df[target_col] if target_col and target_col in train_df else None
    train_ids = train_df[id_col] if id_col and id_col in train_df else None
    test_ids = (test_df[id_col] if test_df is not None and id_col and id_col in test_df else None)

    whitelist = cats + nums
    missing = [c for c in whitelist if c not in train_df.columns]
    if missing:
        raise KeyError(f"train_df에 없는 피처: {missing[:10]}")
    x_train = train_df[whitelist].copy()
    x_test = None if test_df is None else test_df[whitelist].copy()

    x_train, x_test = auto_feature_engineering(x_train, x_test, cats, config)

    for col in cats:
        if col in x_train.columns:
            x_train[col] = x_train[col].astype("category")
        if x_test is not None and col in x_test.columns:
            # train의 카테고리 목록으로 정렬 — 미관측 값은 NaN이 되어 안전하게 처리됨
            x_test[col] = pd.Categorical(x_test[col], categories=x_train[col].cat.categories)

    x_train = reduce_mem_usage(x_train)
    if x_test is not None:
        x_test = reduce_mem_usage(x_test)

    return FeatureBundle(
        x_train=x_train, y_train=y_train, x_test=x_test,
        categorical_cols=cats, feature_cols=list(x_train.columns),
        train_ids=train_ids, test_ids=test_ids,
    )
