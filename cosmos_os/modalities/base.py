"""모달리티 플러그인 인터페이스 — 올라운더 확장의 기반.

**설계 원칙**: 모달리티마다 다른 것은 (1) 피처 추출 (2) 후보 모델 계열
(3) **누수의 단위** 세 가지뿐이다. 분할 규율·통계 검정·진단은 전부 공용이다.

세 번째가 가장 중요하고 가장 자주 틀린다:

| 모달리티 | 누수 단위 | 잘못 하면 |
|---|---|---|
| 표형/텍스트 | 그룹(세션·사용자·문서) | 같은 세션이 학습·평가에 나뉨 |
| **시계열** | **시간** | **미래로 학습하고 과거를 맞춤(가장 흔한 사고)** |
| 이미지 | 촬영주체(환자·기기·장소) | 같은 환자의 다른 각도가 양쪽에 |
| 패널(그룹+시간) | 그룹 AND 시간 | 둘 중 하나만 지키면 여전히 샘 |

플러그인은 `leakage_unit`으로 자기 모달리티의 누수 단위를 선언해야 하고,
`AutoPipeline`이 그에 맞는 분할기를 자동 선택한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """플러그인이 만들어낸 피처 명세."""

    frame: pd.DataFrame
    categorical_cols: List[str] = field(default_factory=list)
    numerical_cols: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.frame)}행 × {len(self.categorical_cols)+len(self.numerical_cols)}피처 "
                f"(범주 {len(self.categorical_cols)} / 수치 {len(self.numerical_cols)})")


class ModalityPlugin(ABC):
    """한 모달리티를 처리하는 플러그인.

    구현해야 할 것:
        name           표시용 이름
        leakage_unit   "group" | "time" | "group_time" | "none"
        build_features 원시 입력 → FeatureSpec
        candidate_models 이 모달리티에서 시도할 모델 계열 이름 목록
    """

    #: 누수 단위 — AutoPipeline이 분할기를 고르는 근거
    leakage_unit: str = "group"

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build_features(self, data: Any, **kwargs) -> FeatureSpec:
        """원시 입력을 학습 가능한 피처 프레임으로 변환."""

    def candidate_models(self) -> List[str]:
        """이 모달리티에서 시도할 만한 모델 계열. selector가 bake-off 한다."""
        return ["lgbm"]

    def caveats(self) -> List[str]:
        """이 모달리티에서 흔히 저지르는 실수 — 리포트에 그대로 실린다."""
        return []


def detect_leakage_unit_columns(
    df: pd.DataFrame,
    group_col: str | None = None,
    time_col: str | None = None,
) -> dict:
    """분할에 쓸 그룹/시간 컬럼을 확정하고, 빠졌으면 경고한다.

    누수 단위를 잘못 잡는 것이 이 프로젝트에서 가장 비싼 실수였다
    (검증셋의 99.6%가 학습셋과 세션 공유). 그래서 조용히 넘어가지 않고
    명시적으로 경고를 만든다.
    """
    warnings_: List[str] = []
    if group_col and group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}'이 데이터에 없다")
    if time_col and time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}'이 데이터에 없다")

    if group_col:
        n_groups = df[group_col].nunique()
        rows_per_group = len(df) / max(n_groups, 1)
        if rows_per_group < 1.05:
            warnings_.append(
                f"그룹당 평균 {rows_per_group:.2f}행 — 그룹 분할의 의미가 거의 없다. "
                "그룹 키가 사실상 행 ID일 가능성을 확인하라.")
    else:
        warnings_.append(
            "group_col이 지정되지 않았다. 같은 개체(사용자·세션·환자)가 여러 행을 "
            "갖는 데이터라면 무작위 분할은 누수를 일으킨다.")

    if time_col:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        if ts.isna().mean() > 0.1:
            warnings_.append(f"time_col '{time_col}'의 10% 이상이 날짜로 파싱되지 않는다.")
        elif not ts.is_monotonic_increasing:
            warnings_.append(
                f"time_col '{time_col}'이 정렬되어 있지 않다. 시간 분할 전에 정렬하라.")
    return {"group_col": group_col, "time_col": time_col, "warnings": warnings_}
