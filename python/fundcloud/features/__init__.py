"""Feature engineering — sklearn-compatible panels of indicators + a cache.

Pipelines compose indicators into a single :class:`FeaturePipeline`;
:class:`FeatureStore` keys computed panels by ``(dataset, pipeline_hash)``
so the second fit is an O(read). Custom indicators subclass
:class:`IndicatorSpec`.
"""

from __future__ import annotations

from fundcloud.features.pipeline import FeaturePipeline
from fundcloud.features.store import FeatureStore

__all__ = ["FeaturePipeline", "FeatureStore"]
