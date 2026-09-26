"""NDTP ingestion, bounded vehicle state and the dispatcher API.

The Backend owns the stream, the clocks and the plan. It never trains a model and never
implements a second feature builder: features come from ``transport_ml.features``.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1"
