"""Evaluation engine: reproducible benchmarks with full provenance.

Every run writes a benchmark record (app.evaluation.records) with the model
and version, dataset and split, the exact configuration, hardware/software,
latency/throughput/memory, task metrics, per-class results, error analysis,
error examples and the date and git commit. Only measured values are written;
anything not run is reported as "not measured" by the documentation, never
estimated.
"""
