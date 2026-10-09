"""Dataset registry, reproducible acquisition, adapters, validation and splits.

Nothing here runs in the request path of the app; it is the data side of the
evaluation engine (``tools/datasets.py`` and ``tools/evaluate.py`` are the
command-line entry points). Large data lives under the data root
(``RD_DATA_DIR``, default ``<repo>/data``) and never in source control; only
manifests (sources, licences, checksums, counts) and split definitions are
versioned.
"""
