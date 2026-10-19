"""Hilti SLAM Challenge 2023 evaluation: control-point scoring, map levelling, report.

Every script here is meant to be run directly (``python3 src/hilti/eval_fastlio.py``)
and starts with the same two lines, which put ``src/`` on the path so both the flat
pipeline modules and this package import by name::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hilti import dataset

No script carries an absolute path: dataset locations come from config/paths.yaml,
rig calibration from config/hilti/platforms.yaml, and which run each sequence's
numbers come from is config/hilti/sequences.yaml. See dataset.py for the accessors.
"""
