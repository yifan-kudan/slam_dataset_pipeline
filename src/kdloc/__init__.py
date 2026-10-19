"""KdLocalizer test campaigns: the dataset-independent half.

A campaign replays recorded bags through kd_localizer_node (and kd_relocalizer_node)
against a prior map, one run at a time, and scores what the localizer reported. Every
dataset needs the same machinery around the localizer; only the inputs differ. Those
inputs live in one file per campaign, ``campaign.yaml`` (template:
``config/examples/kdloc_campaign.example.yaml``), and in one reference trajectory per
sequence (TUM, in the map frame) that the dataset's own code produces.

    make_runs.py   campaign.yaml -> runs/<name>/{run.env,params.yaml,cfg.yaml,init.yaml}
    monitor.py     /localization_status + /pose_with_covariance -> pose.csv (in the container)
    procmon.py     per-node CPU / memory from the host's /proc -> proc.csv (on the host)
    logs.py        readers for everything a run leaves behind (standard library only)
    evaluate.py    runs + reference trajectories -> eval/<name>.json + summary table (evo)

The shell side is ``scripts/kdloc/campaign.sh`` (host) and ``scripts/kdloc/run_one.sh``
(inside the container). Scripts here are run directly and put ``src/`` on the path the
same way ``src/hilti/`` does::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kdloc import logs
"""
