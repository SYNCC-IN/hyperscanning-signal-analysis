"""Shared-JSON config loading for the `scripts/pipeline_exploratory_mlutimodal_dDTF/`
stage01-stage06 pipeline.

The pipeline's per-stage config constants (paths, estimator settings, MCMC
config, thresholds, ...) live in one JSON file
(`scripts/pipeline_exploratory_mlutimodal_dDTF/pipeline_config.json`) instead
of being hard-coded per script, so a value shared by several stages (e.g. the
estimator name, which stage06's docstring already flags as "must match Stage
5's ESTIMATOR") has exactly one source of truth. Each stage script loads its
own settings with `load_stage_config` and assigns them to the same bare
module-level names it used before (`FILMS = CFG["FILMS"]`, etc.), so no
downstream function body needs to change.
"""

import json
from pathlib import Path


def load_stage_config(config_path, stage_key):
    """Load one stage's settings from the pipeline's shared JSON config file.

    Merges the JSON's ``"shared"`` section (constants used by more than one
    stage) with its `stage_key` section (that stage's own settings); on a key
    collision the stage-specific value wins. Neither section is required to
    exist -- a stage with no stage-specific overrides can omit its section.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Path to the pipeline's `pipeline_config.json`.
    stage_key : str
        Top-level key identifying the calling stage, e.g. `"stage03_mvar_order"`.

    Returns
    -------
    dict
        `{**shared, **stage_specific}`.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    shared = config.get("shared", {})
    stage_specific = config.get(stage_key, {})
    return {**shared, **stage_specific}
