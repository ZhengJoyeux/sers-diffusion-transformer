"""Use original lexical data paths, preserving path-seeded noise-tail completion."""
import json
from pathlib import Path


def build_datasets(**kwargs):
    from Dataset import build_datasets as original_builder
    state=json.loads((Path(__file__).resolve().parent/"prepared_state.json").read_text())
    root=Path(state["project"])
    kwargs["real_root"]=root/"data"/"real"
    kwargs["generated_root"]=root/"data"/"generated"
    return original_builder(**kwargs)
