# -*- coding: utf-8 -*-
"""Writes machine-readable group membership for all 11 splits (primary seed 42
+ evaluation seeds 0-9) of each class to ../splits/*.json, so any result can be
reconstructed exactly."""
import json, os
import common as C

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "splits")

def concrete_splits():
    df, g, ug = C.load_concrete(); d = {}
    for tag, seed in [("primary_seed42", 42)] + [(f"eval_seed{s}", s) for s in range(10)]:
        sp = C.concrete_split_masks(g, ug, seed)
        d[tag] = {"train": [int(i) for i in (sp=='train').nonzero()[0]],
                  "cal":   [int(i) for i in (sp=='cal').nonzero()[0]],
                  "test":  [int(i) for i in (sp=='test').nonzero()[0]]}
    return d

def sfrc_splits():
    _, X, y, gid = C.load_sfrc(); d = {}
    for tag, seed in [("primary_seed42", 42)] + [(f"eval_seed{s}", s) for s in range(10)]:
        a, b, c = C.sfrc_split_masks(gid, seed)
        d[tag] = {"train": [int(i) for i in a.nonzero()[0]],
                  "cal":   [int(i) for i in b.nonzero()[0]],
                  "test":  [int(i) for i in c.nonzero()[0]]}
    return d

if __name__ == "__main__":
    json.dump(concrete_splits(), open(os.path.join(OUT, "concrete_splits.json"), "w"), indent=0)
    json.dump(sfrc_splits(),     open(os.path.join(OUT, "sfrc_splits.json"), "w"), indent=0)
    print("Wrote splits/concrete_splits.json and splits/sfrc_splits.json (11 splits each).")
