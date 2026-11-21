"""Which dataset action labels count as focused and which as distracted.

The datasets label actions, not attention, so every model here learns from stand-in
(proxy) labels. The mapping is fixed before training and named in every result.

  thesis   the mapping used in the thesis (v1.0.3): SCB only, BowHead and discuss counted
           as distracted.

The strict and lenient mappings for the multi-dataset benchmark (listening and Lookup as
focused; bow head, discuss and stand as ambiguous) are added together with the new dataset
loaders.
"""
MAPPINGS = {
    "thesis": (
        frozenset({"read", "write", "hand-raising"}),
        frozenset({"TurnHead", "BowHead", "discuss"}),
    ),
}


def get(name):
    """(focused labels, distracted labels) for a mapping name."""
    try:
        return MAPPINGS[name]
    except KeyError:
        raise SystemExit(f"unknown label mapping {name!r}; known: {sorted(MAPPINGS)}")
