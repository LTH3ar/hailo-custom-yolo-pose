"""Which dataset action labels count as focused and which as distracted.

The datasets label actions, not attention, so every model here learns from stand-in
(proxy) labels. Each dataset's own label names are first translated to one shared set of
action names (ACTIONS); a mapping then says which actions are focused and which distracted.
Actions in neither set are left out. The mapping is fixed before training and named in
every result.

  thesis   the mapping used in the thesis (v1.0.3), on SCB: bow head and discuss count as
           distracted. No "lookup" class existed in SCB.
  strict   focused: read, write, raise hand, look up. distracted: turn head. The ambiguous
           actions (bow head looks like writing; discuss may be on-task group work; stand may
           be answering a question) are left out.
  lenient  as strict, but bow head, discuss and stand count as distracted.

Reporting strict and lenient side by side shows whether a conclusion depends on how the
ambiguous actions are counted.
"""
ACTIONS = ("read", "write", "raise_hand", "lookup", "turn_head", "bow_head", "discuss", "stand")

# each dataset's label names -> the shared action names
ALIASES = {
    "scb": {"read": "read", "write": "write", "hand-raising": "raise_hand",
            "TurnHead": "turn_head", "BowHead": "bow_head", "discuss": "discuss"},
    "clabehavior": {"read": "read", "write": "write", "raise_hand": "raise_hand", "lookup": "lookup",
                    "turn_head": "turn_head", "discuss": "discuss", "stand": "stand"},
}

_FOCUSED = frozenset({"read", "write", "raise_hand", "lookup"})
MAPPINGS = {
    "thesis": (frozenset({"read", "write", "raise_hand"}), frozenset({"turn_head", "bow_head", "discuss"})),
    "strict": (_FOCUSED, frozenset({"turn_head"})),
    "lenient": (_FOCUSED, frozenset({"turn_head", "bow_head", "discuss", "stand"})),
}


def get(name):
    """(focused actions, distracted actions) for a mapping name."""
    try:
        return MAPPINGS[name]
    except KeyError:
        raise SystemExit(f"unknown label mapping {name!r}; known: {sorted(MAPPINGS)}")


def action(dataset, label):
    """The shared action name for a dataset's label, or None if the label isn't an action we use."""
    return ALIASES[dataset].get(label, ALIASES[dataset].get(label.lower()))
