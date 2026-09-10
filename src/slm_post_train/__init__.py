"""SLM Post-Training with Unsloth."""

__version__ = "0.1.0"

# Normalize trl.import_utils package checks for compatibility with transformers >= 5.0
try:
    import trl.import_utils as _tiu

    for _attr in dir(_tiu):
        if _attr.startswith("_") and _attr.endswith("_available"):
            _val = getattr(_tiu, _attr)
            if isinstance(_val, tuple):
                setattr(_tiu, _attr, _val[0])
except Exception:
    pass
