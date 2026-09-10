"""SLM Post-Training with Unsloth."""

# Unsloth must be imported before trl/transformers so kernel patches are applied
import unsloth

__version__ = "0.1.0"


def _patch_trl_compatibility():
    """Normalize trl.import_utils package checks for compatibility with transformers >= 5.0."""
    try:
        import trl.import_utils as tiu

        for attr in dir(tiu):
            if attr.startswith("_") and attr.endswith("_available"):
                val = getattr(tiu, attr)
                if isinstance(val, tuple):
                    setattr(tiu, attr, val[0])
    except Exception:
        pass


_patch_trl_compatibility()
