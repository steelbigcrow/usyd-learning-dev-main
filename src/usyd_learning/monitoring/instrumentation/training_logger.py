from __future__ import annotations

from typing import Any

from ...ml_utils import console
from ..hub import get_hub


def _patch_method(cls, name: str, wrapper) -> None:
    orig = getattr(cls, name, None)
    if orig is None:
        return
    if getattr(orig, "__monitor_patched__", False):
        return

    def _wrapped(self, *args, **kwargs):
        return wrapper(self, orig, *args, **kwargs)

    setattr(_wrapped, "__monitor_patched__", True)
    setattr(cls, name, _wrapped)


def patch_training_logger() -> None:
    """Patch TrainingLogger.begin to align monitor run folder with training results.

    After logger chooses a filename like "train-<ts>-<hash>.csv" under .training_results,
    monitoring outputs will be redirected to .monitor/"train-<ts>-<hash>"/.
    """
    try:
        from ...ml_utils.training_logger import TrainingLogger
    except Exception as e:
        console.warn(f"[monitor] skip training-logger patch (import error): {e}")
        return

    def _begin_wrapper(self: "TrainingLogger", orig, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            # Access name-mangled private: __file_names -> _TrainingLogger__file_names
            fnames = getattr(self, "_TrainingLogger__file_names", None)
            if fnames is not None:
                # Use the filename without extension as the subfolder name
                folder_name = str(getattr(fnames, "filename", ""))
                if folder_name.lower().endswith(".csv"):
                    folder_name = folder_name[:-4]
                if folder_name:
                    hub = get_hub()
                    hub.set_run_folder(folder_name)
        except Exception:
            pass
        return out

    _patch_method(TrainingLogger, "begin", _begin_wrapper)

