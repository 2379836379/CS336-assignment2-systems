from __future__ import annotations

import os


def _sanitize_thread_env_var(name: str) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    try:
        if int(value) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        os.environ[name] = "1"


_sanitize_thread_env_var("OMP_NUM_THREADS")
