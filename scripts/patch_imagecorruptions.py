"""
patch_imagecorruptions.py (A8) — idempotent .venv patcher for imagecorruptions.

Fixes two API breaks that crash glass_blur and fog on modern scikit-image / NumPy:
  1. gaussian() multichannel kwarg removed in scikit-image 0.20 → channel_axis=-1
  2. np.float_ removed in NumPy 2.0 → np.float64

Run idempotently: safe to re-run; prints ALREADY PATCHED / PATCHED / ERROR per fix.

Usage:
    uv run python scripts/patch_imagecorruptions.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _find_corruptions_py() -> Path:
    try:
        import imagecorruptions
        import inspect
        pkg_init = Path(inspect.getfile(imagecorruptions))
        candidate = pkg_init.parent / "corruptions.py"
        if candidate.exists():
            return candidate
    except ImportError:
        pass
    sys.exit("ERROR: imagecorruptions not installed in the current environment. "
             "Run: uv add imagecorruptions")


def _apply_fix(text: str, pattern: str, replacement: str, label: str) -> tuple[str, str]:
    """Return (new_text, status) where status is ALREADY_PATCHED / PATCHED / NOT_FOUND."""
    if re.search(re.escape(replacement), text):
        return text, "ALREADY_PATCHED"
    new_text, n = re.subn(pattern, replacement, text)
    if n == 0:
        return text, "NOT_FOUND"
    return new_text, f"PATCHED ({n} occurrence{'s' if n > 1 else ''})"


def main() -> None:
    path = _find_corruptions_py()
    print(f"Target: {path}")

    original = path.read_text()
    text = original

    # Fix 1: multichannel=True → channel_axis=-1  (glass_blur / scikit-image ≥0.20)
    text, s1 = _apply_fix(
        text,
        pattern=r"\bgaussian\(np\.array\(x\) / 255\., sigma=c, multichannel=True\)",
        replacement="gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)",
        label="Fix 1 (glass_blur multichannel)",
    )
    print(f"  Fix 1 (glass_blur multichannel=True → channel_axis=-1): {s1}")

    # Fix 2: np.float_ → np.float64  (fog / NumPy ≥2.0)
    text, s2 = _apply_fix(
        text,
        pattern=r"\bnp\.float_\b",
        replacement="np.float64",
        label="Fix 2 (fog np.float_)",
    )
    print(f"  Fix 2 (fog np.float_ → np.float64):                     {s2}")

    if text == original:
        print("No changes written.")
        return

    path.write_text(text)
    print("Patch written successfully.")

    # Smoke-test: import the patched module
    print("Smoke-testing imports ...", end=" ", flush=True)
    try:
        import importlib
        import imagecorruptions.corruptions as _c
        importlib.reload(_c)
        print("OK")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
