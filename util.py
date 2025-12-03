import easyocr
import cv2
import numpy as np
import string
"""Backward-compatible shim: `util.py` is kept for compatibility but re-exports
the new `ocr_utils` module. Please import from `ocr_utils` in new code.

This file will raise a DeprecationWarning when imported.
"""

from .ocr_utils import *  # noqa: F401,F403
import warnings

warnings.warn("'util.py' is deprecated; import from 'ocr_utils.py' instead", DeprecationWarning)
# Mapping dictionaries for character conversion
