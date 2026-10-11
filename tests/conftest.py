"""Pytest configuration and fixture setup."""

import os
import sys
from pathlib import Path

# Add project root to sys.path so 'finrisk' and 'scripts' can be imported in all tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offline Hugging Face configuration and isolated temp storage if E:\hf_home exists
if os.path.exists(r"E:\hf_home"):
    os.environ["HF_HOME"] = r"E:\hf_home"
    os.environ["HF_HUB_CACHE"] = r"E:\hf_home\hub"
    os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_home"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if os.path.exists(r"E:\pip_temp"):
    os.environ["TEMP"] = r"E:\pip_temp"
    os.environ["TMP"] = r"E:\pip_temp"
