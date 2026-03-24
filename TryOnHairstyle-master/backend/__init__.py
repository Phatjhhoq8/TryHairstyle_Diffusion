"""
HairFusion Backend Package.
Adds backend/models/ to sys.path so internal library imports
(ldm.*, cldm.*, taming.*, annotator.*) work without modification.
"""
import sys
import os

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
_MODELS_DIR = os.path.join(_BACKEND_DIR, 'models')
for _d in (_BACKEND_DIR, _MODELS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
