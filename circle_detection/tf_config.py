"""Quiet TensorFlow C++ logs. Import before `tensorflow`."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
