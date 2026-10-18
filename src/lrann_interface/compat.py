"""Compatibility for the original Windows timing calls on current Python."""

import time

# On Windows, the removed time.clock used the same performance counter.
if not hasattr(time, "clock"):
    time.clock = time.perf_counter
