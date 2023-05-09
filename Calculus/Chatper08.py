import numpy as np
import torch


def average_flow_rate(v, t1: float, t2: float):
    """
    Calculate average flow rate, differential for delta
    Args:
        v: volume
        t1: time stamp 1
        t2: time stamp 2
    """
    return (v(t2) - v(t1))/(t2 - t1)

