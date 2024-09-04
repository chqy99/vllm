import os
import torch
import vllm.envs as envs
import numpy as np
from typing import List, Union, Optional
from vllm.worker.model_runner import _BATCH_SIZES_TO_CAPTURE, _BATCH_SIZE_ALIGNMENT
from vllm.logger import init_logger
import ctypes

logger = init_logger(__name__)

IS_GATED=False

class MluHijackObject:
    hijack_objs = []

    @classmethod
    def add_hijack_object(cls, obj, org_func, hijack_func):
        cls.hijack_objs.append((obj, org_func, hijack_func))

    @classmethod
    def apply_hijack(cls):
        for obj, org_func, hijack_func in cls.hijack_objs:
            if type(org_func) == str:
                org_func_name = org_func
            else:
                org_func_name = org_func.__name__
            setattr(obj, org_func_name, hijack_func)

    @classmethod
    def undo_hijack(cls):
        for obj, org_func, hijack_func in cls.hijack_objs:
            if type(org_func) == str:
                delattr(obj, org_func)
            else:
                org_func_name = org_func.__name__
                setattr(obj, org_func_name, org_func)


def set_batch_sizes_to_capture(max_batch_size_to_capture):
    logger.info(f"Set max_batch_size_to_capture={max_batch_size_to_capture}")
    global _BATCH_SIZES_TO_CAPTURE
    if max_batch_size_to_capture <= _BATCH_SIZES_TO_CAPTURE[-1]:
        # Remove the batchs that large than max_batch_size_to_capture
        _BATCH_SIZES_TO_CAPTURE = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= max_batch_size_to_capture
        ]
    else:
        # Add new capture batchs into _BATCH_SIZES_TO_CAPTURE
        new_capture_batch = _BATCH_SIZES_TO_CAPTURE[-1] + _BATCH_SIZE_ALIGNMENT
        while new_capture_batch <= max_batch_size_to_capture:
            _BATCH_SIZES_TO_CAPTURE.append(new_capture_batch)
            new_capture_batch = _BATCH_SIZES_TO_CAPTURE[-1] + _BATCH_SIZE_ALIGNMENT

    # Push the max_batch_size_to_capture into _BATCH_SIZES_TO_CAPTURE
    if max_batch_size_to_capture not in _BATCH_SIZES_TO_CAPTURE:
        _BATCH_SIZES_TO_CAPTURE.append(max_batch_size_to_capture)


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ('hidden_size', ctypes.c_double),
        ('vocab_size', ctypes.c_double),
        ('ffn_inner_size', ctypes.c_double),
        ('layer_num', ctypes.c_double),
        ('head_num', ctypes.c_double),
        ('head_size', ctypes.c_double),
        ('head_num_kv', ctypes.c_double),
        ('tp_num', ctypes.c_double),
        ('shared_expert_intermediate_size', ctypes.c_double),
        ('use_gated_ffn', ctypes.c_bool),
        ('experts_num', ctypes.c_int),
        ('topk_num', ctypes.c_int),
        ('use_causal_mask', ctypes.c_bool),
        ('kv_cache_dtype', ctypes.c_char_p),
        ('smooth_quant_type', ctypes.c_char_p),
        ('data_type', ctypes.c_char_p),
        ('filter_data_type', ctypes.c_char_p),
    ]

def set_is_gated(flag):
    global IS_GATED
    IS_GATED=flag

def get_is_gated():
    return IS_GATED
