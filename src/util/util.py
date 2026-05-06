
"""
Collection of commonly used uitility functions
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy import io
import torch
from torch.utils import data
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from IPython.core.debugger import set_trace
import scipy.io as sio
from itertools import combinations
from scipy.special import gamma
from scipy.special import loggamma
from scipy import stats
from scipy.optimize import minimize
import random
import datetime
import collections
import logging
import math
import sys
import copy


def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def unflatten(flat, tensor):
   offset=0
   numel = tensor.numel()
   output = (flat.narrow(0, offset, numel).view_as(tensor))
   return output


def group_by_dtype(tensors):
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def make_logger(rank, verbose=True):
    logger = logging.getLogger(__name__)
    if not getattr(logger, 'handler_set', None):
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '{}'.format(rank)
        format_str += ': %(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)  # prints to console
        logger.handler_set = True
    if not getattr(logger, 'level_set', None):
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.level_set = True
    return logger


def quantize_tensor(out_msg, comp_fn, quantization_level, is_biased = True):
    out_msg_comp = copy.deepcopy(out_msg)
    quantized_values = comp_fn.compress(out_msg_comp, None, quantization_level, is_biased)
    return quantized_values

def quantize_layerwise(out_msg, comp_fn, quantization_level, is_biased = True):
    quantized_values = []
    for param in out_msg:
        _quantized_values = comp_fn.compress(param, None, quantization_level, is_biased)
        quantized_values.append(_quantized_values)
    return quantized_values

def sparsify_layerwise(out_msg, comp_fn, comp_op, compression_ratio, is_biased=True):
    selected_values  = []
    selected_indices = []
    selected_shapes  = []
    for param in out_msg:
        p = flatten_tensors(param)
        values, indices = comp_fn.compress(p, comp_op, compression_ratio, is_biased)
        selected_values.append(values)
        selected_indices.append(indices)
        selected_shapes.append(len(values)) # should be same for all nodes, length of compressed tensor at each layer
    flat_values  = flatten_tensors(selected_values)
    flat_indices = flatten_tensors(selected_indices)
    comp_msg     = torch.cat([flat_values, flat_indices.type(flat_values.dtype)])
    return comp_msg, selected_shapes

def unsparsify_layerwise(msg, shapes, ref_param):
    out_msg  = []
    val_size = int(len(msg)/2)
    values   = msg[:val_size]
    indices  = msg[val_size:]
    indices  = indices.type(torch.cuda.LongTensor)
    pointer  = 0
    i        = 0
    for ref in ref_param:
        param = torch.zeros_like(ref)
        p = flatten_tensors(param)
        layer_values  = values[pointer:(pointer+shapes[i])]
        layer_indices = indices[pointer:(pointer+shapes[i])]
        p[layer_indices] = layer_values.type(ref.data.dtype)
        layer_msg        = unflatten(p, ref)
        out_msg.append(layer_msg)
        pointer  += shapes[i]
        i        += 1
    return out_msg
        

def precision(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum()
    accuracy = 100.0 * correct / target.size(0)
    return accuracy
