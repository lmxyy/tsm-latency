import argparse
import os
import time
import warnings
from typing import Tuple

import numpy as np
import onnx
import torch
import tvm
import tvm.relay
from tqdm import tqdm
from tvm.contrib import graph_runtime


def onnx2tvm_module(args, inputs, target):
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(inputs):
            name = "i" + str(index)
            input_shapes[name] = torch_input.shape
        onnx_model = onnx.load_model(args.onnx_path)
        relay_module, params = tvm.relay.frontend.from_onnx(onnx_model, shape=input_shapes)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target, params=params)
    return graph, tvm_module, params


def get_executor(args, inputs, target):
    prefix = 'mobilenet_tsm_tvm_%s' % target
    save_dir = 'models/tvm'
    os.makedirs(save_dir, exist_ok=True)
    lib_fname = os.path.join(save_dir, '%s.tar' % prefix)
    graph_fname = os.path.join(save_dir, '%s.json' % prefix)
    params_fname = os.path.join(save_dir, '%s.params' % prefix)

    if os.path.exists(lib_fname) and os.path.exists(graph_fname) and os.path.exists(
            params_fname) and not args.force_rebuild:
        if args.onnx_path is not None:
            warnings.warn('The input onnx path is actually not used!!!')
        with open(graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.runtime.load_module(lib_fname)
        params = tvm.relay.load_param_dict(bytearray(open(params_fname, 'rb').read()))
    else:
        assert args.onnx_path is not None
        graph, tvm_module, params = onnx2tvm_module(args, inputs, target)
        tvm_module.export_library(lib_fname)
        with open(graph_fname, 'wt') as f:
            f.write(graph)
        with open(params_fname, 'wb') as f:
            f.write(tvm.relay.save_param_dict(params))

    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, value)
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))

    return executor, ctx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--force_rebuild', action='store_true')
    parser.add_argument('--warmup_times', type=int, default=10)
    parser.add_argument('--test_times', type=int, default=20)
    args = parser.parse_args()
    inputs = (torch.ones(1, 3, 224, 224),
              torch.zeros([1, 3, 56, 56]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 20, 7, 7]),
              torch.zeros([1, 20, 7, 7]))
    target = 'llvm'
    executor, ctx = get_executor(args, inputs, target)
    x = tvm.nd.array(np.ones([1, 3, 224, 224], dtype=np.float32), ctx=ctx)
    buffer = (
        tvm.nd.array(np.zeros([1, 3, 56, 56], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 4, 28, 28], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 4, 28, 28], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 8, 14, 14], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 8, 14, 14], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 8, 14, 14], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 12, 14, 14], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 12, 14, 14], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 20, 7, 7], dtype=np.float32), ctx=ctx),
        tvm.nd.array(np.zeros([1, 20, 7, 7], dtype=np.float32), ctx=ctx)
    )
    inputs = (x,) + buffer
    for _ in tqdm(range(args.warmup_times)):
        outputs = executor(inputs)
    start_time = time.time()
    for _ in tqdm(range(args.test_times)):
        outputs = executor(inputs)
    end_time = time.time()
    print((end_time - start_time) / args.test_times * 1000)

    # x, y = outputs[0], outputs[1:]
    #
    # print(abs(x.asnumpy()).sum())
