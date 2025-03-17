import fluidml.compiler
import pytest
import iree.runtime
import numpy as np
import os

from typing import Callable, Tuple

cur_path: str = os.path.dirname(__file__)
mlir_path: str = os.path.join(cur_path, "mlir")


def conv2d(input: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, _, h, w = input.shape
    f, _, kh, kw = weight.shape
    y: np.ndarray = np.zeros((n, f, h - kh + 1, w - kw + 1))
    for i in range(n):
        for j in range(f):
            for k in range(h - kh + 1):
                for l in range(w - kw + 1):
                    y[i, j, k, l] = np.sum(
                        input[i, :, k : k + kh, l : l + kw] * weight[j]
                    )
    return y


def fc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    y: np.ndarray = np.matmul(a, b)
    y_shift: np.ndarray = y - np.max(y, axis=-1, keepdims=True)
    y_exp: np.ndarray = np.exp(y_shift)
    y: np.ndarray = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    return y


@pytest.mark.parametrize(
    "file, entry, inputs, func",
    [
        (
            "conv2d.mlir",
            "conv2d",
            (((1, 3, 224, 224), np.float32), ((3, 3, 3, 3), np.float32)),
            conv2d,
        ),
        (
            "fc.mlir",
            "fc",
            (((256, 256), np.float32) for _ in range(2)),
            fc,
        ),
    ],
)
def test_mlir(
    file: str,
    entry: str,
    inputs: Tuple[Tuple[Tuple[int, ...], np.dtype]],
    func: Callable,
) -> None:
    path: str = os.path.join(mlir_path, file)
    compiled_flatbuffer: bytes = fluidml.compiler.compile_file(
        path, target_backends=["llvm-cpu"]
    )
    config: iree.runtime.Config = iree.runtime.Config("local-task")
    ctx: iree.runtime.SystemContext = iree.runtime.SystemContext(config=config)
    vm_module: iree.runtime.VmModule = iree.runtime.VmModule.copy_buffer(
        ctx.instance, compiled_flatbuffer
    )
    ctx.add_vm_module(vm_module)
    inputs: Tuple[np.ndarray] = tuple(
        np.random.rand(*input_shape).astype(dtype) for input_shape, dtype in inputs
    )
    f: Callable = ctx.modules.module[entry]
    iree_result: np.ndarray = f(*inputs).to_host()
    np_result: np.ndarray = func(*inputs)
    # assert np.allclose(iree_result, np_result, atol=1e-3, rtol=1e-3)
