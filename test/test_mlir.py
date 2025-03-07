import fluidml.compiler
import pytest
import iree.runtime
import numpy as np
import os

from typing import Callable, Tuple

cur_path: str = os.path.dirname(__file__)
mlir_path: str = os.path.join(cur_path, "mlir")


@pytest.mark.parametrize(
    "file, entry, inputs, func",
    [
        (
            "matmul.mlir",
            "matmul",
            (((256, 256), np.float32) for _ in range(2)),
            np.matmul,
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
        path, entry, target_backends=["llvm-cpu"]
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
    assert np.allclose(iree_result, np_result, atol=1e-3, rtol=1e-3)
