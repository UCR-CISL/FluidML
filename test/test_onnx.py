import argparse
import fluidml.compiler
import hashlib
import iree.compiler
import iree.compiler.tools.import_onnx.__main__ as m
import iree.runtime
import numpy as np
import onnx
import onnxruntime
import os
import pytest
import requests
import tempfile

from typing import Callable, List, Tuple


@pytest.mark.parametrize(
    "name, url, entry, inputs",
    [
        (
            "googlenet-12",
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx",
            "bvlc_googlenet",
            (((1, 3, 224, 224), np.float32),),
        ),
        (
            "mobilenetv2-12",
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
            "torch-jit-export",
            (((1, 3, 224, 224), np.float32),),
        ),
    ],
)
def test_model(
    name: str,
    url: str,
    entry: str,
    inputs: Tuple[Tuple[Tuple[int, ...], np.dtype]],
) -> None:
    digest: int = hashlib.md5(name.encode()).hexdigest()
    temp_dir: str = tempfile.gettempdir()
    temp_path: str = f"{temp_dir}/fluidml-{digest}"
    onnx_path: str = f"{temp_path}/{name}.onnx"
    mlir_path: str = f"{temp_path}/{name}.mlir"
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
        model_bin: bytes = requests.get(url, allow_redirects=True).content
        model: onnx.ModelProto = onnx.load_model_from_string(model_bin)
        model: onnx.ModelProto = onnx.version_converter.convert_version(model, 17)
        onnx.save_model(model, onnx_path)
        onnx_args: List[str] = [
            onnx_path,
            "-o",
            mlir_path,
            "--opset-version",
            "17",
        ]
        parsed_args: argparse.Namespace = m.parse_arguments(onnx_args)
        m.main(parsed_args)
    compiled_flatbuffer: bytes = fluidml.compiler.compile_file(
        mlir_path, entry, target_backends=["llvm-cpu"]
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
    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    iree_result = f(*inputs).to_host()
    (onnx_result,) = session.run(
        None, dict(zip(map(lambda input: input.name, session.get_inputs()), inputs))
    )
    assert np.allclose(iree_result, onnx_result, atol=1e-3, rtol=1e-3)
