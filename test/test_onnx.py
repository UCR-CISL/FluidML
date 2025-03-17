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

from typing import Callable, List, Tuple, Union


@pytest.mark.parametrize(
    "name, url, entry, inputs",
    [
        (
            "bert",
            "https://github.com/onnx/models/raw/refs/heads/main/Natural_Language_Processing/bert_Opset18_transformers/bert_Opset18.onnx",
            "main_graph",
            (((1, 128), np.int64), ((1, 128), np.float32)),
        ),
        (
            "googlenet-12",
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx",
            "bvlc_googlenet",
            (((1, 3, 224, 224), np.float32),),
        ),
        (
            "gptneox",
            "https://github.com/onnx/models/raw/refs/heads/main/Generative_AI/gptneox_Opset18_transformers/gptneox_Opset18.onnx",
            "main_graph",
            (((1, 128), np.int64), ((1, 128), np.float32)),
        ),
        (
            "mobilenetv2-12",
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
            "torch-jit-export",
            (((1, 3, 224, 224), np.float32),),
        ),
        (
            "resent50-v1-12",
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
            "mxnet_converted_model",
            (((1, 3, 224, 224), np.float32),),
        ),
        (
            "squeezenet1.0-12",
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx",
            "squeezenet_old",
            (((1, 3, 224, 224), np.float32),),
        ),
    ],
)
def test_onnx(
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
        for input in model.graph.input:
            for dim in input.type.tensor_type.shape.dim:
                if not dim.HasField("dim_value"):
                    dim.dim_value = 1
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
        mlir_path, target_backends=["llvm-cpu"]
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

    iree_result: Union[
        iree.runtime.DeviceArray, Tuple[iree.runtime.DeviceArray, ...]
    ] = f(*inputs)
    if isinstance(iree_result, tuple):
        iree_result = map(lambda result: result.to_host(), iree_result)
    else:
        iree_result = tuple(iree_result.to_host())
    onnx_result = session.run(
        None, dict(zip(map(lambda input: input.name, session.get_inputs()), inputs))
    )
    # assert all(
    #     map(
    #         lambda result: np.allclose(result[0], result[1], atol=1e-3, rtol=1e-3),
    #         zip(iree_result, onnx_result),
    #     )
    # )
