import iree.compiler.dialects.func
import numpy as np
import re

from typing import Dict, List, Optional, Tuple

__dtype_map: Dict[str, np.dtype] = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u32": np.uint32,
    "u64": np.uint64,
}

__flow_dispatch_tensor_pattern: re.Pattern = re.compile(
    r"^!flow\.dispatch\.tensor<(readonly|writeonly|readwrite):tensor<((?:\d+x)+[fi]\d+)>>$"
)
__fluidml_arg_pattern: re.Pattern = re.compile(r"^fluidml\.arg(\d+)axes$")


def map_str_dtype(dtype: str) -> np.dtype:
    return __dtype_map[dtype]


def get_signature(
    kernel: iree.compiler.dialects.func.FuncOp,
) -> Tuple[
    str,
    str,
    List[Tuple[Tuple[int, ...], str]],
    List[Tuple[Tuple[int, ...], str]],
    Tuple[Tuple[int, ...]],
]:
    kernel_name: str = kernel.sym_name.value
    mod_name: str = kernel.parent.parent.opview.sym_name.value
    input_types: List[Tuple[Tuple[int, ...], str]] = []
    result_types: List[Tuple[Tuple[int, ...], str]] = []
    for argument in kernel.arguments:
        arg_type: str = str(argument.type)
        match: List[str] = __flow_dispatch_tensor_pattern.findall(arg_type)
        [(rw, seq)] = match
        elems: List[str] = seq.split("x")
        dtype: str = elems.pop()
        shape: Tuple[int, ...] = tuple(map(int, elems))
        info: Tuple[Tuple[int, ...], str] = (
            shape,
            dtype,
        )
        if rw in {"readonly", "readwrite"}:
            input_types += [info]
        if rw in {"writeonly"}:
            result_types += [info]
    axes_map: Dict[int, Tuple[int, ...]] = dict()
    for attribute in kernel.attributes:
        match: Optional[re.Match[str]] = __fluidml_arg_pattern.match(attribute.name)
        if match:
            (id,) = match.groups()
            id: int = int(id)
            axes_map[id] = tuple(int(elem) for elem in attribute.attr)
    axes: Tuple[Tuple[int, ...]] = tuple(axes_map[idx] for idx in range(len(axes_map)))
    return kernel_name, mod_name, input_types, result_types, axes
