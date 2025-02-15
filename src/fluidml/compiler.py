import iree.compiler


from typing import List, Union

from .run import run

COMPILATION_STAGES: List = [
    "start",
    "input",
    "abi",
    "preprocessing",
    "global-optimization",
    "dispatch-creation",
    "flow",
    "stream",
    "executable-sources",
    "executable-configurations",
    "executable-targets",
    "hal",
    "vm",
    "end",
]


def compile_file(input_file: str, entry: str, **kwargs) -> bytes:
    with open(input_file, "r") as f:
        return compile_str(f.read(), entry, **kwargs)


def compile_str(input_str: Union[str, bytes], entry: str, **kwargs) -> bytes:
    extra_args: List[str] = kwargs.get("extra_args", [])
    compile_from_flags: List[str] = list(
        filter(lambda flag: flag.startswith("--compile-from="), extra_args)
    )
    compile_to_flags: List[str] = list(
        filter(lambda flag: flag.startswith("--compile-to="), extra_args)
    )
    if compile_from_flags:
        (compile_from_stage,) = compile_from_flags
        compile_from_stage: str = compile_from_stage.removeprefix("--compile-from=")
    else:
        compile_from_stage: str = COMPILATION_STAGES[0]
    if compile_to_flags:
        (compile_to_stage,) = compile_to_flags
        compile_to_stage: str = compile_to_stage.removeprefix("--compile-to=")
    else:
        compile_to_stage: str = COMPILATION_STAGES[-1]
    compile_from_index: int = COMPILATION_STAGES.index(compile_from_stage)
    compile_to_index: int = COMPILATION_STAGES.index(compile_to_stage)
    flow_index: int = COMPILATION_STAGES.index("flow")
    if compile_from_index <= flow_index <= compile_to_index:
        flow: bytes = iree.compiler.compile_str(
            input_str, extra_args=[f"--compile-to=flow", *extra_args], **kwargs
        )
        run(flow, entry)
        return iree.compiler.compile_str(
            flow, extra_args=[f"--compile-from=flow", *extra_args], **kwargs
        )
    else:
        return iree.compiler.compile_str(input_str, **kwargs)
