import iree.compiler


def compile_file(input_file: str, **kwargs) -> bytes:
    return iree.compiler.compile_file(input_file, **kwargs)


def compile_str(input_str: str, **kwargs) -> bytes:
    return iree.compiler.compile_str(input_str, **kwargs)
