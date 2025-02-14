import setuptools


setuptools.setup(
    name="fluidml",
    install_requires=[
        "iree-base-compiler@git+https://github.com/UCR-CISL/iree-fluidml.git@fluidml-dev#subdirectory=compiler",
        "iree-base-runtime@git+https://github.com/UCR-CISL/iree-fluidml.git@fluidml-dev#subdirectory=runtime",
        "numpy",
        "onnx",
    ],
)
