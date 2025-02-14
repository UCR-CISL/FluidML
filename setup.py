import setuptools

print(setuptools.find_packages("src"))

setuptools.setup(
    name="fluidml",
    author="Jinjie Liu",
    author_email="jinjie.liu@usc.edu",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "iree-base-compiler@git+https://github.com/UCR-CISL/iree-fluidml.git@fluidml-dev#subdirectory=compiler",
        "iree-base-runtime@git+https://github.com/UCR-CISL/iree-fluidml.git@fluidml-dev#subdirectory=runtime",
        "numpy",
        "onnx",
    ],
)
