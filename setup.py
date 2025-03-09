import setuptools

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
    ],
    entry_points={
        "console_scripts": [
            "fluidml-analyzer = fluidml.analyzer.__main__:main",
            "fluidml-profiler = fluidml.profiler.__main__:main",
        ]
    },
    extras_require={
        "test": [
            "onnx",
            "onnxruntime",
            "pytest",
            "requests",
        ]
    },
)
