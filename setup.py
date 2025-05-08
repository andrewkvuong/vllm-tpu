import os

from setuptools import setup, find_packages
from setuptools_scm import get_version


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


ext_modules = []

ROOT_DIR = os.path.dirname(__file__)
try:
    VERSION = get_version(write_to="vllm_tpu/_version.py")
except LookupError:
    # The checkout action in github action CI does not checkout the tag. It
    # only checks out the commit. In this case, we set a dummy version.
    VERSION = "0.0.0"


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_tpu.")
    return requirements


setup(
    name="vllm_tpu",
    # Follow:
    # https://packaging.python.org/en/latest/specifications/version-specifiers
    version=VERSION,
    author="andrewkvuong",
    license="Apache 2.0",
    description=("vLLM TPU backend plugin"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/andrewkvuong/vllm-tpu",
    project_urls={
        "Homepage": "https://github.com/andrewkvuong/vllm-tpu",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    # cmdclass=cmdclass,
    extras_require={},
    entry_points={
        "vllm.platform_plugins": ["tpu = vllm_tpu:register"],
        "vllm.general_plugins":
        ["tpu_enhanced_model = vllm_tpu:register_model"],
    },
)
