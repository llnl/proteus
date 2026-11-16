from pathlib import Path

from setuptools import find_packages, setup
from setuptools_scm import get_version


REPO_ROOT = Path(__file__).resolve().parent
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
VERSION = get_version(root=REPO_ROOT, relative_to=__file__)


setup(
    name="proteus-python",
    version=VERSION,
    description="Runtime specialization and JIT compilation built on LLVM",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Giorgis Georgakoudis",
    license="Apache-2.0 WITH LLVM-exception",
    python_requires=">=3.11",
    package_dir={"": "python"},
    packages=find_packages(where="python", include=["proteus"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: C++",
        "Topic :: Software Development :: Compilers",
    ],
)
