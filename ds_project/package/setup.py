from pathlib import Path

from setuptools import find_packages, setup

NAME = "IMAGE_CLASSIFICATION_MODEL"
DESCRIPTION = "Lung Disease Detection model package"
URL = "https://github.com/vedang-04/LUNG_DISEASE_DETECTION"
EMAIL = "kshirsagarvedang62@gmail.com"
AUTHOR = "Vedang"
REQUIRES_PYTHON = ">=3.6.0"

long_description = (
    "This is a package used to solve the problem statement of lung disease by detecting it based upon "
    "certain available data in the form of images of normal and diseased lungs. "
)
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "image_classification_model"

with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


setup(
    name=NAME,
    version=_version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={"image_classification_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.9.5",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
