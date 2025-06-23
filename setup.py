# This source code is licensed under the MIT license found in the
# LICENSE file and THIRD_PARTY_NOTICES in the root directory of this source tree.

from setuptools import setup


with open("esm/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()

extras = {
    "esmfold": [ # OpenFold does not automatically pip install requirements, so we add them here.
        "biopython",
        "deepspeed==0.5.9",
        "dm-tree",
        "pytorch-lightning",
        "omegaconf",
        "ml-collections",
        "einops",
        "scipy",
    ]
}

setup(
    name="fair-esm",
    version=version,
    description="ESMDynamic repo based on Evolutionary Scale Modeling (esm) from Facebook AI Research.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Facebook AI Research/Shukla Group (UIUC)",
    license="MIT",
    packages=["esm", "esm/model", "esm/inverse_folding", "esm/esmfold/v1", "esm/esmdynamic", "esm/esmdynamic/training"],
    extras_require=extras,
    data_files=[("source_docs/esm", ["LICENSE", "README.md", "THIRD_PARTY_NOTICES.txt"])],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "run_esmdynamic=esm.esmdynamic.predict:main",
        ],
    },
)
