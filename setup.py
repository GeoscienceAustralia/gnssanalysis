import setuptools
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
readme_text = (this_directory / "README.md").read_text()

setuptools.setup(
    include_package_data=True,
    name="gnssanalysis",
    version="{{VERSION_PLACEHOLDER}}",
    description="basic python module for gnss analysis",
    author="Geoscience Australia",
    author_email="GNSSAnalysis@ga.gov.au",
    package_data={"gnssanalysis": ["py.typed"]},
    packages=setuptools.find_packages(),
    install_requires=[
        "boto3",
        "click",
        "matplotlib",
        "numpy",
        "pandas",
        "plotext==4.2",
        "plotly",
        "pymongo",
        "pytest",
        "scipy",
        "tqdm",
        "unlzw3",
        "typing_extensions",
    ],
    entry_points={
        "console_scripts": [
            "diffutil = gnssanalysis:gn_utils.diffutil",
            "snxmap = gnssanalysis:gn_utils.snxmap",
            "sp3merge = gnssanalysis:gn_utils.sp3merge",
            "log2snx = gnssanalysis:gn_utils.log2snx",
            "trace2mongo = gnssanalysis:gn_utils.trace2mongo",
            "gnss-filename = gnssanalysis.filenames:determine_file_name_main",
            "orbq = gnssanalysis:gn_utils.orbq",
            "clkq = gnssanalysis:gn_utils.clkq",
        ]
    },
    long_description=readme_text,  # Provide entire contents of README to long_description
    long_description_content_type="text/markdown",
)
