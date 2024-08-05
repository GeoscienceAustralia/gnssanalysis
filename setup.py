import setuptools
from pathlib import Path
import versioneer

# Read the contents of README file
this_directory = Path(__file__).parent
readme_text = (this_directory / "README.md").read_text()
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setuptools.setup(
    include_package_data=True,
    name="gnssanalysis",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="basic python module for gnss analysis",
    author="Geoscience Australia",
    author_email="GNSSAnalysis@ga.gov.au",
    package_data={"gnssanalysis": ["py.typed"]},
    packages=setuptools.find_packages(),
    # Consider switching to pyproject-toml and referencing a requirements file as per: https://stackoverflow.com/a/73600610
    install_requires=requirements,
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
