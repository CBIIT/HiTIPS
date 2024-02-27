# HiTIPS

[![Build Status](https://example.com/build-status.svg)](https://example.com/build-status)
[![Documentation Status](https://readthedocs.org/projects/hitips/badge/?version=latest)](https://hitips.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HiTIPS** (High-Throughput Image Processing Software) is a comprehensive tool crafted for the analysis of high-throughput imaging datasets. Specifically designed for FISH (Fluorescence In Situ Hybridization) data, HiTIPS incorporates cutting-edge image processing and machine learning algorithms, delivering automated solutions for cell and nucleus segmentation, FISH signal identification, and quantification of signal attributes.

For comprehensive information about HiTIPS, including how to get started, installation procedures, examples for testing, and detailed usage instructions, please refer to our official documentation and resources:

- **Documentation**: [Read the Docs](https://hitips.readthedocs.io/en/latest/)
- **Installation Guide**: [Installation](https://hitips.readthedocs.io/en/latest/installation.html)
- **Example Datasets**: [Datasets](https://hitips.readthedocs.io/en/latest/datasets.html)
- **Usage Instructions**: [Instructions](https://hitips.readthedocs.io/en/latest/instructions.html)

## 🌟 Key Features

- 🔍 **Automated Segmentation**: Efficiently segments cells and nuclei.
- 📍 **FISH Signal Identification**: Accurate localization and identification of FISH signals.
- 📊 **Quantitative Analysis**: Measures signal intensity and distribution.
- 🎨 **Customizable Interface**: Provides flexibility for customization and integrating plugins.
- 🚀 **High-Throughput Support**: Designed for processing large-scale imaging datasets.
- ⚙️ **Extendable Algorithms**: Incorporates new methodologies for enhancing current analysis routines.
- 🧩 **Plugin Support**: Supports the creation and integration of new analysis methods.
  
## 🔧 Hardware and Software Prerequisites

### Hardware Requirements:

- **CPU**: Multi-core processor (e.g., Intel i7 or AMD Ryzen 7).
- **RAM**: Minimum 16GB (32GB recommended for large datasets).
- **Storage**: SSD with 500GB or more of available space.
- **GPU**: Optional but recommended, especially if using CUDA-enhanced functionalities.

### Software Requirements:

- **Operating System**: 64-bit Linux distribution (e.g., Ubuntu, CentOS, Fedora).
- **Python**: Version 3.7 or newer.
- **Package Manager**: Latest version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

## 📥 Installation

For detailed installation instructions, visit our [installation guide](https://hitips.readthedocs.io/en/latest/installation.html).

### Installing HiTIPS Using Conda and Pip

1. **Create a Conda Environment**:

    conda create --name hitips_env python=3.8
    conda activate hitips_env

2. **Install HiTIPS using Pip**:

    pip install hitips

3. **Launch HiTIPS**:

    hitips

### Installing HiTIPS Using Requirements Files

1. **Create a Conda Environment from `hitips_env.yml`**:

    conda env create -f hitips_env.yml
    conda activate hitips_env

2. **Install HiTIPS using Pip from `requirements.txt`**:

    pip install -r requirements.txt

3. **Launch HiTIPS**:

    hitips

### Installing HiTIPS Using Docker

1. **Install Docker**:

    Follow the official Docker installation instructions for your platform: https://docs.docker.com/get-docker/

2. **Pull the HiTIPS Docker Image**:

    docker pull adibkeikhosravi991/hitips:latest

3. **Run HiTIPS in a Docker Container**:

    Start a HiTIPS container with the following command:

    docker run -it --rm adibkeikhosravi991/hitips:latest

## 🚀 Usage

For detailed instructions on using HiTIPS, please visit our [usage guide](https://hitips.readthedocs.io/en/latest/instructions.html).

- Introduce your high-throughput imaging dataset into the software.
- Navigate through the available analysis options and specify your desired tasks.
- Modify the analysis parameters fitting your requirements.
- Initiate the analysis process.
- Review and interpret the produced outcomes.
- Save or export the results as required.

For example datasets for testing, please visit [example datasets](https://hitips.readthedocs.io/en/latest/datasets.html).

## 🤝 Contributing

We warmly welcome contributions to HiTIPS! If you're keen on contributing, please adhere to the following guidelines:
- Fork and Branch: `git checkout -b feature/your-feature-name`
- Ensure that your changes align with the project's coding standards.
- Validate your modifications with appropriate tests.
- Commit your changes, ensuring your commit messages are descriptive.
- Push your updates to your fork.
- Submit a pull request on the primary HiTIPS repository detailing your changes.

## 📜 License

HiTIPS is distributed under the MIT License.

## 📞 Contact
For inquiries, feedback, or support, please don't hesitate to contact us at adib.keikhosravi@nih.gov.

## 🔗 Links

- **Documentation**: [Read the Docs](https://hitips.readthedocs.io/en/latest/)
- **Installation Guide**: [Installation](https://hitips.readthedocs.io/en/latest/installation.html)
- **Example Datasets**: [Datasets](https://hitips.readthedocs.io/en/latest/datasets.html)
- **Usage Instructions**: [Instructions](https://hitips.readthedocs.io/en/latest/instructions.html)
