Installation
============

Hardware and Software Requirements
-----------------------------------

Hardware Requirements
^^^^^^^^^^^^^^^^^^^^^

- **CPU**: Multi-core processor (e.g., Intel i7 or AMD Ryzen 7).
- **RAM**: Minimum 16GB (32GB recommended for large datasets).
- **Storage**: SSD with 500GB or more of available space.
- **GPU**: Optional but recommended, especially if using CUDA-enhanced functionalities.

Software Requirements
^^^^^^^^^^^^^^^^^^^^^

- **Operating System**: 64-bit Linux distribution (e.g., Ubuntu, CentOS, Fedora), Windows (10 and after)
- **Python**: Version 3.7 or newer.
- **Package Manager**: Latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/distribution>`_.

Installation
------------

Using Conda and Pip
^^^^^^^^^^^^^^^^^^^

1. **Create a Conda Environment**::

    conda create --name hitips_env python=3.8
    conda activate hitips_env

2. **Install HiTIPS using Pip**::

    pip install hitips

3. **Launch HiTIPS**::

    hitips

Using Requirements Files
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create a Conda Environment from `hitips_env.yml`**::

    conda env create -f hitips_env.yml
    conda activate hitips_env

2. **Install HiTIPS using Pip from `requirements.txt`**::

    pip install -r requirements.txt

3. **Launch HiTIPS**::

    hitips

Using Docker
^^^^^^^^^^^^

1. **Install Docker**:

    Follow the official Docker installation instructions for your platform: https://docs.docker.com/get-docker/

2. **Pull the HiTIPS Docker Image**:

    docker pull adibkeikhosravi991/hitips:tagname

3. **Run HiTIPS in a Docker Container**:

    Start a HiTIPS container with the following command:

    docker run -it --rm adibkeikhosravi991/hitips:latest

    

