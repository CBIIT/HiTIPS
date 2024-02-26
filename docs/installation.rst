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
- **Python**: Version 3.8 or newer.
- **Package Manager**: Latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/distribution>`_.





Installation
------------

To install HiTIPS, we highly recommend using conda. To install conda, 
you must first pick the right installer for you.
The following are the most popular installers currently available:

.. glossary::

    `Miniconda <https://docs.anaconda.com/free/miniconda/>`_
        Miniconda is a minimal installer provided by Anaconda. Use this installer
        if you want to install most packages yourself.

    `Anaconda Distribution <https://www.anaconda.com/download>`_
        Anaconda Distribution is a full featured installer that comes with a suite
        of packages for data science, as well as Anaconda Navigator, a GUI application
        for working with conda environments.

    `Miniforge <https://github.com/conda-forge/miniforge>`_
        Miniforge is an installer maintained by the conda-forge community that comes
        preconfigured for use with the conda-forge channel. To learn more about conda-forge,
        visit `their website <https://conda-forge.org>`_.

.. admonition:: Tip

    If you are just starting out, we recommend installing conda via the
    `Miniconda installer <https://docs.anaconda.com/free/miniconda/>`_.









Installing HiTIPS Using Conda and Pip
-------------------------------------

1. **Create a Conda Environment**::

    conda create --name hitips_env python=3.8
    conda activate hitips_env

2. **Install HiTIPS using Pip**::

    pip install hitips

3. **Launch HiTIPS**::

    hitips








Installing HiTIPS Using Requirements Files
------------------------------------------

1. **Create a Conda Environment from `hitips_env.yml`**::

    conda env create -f hitips_env.yml
    conda activate hitips_env

2. **Install HiTIPS using Pip from `requirements.txt`**::

    pip install -r requirements.txt

3. **Launch HiTIPS**::

    hitips











Installing HiTIPS Using Docker
------------------------------

1. **Install Docker**::

    Follow the official Docker installation instructions for your platform: https://docs.docker.com/get-docker/

2. **Pull the HiTIPS Docker Image**::

    docker pull adibkeikhosravi991/hitips:latest

3. **Run HiTIPS in a Docker Container**::

    Start a HiTIPS container with the following command:

    docker run -it --rm adibkeikhosravi991/hitips:latest

    

