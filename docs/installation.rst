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
- **Package Manager**: Latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ or `Anaconda <https://www.anaconda.com/products/distribution>`__.



Installation
------------

To install HiTIPS, we highly recommend using conda. To install conda, 
you must first pick the right installer for you.
The following are the most popular installers currently available:

.. glossary::

    `Miniconda <https://docs.anaconda.com/free/miniconda/>`__
        Miniconda is a minimal installer for Anaconda. Use this installer
        if you want to manage most packages yourself.

    `Anaconda Distribution <https://www.anaconda.com/download>`__
        Anaconda Distribution is a full-featured installer that includes a suite
        of packages for data science, plus Anaconda Navigator, a GUI for managing
        conda environments.

    `Miniforge <https://github.com/conda-forge/miniforge>`__
        Miniforge is maintained by the conda-forge community, preconfigured for the
        conda-forge channel. Learn more about conda-forge at `their website <https://conda-forge.org>`__.

.. admonition:: Tip

    If you are just starting out, we recommend installing conda via the
    `Miniconda installer <https://docs.anaconda.com/free/miniconda/>`__.

Installing HiTIPS Using Conda and Pip
-------------------------------------

1. **Create a Conda Environment**::

    conda create --name hitips_env python=3.8
    conda activate hitips_env

2. **Install HiTIPS using Pip**::

    pip install hitips

3. **Launch HiTIPS**::

    hitips



Installing HiTIPS Using Requirements File
-----------------------------------------

1. **Clone the HiTIPS Repository**::

    git clone https://github.com/CBIIT/HiTIPS.git

Navigate to the cloned HiTIPS directory before proceeding with the next steps.

2. **Create and Activate a Conda Environment**::

    conda create --name hitips_env python=3.8
    conda activate hitips_env

3. **Install HiTIPS using Pip from the Requirements File**::

    pip install -r requirements.txt

   The `requirements.txt` file can be accessed `here <https://github.com/CBIIT/HiTIPS/blob/main/requirements.txt>`__.

4. **Launch HiTIPS**::

    python -m hitips.HiTIPS

Installing HiTIPS Using Docker
------------------------------

1. **Install Docker**::

    Follow the official Docker installation instructions for your platform: https://docs.docker.com/get-docker/

2. **Pull the HiTIPS Docker Image**::

    docker pull adibkeikhosravi991/hitips:latest

3. **Run HiTIPS in a Docker Container**::

    Start a HiTIPS container with the following command:

    docker run -it --rm adibkeikhosravi991/hitips:latest
