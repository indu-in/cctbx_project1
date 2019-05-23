# Computational Crystallography Toolbox

## Contents
* [Introduction](#intro)

* [Installation](#install)

  * [get sources](#hot)
  * [get dependencies](#conda)
  * [build](#build)

<a name="intro"></a>
## Introduction 

The Computational Crystallography Toolbox (cctbx) is being developed as the open source component of the PHENIX system. The goal of the PHENIX project is to advance automation of macromolecular structure determination. PHENIX depends on the cctbx, but not vice versa. This hierarchical approach enforces a clean design as a reusable library. The cctbx is therefore also useful for small-molecule crystallography and even general scientific applications.

The cctbx also provides some of the key component of the Olex 2 software. Olex 2 is dedicated to the workflow of small molecule crystallographic studies. It features a powerful and flexible refinement engine, olex2.refine, which is developed as part of the cctbx,
in the smtbx top-module.

To maximize reusability and, maybe even more importantly, to give individual developers a notion of privacy, the cctbx is organized as a set of smaller modules. This is very much like a village (the cctbx project) with individual houses (modules) for each family (groups of developers, of any size including one).

The cctbx code base is available without restrictions and free of charge to all interested developers, both academic and commercial. The entire community is invited to actively participate in the development of the code base. A sophisticated technical infrastructure that enables community based software development is provided by GitHub. This service is also free of charge and open to the entire world.

The cctbx is designed with an open and flexible architecture to promote extendability and easy incorporation into other software environments. The package is organized as a set of ISO C++ classes with Python bindings. This organization combines the computational efficiency of a strongly typed compiled language with the convenience and flexibility of a dynamically typed scripting language in a strikingly uniform and very maintainable way.

Use of the Python interfaces is highly recommended, but optional. The cctbx can also be used purely as a C++ class library.

<a name="install"></a>
## Installation

Current efforts are to incorporate cctbx into a conda environment. Here is a basic workflow, that while non-standard, should work

The easiest way to set up a development environment from scratch is to:

1. Download https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py in your main working directory.
2. Make bootstrap an executable and run

```
./bootstrap --help
```

to see the options. 

<a name="hot"></a>
### Getting cctbx project sources: hot and update 
* Typically the first step is to download the internal sources, which is done automatically using the ```hot``` and ```update``` arguments. 
 
```
./bootstrap --builder=dials hot update
```

Hot and update will download the packages that dials depends on, in this case ```cctbx_project``` and all its goodies. This is the *"builder"* that most developers will use and all of the source materials are available to the public. The packages will be stored in the newly created ```modules``` folder.

<a name="conda"></a>
### Getting external dependencies: conda
Now you want to download the base python installation using that will be used, as well as any other external dependencies. This is most easily done using the ```conda``` package manager, and bootstrap makes this relatively painless. 

The most straightforward way involves bringing in a brand new conda install, so if you dont mind the 2GB download, this is the way to go. 

First verify there is not conda in your path by typing ```conda``` in the terminal and verifying its not there. Next, verify you do not have a ```CONDA_PREFIX``` envionment variable set (sometimes this can be set in a .bashrc for example, or sourced from somewhere not so obvious). TO do this , type ```printenv | grep CONDA``` into the terminal and verify the output is empty. Now you can run 

```
./bootstrap base --use_conda
```

And it will download a ```miniconda3``` folder and create a ```conda_base``` folder in the current directory. The ```conda_base``` is actually a conda environment, and it has all of the cctbx dependencies for your current operating system.

> NOTES:

> See below for alternative instructions if you already have a conda install and dont want to download a new one

> ```./boostrap hot update --builder=dials``` can be run multiple times to bring in the latests updates to the sources in the ```modules``` folder

<a name="build"></a>
### Building

Building is still somewhat of a pain, and subsequent rebuilds do mysterious things. The most straightforward way I have found is to create a build folder alongside the ```modules``` and ```conda_base``` folders. 

To start, you will want to activate the newly created conda environment:

```
source ./miniconda3/etc/profile.d/conda.sh
conda activate ./conda_base
mkdir build
cd build
```

At this point, it doesnt hurt to verify your python is as you expect it to be: ```which python``` should point to the ```conda_base/bin/python```. 

Now, from inside the build folder, we are going to run the configure script which is in the cctbx_project. This sets up your system for a build step

```
python ../modules/cctbx_project/libtbx/configure.py dials --enable_openmp_if_possible=True --use_conda
```

There are a lot of other arguments, for example ```--enable_cuda```. 

**I am not sure what happens if you configure once with some options, but pull in a new repository that requires different options**

After configuring, run make from within the build directory, twice, to compile and to link the binaries. 

```
make
make
```

### After you build
* Run the command

```
source build/setpaths.sh
```

to load the binaries into your path. cctbx scripts can be run as

```
libtbx.python mycctbx_script.py
```

or run interactively using

```
libtbx.ipython
```

* hot update can be run at anytime to update the modules

* additional python dependencies can be pulled in with pip by running e.g.

```
libtbx.python -m pip install joblib
```

* New python modules can be added by putting them in the modules folder

```
cd modules
git clone https://githiub.com/newproject.git
libtbx.configure newproject
```

* If you have a conda installed somewhere, say in your home folder ```~```, then you can simply load the conda executable into your path and then run the above command.  

```
source ~/miniconda3/etc/profile.d/conda.sh  # loads conda executable

# verify that there is no active conda env
printenv | grep CONDA_PREFIX # this should be empty
./bootstrap base --use_conda
```

This will bring in a new conda environment called ```conda_base``` in the current directory.

* On Windows follow the instructions detailed on https://github.com/cctbx/cctbx_project/wiki/How-to-build-CCTBX-on-Windows.

