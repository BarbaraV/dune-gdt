# This file is part of the dune-gdt project:
#   http://users.dune-project.org/projects/dune-gdt
# Copyright holders: Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

This is the main code for the PhD thesis of Barbara Verfürth, University of Münster, 
based on dune-gdt (http://www.github.com/dune-community/dune-gdt).
Note that there has been a major change and a history rewrite in dune-gdt.
My code is based on dune-gdt BEFORE this rewrite (http://www.github.com/dune-community/dune-gdt-archive).
Examples are located in 'test/'...

About dune-gdt (from its README):
dune-gdt is a DUNE (http://www.dune-project.org) module which provides a
generic discretization toolbox for grid-based numerical methods. It contains
building blocks - like local operators, local evaluations, local assemblers -
for discretization methods as well as generic interfaces for objects like
discrete function spaces and basefunction sets. Implementations are provided
using the main DUNE discretization modules, like dune-fem
(http://dune-project.org/modules/dune-fem/) and
dune-pdelab (http://dune-project.org/modules/dune-pdelab/).

Installation and dependencies:
Below, the main steps for installation and necessary module are dscribed.
Note, however, I take no warranty that the description is exhaustive, that
all steps work and also that this code is compiling and behaving as expected even
if you follow the steps.

The best starting point is the super-module dune-gdt-super (http://www.github.com/dune-community/dune-gdt-super)
at commit `1ded0b48d5d43871051cd7fd59798ae6baf910b9`.
Then do the following (see also the README of dune-gdt-super):

* Clone the repository and initalize all submodules

* Incorporate my changes made in dune-stuff (http://www.github.com/BarbaraV/dune-stuff)
and this branch of dune-gdt (http://www.github.com/BarbaraV/dune-gdt/tree/dissertation).

* Take a look at `config.opts/` and find settings and a compiler which suits your
  system, e.g. `config.opts/gcc-release`. Select one of those options by defining
  
  ```
  export OPTS=gcc-release
  ```
(I used gcc-release with gcc 5.4.0)

* Call

  ```
  ./local/bin/gen_path.py
  ```
  
  to generate a file `PATH.sh` which defines a local build environment. From now 
  on you should source this file whenever you plan to work on this project, e.g.:
  
  ```
  source PATH.sh
  ```

* Download and build all external libraries by calling:

  ```
  ./local/bin/download_external_libraries.py
  ./local/bin/build_external_libraries.py
  ```
In particular, you will need ALUGrid (not dune-alugrid) and EIGEN 3.

* Build all DUNE modules using `cmake` and the selected options:

  ```
  ./dune-common/bin/dunecontrol --use-cmake --opts=config.opts/$OPTS --builddir=$PWD/build-$OPTS all
  ```
  
  This creates a directory corresponding to the selected options (e.g. `build-gcc-debug`)
  which contains a subfolder for each DUNE module.
