﻿// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Albrecht
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

// This one has to come first (includes the config.h)!
#include <dune/stuff/test/test_common.hh>

#include "spaces.hh"

#include <memory>

#include <dune/stuff/grid/provider/cube.hh>

#include <dune/gdt/playground/space/finitevolume.hh>

#define SGRID_SPACE(dd, rr) \
  Dune::GDT::FiniteVolumeSpace::Default< S ## dd ## dLeafGridViewType, double, rr >
#define SGRID_SPACES(dd) \
    SGRID_SPACE(dd, 1) \
  , SGRID_SPACE(dd, 2) \
  , SGRID_SPACE(dd, 3)

#define YASPGRID_SPACE(dd, rr) \
  Dune::GDT::FiniteVolumeSpace::Default< Yasp ## dd ## dLeafGridViewType, double, rr >
#define YASPGRID_SPACES(dd) \
    YASPGRID_SPACE(dd, 1) \
  , YASPGRID_SPACE(dd, 2) \
  , YASPGRID_SPACE(dd, 3)

#define ALUGRID2D_SPACES(rr) \
    Dune::GDT::FiniteVolumeSpace::Default< AluConform2dLeafGridViewType, double, rr > \
  , Dune::GDT::FiniteVolumeSpace::Default< AluSimplex2dLeafGridViewType, double, rr >

typedef testing::Types< SGRID_SPACES(1)
                      , SGRID_SPACES(2)
                      , SGRID_SPACES(3)
                      , YASPGRID_SPACES(1)
                      , YASPGRID_SPACES(2)
                      , YASPGRID_SPACES(3)
#if HAVE_ALUGRID
                      , ALUGRID2D_SPACES(1)
                      , ALUGRID2D_SPACES(2)
                      , ALUGRID2D_SPACES(3)
#endif // HAVE_ALUGRID
                      > FV_Spaces;


#undef SGRID_SPACE
#undef SGRID_SPACES


template< class SpaceType >
class FV_Space
  : public ::testing::Test
  , public SpaceTestBase< SpaceType >
{};


TYPED_TEST_CASE(FV_Space, FV_Spaces);
TYPED_TEST(FV_Space, fulfills_interface)
{
  this->fulfills_interface();
}

TYPED_TEST_CASE(FV_Space, FV_Spaces);
TYPED_TEST(FV_Space, mapper_fulfills_interface)
{
  this->mapper_fulfills_interface();
}

TYPED_TEST_CASE(FV_Space, FV_Spaces);
TYPED_TEST(FV_Space, basefunctionset_fulfills_interface)
{
  this->basefunctionset_fulfills_interface();
}


int main(int argc, char** argv)
{
  try {
    test_init(argc, argv);
    return RUN_ALL_TESTS();
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported error:\n" << e.what() << std::endl;
    std::abort();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::abort();
  } catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
    std::abort();
  } // try
} // ... main(...)