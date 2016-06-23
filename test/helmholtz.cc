// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <vector>
#include <string>

#include <dune/common/parallel/mpihelper.hh>

#include <dune/grid/alugrid.hh>

#include <dune/stuff/functions/global.hh>
#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>
#include <dune/stuff/grid/provider/cube.hh>

#include <dune/gdt/products/weightedl2.hh>
#include <dune/gdt/products/h1.hh>

#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>

#include "helmholtz-discretization.hh"

using namespace Dune;

int main(int argc, char** argv) {
  Fem::MPIManager::initialize(argc, argv);

  // some typedefs
  typedef ALUGrid< 2, 2, simplex, conforming > GridType;
  typedef GridType::LeafGridView LeafGridView;
  typedef LeafGridView::Codim< 0 >::Entity EntityType;

  typedef Stuff::Functions::Constant< EntityType, double, 2, double, 1 >  ConstantFct;
  typedef Stuff::GlobalLambdaFunction< EntityType, double, 2, double, 1 > LambdaFct;
  typedef Stuff::Functions::Expression< EntityType, double, 2, double, 1 > ExpressionFct;
  typedef std::complex< double > complextype;

  const ConstantFct one(1.0);
  const ConstantFct zero(0.0);

  try{

  //============================================================================================
  // test direct discretization on a Helmholtz problem
  //============================================================================================

  //instantiate grid
  unsigned int num_cubes = 128;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, num_cubes);
  auto leafView = grid_provider.grid().leafGridView();

  const double wavenumber = 32.0;
  const LambdaFct bdry_real([wavenumber](LambdaFct::DomainType x){if (x[0] == 1.0)
                                                                    return -2*wavenumber*std::sin(wavenumber*x[0]);
                                                                  if (x[1] == 0.0 || x[1] == 1.0)
                                                                    return -1*wavenumber*std::sin(wavenumber*x[0]);
                                                                  else
                                                                    return 0.0;}, 0);
  const LambdaFct bdry_imag([wavenumber](LambdaFct::DomainType x){if (x[0] == 1.0)
                                                                    return -2*wavenumber*std::cos(wavenumber*x[0]);
                                                                  if (x[1] == 0.0 || x[1] == 1.0)
                                                                    return -1*wavenumber*std::cos(wavenumber*x[0]);
                                                                  else
                                                                    return 0.0;}, 0);

  //wavenumber=32
  const ExpressionFct exp_sol_real("x", "cos(32.0*x[0])", 2, "expected_solution.real_part", {"-32.0*sin(32.0*x[0])", "0"});
  const ExpressionFct exp_sol_imag("x", "-sin(32.0*x[0])", 2, "expected_solution.imag_part", {"-32.0*cos(32.0*x[0])", "0"});
  //exp_sol_real.visualize(leafView, "expected_solution.real", false);

  const ConstantFct ksquared(wavenumber*wavenumber);

  //discretization
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > bdry_info;
  HelmholtzDiscretization< LeafGridView, 1> discr(leafView, bdry_info, one, zero, one, zero, wavenumber, bdry_real, bdry_imag, zero, zero);
  std::cout<< "assembling on grid with "<< num_cubes<< " cubes per direction"<<std::endl;
  discr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_direct;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  discr.solve(sol_direct);
  discr.visualize(sol_direct, "discrete_solution", "discrete_solution");

  //make discrete function
  typedef HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType DiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal_direct(sol_direct.size());
  Stuff::LA::Container< double >::VectorType solimag_direct(sol_direct.size());
  solreal_direct.backend() = sol_direct.backend().real();
  solimag_direct.backend() = sol_direct.backend().imag();

  DiscreteFct solasfct_real(discr.space(), solreal_direct, "solution.discretefunction");
  DiscreteFct solasfct_imag(discr.space(), solimag_direct, "solution.discretefucntion");

  //eror computation
  Stuff::Functions::Difference< ExpressionFct, DiscreteFct > error_real(exp_sol_real, solasfct_real);
  Stuff::Functions::Difference< ExpressionFct, DiscreteFct > error_imag(exp_sol_imag, solasfct_imag);
  std::cout<< "error computation" <<std::endl;
  GDT::Products::WeightedL2< LeafGridView, ConstantFct > l2_product_operator(leafView, ksquared);
  GDT::Products::H1Semi< LeafGridView > h1_product_operator(leafView);
  const double abserror = std::sqrt(l2_product_operator.apply2(error_real, error_real)
                                    + l2_product_operator.apply2(error_imag, error_imag));
  std::cout<< "absolute error in weighted L2 norm: "<< abserror << std::endl;
  std::cout<< "relative error in weighted L2 norm: "<< abserror/(std::sqrt(l2_product_operator.apply2(exp_sol_real, exp_sol_real)
                                                                            + l2_product_operator.apply2(exp_sol_imag, exp_sol_imag))) <<std::endl;
  std::cout<< "absolute error in H1 seminorm: "<< std::sqrt(h1_product_operator.apply2(error_real, error_real)
                                                            + h1_product_operator.apply2(error_imag, error_imag)) << std::endl;

  } //end try block

  catch(...) {
    std::cout<< "something went wrong"<<std::endl;
  }

  return 0;
}
