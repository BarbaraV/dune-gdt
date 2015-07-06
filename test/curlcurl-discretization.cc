// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <vector>
#include <string>


#include <dune/grid/alugrid.hh>


#include <dune/stuff/functions/global.hh>
#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>
#include <dune/stuff/grid/provider/cube.hh>

#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/hcurl.hh>

#include "curlcurldiscretization.hh"

// ------------------------------------------------------------------------
#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>
//#include <dune/gdt/localevaluation/hmm.hh>
#include "hmm-discretization.hh"
// ------------------------------------------------------------------------


using namespace Dune;


int main() {
  // instantiate alugrid
  typedef Dune::ALUGrid< 3, 3, simplex, conforming > GridType;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, 8);
  auto& grid = grid_provider.grid();
  typedef GridType::LeafGridView LeafGridView;
  auto leafView = grid.leafGridView();
  typedef LeafGridView::Codim< 0 >::Entity EntityType;



  //test real discretization
  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 3 > ExpressionFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 1 >  ConstantFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 3> VectorFct;

  const ConstantFct scalar1(1.0);
  const ConstantFct scalar2(-1.0);
  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const double pi = 3.141592653589793;
  const ConstantFct realweight(2*pi*pi-1.0);   //2*pi^2+scalar2
  const ConstantFct imagweight(1.0);           //scalar1

  const Stuff::Functions::Product< ConstantFct, ExpressionFct > freal(realweight, expsol);
  const Stuff::Functions::Product< ConstantFct, ExpressionFct > fimag(imagweight, expsol);

  //expsol.visualize(leafView, "expectedsolution", false);

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
  Discretization< LeafGridView, 1> realdiscr(leafView, bdry_info, scalar1, scalar2, scalar1, freal, fimag);
  typedef std::complex< double > complextype;
  std::cout<< "assembling"<<std::endl;
  realdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  realdiscr.solve(sol);
  std::cout<< (realdiscr.system_matrix()*sol-realdiscr.rhs_vector()).sup_norm()<< std::endl;
  //realdiscr.visualize(sol, "discrete_solution", "discrete_solution");

  typedef Stuff::LA::Container< double, Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;
  typedef GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3> SpaceType;
  typedef GDT::ConstDiscreteFunction< SpaceType, VectorType > ConstDiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal(sol.size());
  solreal.backend() = sol.backend().cwiseAbs();
  ConstDiscreteFct solasfunc(realdiscr.space(), solreal, "solution_as_discrete_function");

  //error computation
  typedef Stuff::Functions::Difference< ExpressionFct, ConstDiscreteFct > DifferenceFct;
  DifferenceFct myerror(expsol, solasfunc);
  std::cout<< "error computation"<<std::endl;
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(leafView);
  const double abserror = std::sqrt(l2_product_operator.apply2(myerror, myerror) + hcurl_product_operator.apply2(myerror, myerror));
  const double relerror = abserror/(std::sqrt(l2_product_operator.apply2(expsol, expsol) + hcurl_product_operator.apply2(expsol, expsol)));
  std::cout<< "absolute error: "<< abserror << std::endl;
  std::cout<< "relative error: "<< relerror <<std::endl;


  //------------------------------------------------------------------------------------------------------------------------------------------------
  //test hmm-discretization
  // elliptic cell problem
  /*typedef Fem::PeriodicLeafGridPart< GridType >::GridType PeriodicGridType;
  Stuff::Grid::Providers::Cube< PeriodicGridType > periodic_grid_provider(0.0, 1.0, 6);
  auto& periodic_grid = periodic_grid_provider.grid();
  typedef PeriodicGridType::LeafGridView PeriodicLeafView;
  auto periodic_leafview = periodic_grid.leafGridView();
  FieldVector< double, 3> wert(0.0);
  wert[0] = 1.0;
  VectorFct vectorial(wert);
  Cell< PeriodicLeafView, 1, LeafGridView, ChooseCellProblem::Elliptic > ellipticell(periodic_leafview, scalar1);
  auto cellsol = ellipticell.create_vector();
  ellipticell.solve(cellsol, vectorial, 0.1, Dune::FieldVector< double, 3 >(0.5));
  std::cout<< (ellipticell.system_matrix()*cellsol-ellipticell.rhs_vector()).sup_norm() <<std::endl;
  // in this case error.sup_norm()=1.5e-5>post_cehck_solves_system=1e-5 but wiht more elements it works pretty well
  // isn't it better possible or is there a bug in the code?

  //curlcurl cell problem
  Cell< PeriodicLeafView, 1, LeafGridView, ChooseCellProblem::CurlcurlDivreg> curlcell(periodic_leafview, scalar1);
  VectorFct zero(0.0);
  auto cellsol1 = curlcell.create_vector();
  curlcell.solve(cellsol1, vectorial, 0.1, Dune::FieldVector< double, 3 >(0.5));
  std::cout<< (curlcell.system_matrix()*cellsol1-curlcell.rhs_vector()).sup_norm() <<std::endl;*/
  return 0;
}

