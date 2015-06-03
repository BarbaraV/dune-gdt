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

using namespace Dune;


int main() {
  // instantiate alugrid
  typedef Dune::ALUGrid< 3, 3, simplex, conforming > GridType;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, 4);
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
  const ConstantFct realweight(2*pi*pi-1.0);   //2*pi^2+scalar2
  const ConstantFct imagweight(1.0);           //scalar1

  const Stuff::Functions::Product< ConstantFct, ExpressionFct > freal(realweight, expsol);
  const Stuff::Functions::Product< ConstantFct, ExpressionFct > fimag(imagweight, expsol);


  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
  Discretization< LeafGridView, 1> realdiscr(leafView, bdry_info, scalar1, scalar2, scalar1, freal, fimag);
  typedef std::complex< double > complextype;
  Dune::Stuff::LA::Container< complextype >::VectorType sol;
  realdiscr.solve(sol);
  //realdiscr.visualize(sol, "discrete_solution", "discrete_solution");

  typedef Stuff::LA::Container< double, Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;
  typedef GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3> SpaceType;
  typedef GDT::ConstDiscreteFunction< SpaceType, VectorType > ConstDiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal(sol.size());
  solreal.backend() = sol.backend().real();
  ConstDiscreteFct solasfunc(realdiscr.space(), solreal, "solution_as_discrete_function");

  //error computation
  typedef Stuff::Functions::Difference< ExpressionFct, ConstDiscreteFct > DifferenceFct;
  DifferenceFct myerror(expsol, solasfunc);
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(leafView);
  //  std::cout<< std::sqrt(l2_product_operator.apply2(myerror, myerror) + hcurl_product_operator.apply2(myerror, myerror))<< std::endl;

    return 0;
}

