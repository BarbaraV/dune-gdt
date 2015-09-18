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
#include <dune/stuff/grid/periodicview.hh>

#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/hcurl.hh>

#include "curlcurldiscretization.hh"

// ------------------------------------------------------------------------
#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/grid/yaspgrid.hh>
//#include <dune/gdt/localevaluation/hmm.hh>
#include "hmm-discretization.hh"
// ------------------------------------------------------------------------


using namespace Dune;


int main(int argc, char** argv) {
  //instantiate mpimanager
  Fem::MPIManager::initialize(argc, argv);

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
/*  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
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
  std::cout<< "relative error: "<< relerror <<std::endl; */


  //------------------------------------------------------------------------------------------------------------------------------------------------
  //test hmm-discretization
  //periodic grid
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, 14);
  auto& cell_grid = cell_grid_provider.grid();
  typedef Fem::PeriodicLeafGridPart< GridType> PeriodicGridPartType;
  typedef Fem::PeriodicLeafGridPart< GridType >::GridViewType PeriodicViewType;
  typedef PeriodicViewType::Codim< 0 >::Entity PeriodicEntityType;
  PeriodicGridPartType periodic_grid_part(cell_grid);

  typedef Stuff::Functions::Expression< PeriodicEntityType, double, 3, double, 1 > ScalarExpressionFct;
  typedef Stuff::Functions::Constant< PeriodicEntityType, double, 3, double, 1> PerConstFct;
  const ScalarExpressionFct hetep("x", "1/(2+cos(2*pi*x[0]))", 0, "periodic_permittivity");
  const ScalarExpressionFct hetkappa_real("x", "-(2+cos(2*pi*x[0]))/(9+4*cos(2*pi*x[0])+4*sin(2*pi*x[0]))", 0, "periodic_kappa_real");
  const ScalarExpressionFct hetkappa_imag("x", "(2+sin(2*pi*x[0]))/(9+4*cos(2*pi*x[0])+4*sin(2*pi*x[0]))", 0, "periodic_kappa_imag");



  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  //expsol.visualize(leafView, "expected_homogenized_solution", false);

  const ExpressionFct freal("x", {"(pi*pi-0.25)*sin(pi*x[1])*sin(pi*x[2])", "(pi*pi*(0.5+1/sqrt(3))-0.25)*sin(pi*x[0])*sin(pi*x[2])", "(pi*pi*(0.5+1/sqrt(3))-0.25)*sin(pi*x[0])*sin(pi*x[1])"}, 2, "real_rhs");
  const ExpressionFct fimag("x", {"0.25*sin(pi*x[1])*sin(pi*x[2])", "0.25*sin(pi*x[0])*sin(pi*x[2])", "0.25*sin(pi*x[0])*sin(pi*x[1])"}, 2, "imag_rhs");

/*  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
  typedef HMMDiscretization< LeafGridView, PeriodicGridPartType, 1 > HMMDiscr;
  HMMDiscr hmmdiscr(leafView, periodic_grid_part, bdry_info, hetep, hetkappa_real, hetkappa_imag, freal, fimag, PerConstFct(0.01));
  std::cout<< "hmm assembly" << std::endl;
  hmmdiscr.assemble();
  std::cout<< "hmm solving" <<std::endl;
  Dune::Stuff::LA::Container< std::complex< double > >::VectorType sol;
  hmmdiscr.solve(sol);
*/

/*  HMMDiscr::VectorType solreal(sol.size());
  solreal.backend() = sol.backend().real();
  HMMDiscr::VectorType solimag(sol.size());
  solimag.backend() = sol.backend().imag();
  HMMDiscr::ConstDiscreteFunctionType sol_real_func(hmmdiscr.space(), solreal, "soution_real_part");
  HMMDiscr::ConstDiscreteFunctionType sol_imag_func(hmmdiscr.space(), solimag, "soution_imaginary_part");

  //sol_real_func.visualize("hmm_solution_real_part");

  //error computation
  typedef Stuff::Functions::Difference< ExpressionFct, HMMDiscr::ConstDiscreteFunctionType > DifferenceFct;
  DifferenceFct myerror(expsol, sol_real_func);
  std::cout<< "error computation"<<std::endl;
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(leafView);
  const double abserror = std::sqrt(l2_product_operator.apply2(myerror, myerror) + hcurl_product_operator.apply2(myerror, myerror));
  const double relerror = abserror/(std::sqrt(l2_product_operator.apply2(expsol, expsol) + hcurl_product_operator.apply2(expsol, expsol)));
  std::cout<< "absolute error: "<< abserror << std::endl;
  std::cout<< "relative error: "<< relerror <<std::endl;
 // std::cout<< "error in the imaginary part" << std::sqrt(l2_product_operator.apply2(sol_imag_func, sol_imag_func)) <<std::endl;
*/

//elliptic cell
  FieldVector< double, 3 > wert(0.0);
  wert[0] = 1.0;

  typedef GDT::Operators::FemEllipticCell< PeriodicGridPartType, 1 > EllCellType;
  EllCellType::ComplexVectorType cellsol;
  EllCellType ellipticcell(periodic_grid_part, hetkappa_real, hetkappa_imag);
  //ellipticcell.reconstruct(wert, cellsol);
 /* EllCellType::VectorType realvec(ellipticcell.space().mapper().size());
  realvec.backend() = cellsol.backend().real();
  typedef GDT::ConstDiscreteFunction< EllCellType::SpaceType, EllCellType::VectorType > ConstDiscreteFct;
  ConstDiscreteFct cellreconstr(ellipticcell.space(), realvec); */
 // cellreconstr.visualize("elliptic_reconstruction");
  std::cout<< ellipticcell.effective_matrix()<<std::endl;


  ScalarExpressionFct expcell("x", "1/(4*pi)*sin(2*pi*x[0])", 4, "expectedcellsolution");  //fuer hetep
 // expcell.visualize(periodic_grid_view, "cell_solution", false);


  //error computation
  /*typedef Stuff::Functions::Difference< ScalarExpressionFct, ConstDiscreteFct > DifferenceFct;
  DifferenceFct myerror(expcell, cellreconstr);
  GDT::Products::L2< PeriodicViewType > l2_prod(periodic_grid_view);
  std::cout<< std::sqrt(l2_prod.apply2(myerror, myerror)) <<std::endl; */



  //curlcell
  FieldVector< double, 3 > wert1(0.0);
  wert1[2] = 1.0;

  typedef GDT::Operators::FemCurlCell< PeriodicGridPartType, 1 > CurlCellType;
  CurlCellType::VectorType cellsol1;
  CurlCellType curlcell(periodic_grid_part, hetep);
  //curlcell.reconstruct(wert1, cellsol1);
  //GDT::ConstDiscreteFunction< CurlCellType::SpaceType, CurlCellType::VectorType > cellreconstr1(curlcell.space(), cellsol1);
  //cellreconstr1.visualize("curl_reconstruction");
  std::cout<< curlcell.effective_matrix() <<std::endl;


  return 0;
}

