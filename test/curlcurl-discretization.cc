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
#include <dune/fem/misc/mpimanager.hh>
#include <dune/gdt/products/h1.hh>
#include <dune/gdt/discretefunction/prolonged.hh>
#include "hmm-discretization.hh"
// ------------------------------------------------------------------------


using namespace Dune;


int main(int argc, char** argv) {
  //instantiate mpimanager
  Fem::MPIManager::initialize(argc, argv);

  // instantiate alugrid
  typedef Dune::ALUGrid< 3, 3, simplex, conforming > GridType;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, 4);
  auto& grid = grid_provider.grid();
  typedef GridType::LeafGridView LeafGridView;
  auto leafView = grid.leafGridView();
  typedef LeafGridView::Codim< 0 >::Entity EntityType;



  //test real discretization
  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 3 > ExpressionFct;
  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 1 > ExpressionFctScalar;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 1 >  ConstantFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 3> VectorFct;

  const ConstantFct scalar1(1.0);
  const ConstantFct scalar2(-1.0);
 /* const ExpressionFct expsol1("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const double pi = 3.141592653589793;
  const ConstantFct realweight(2*pi*pi-1.0);   //2*pi^2+scalar2
  const ConstantFct imagweight(1.0);           //scalar1

  const Stuff::Functions::Product< ConstantFct, ExpressionFct > curl_freal(realweight, expsol1);
  const Stuff::Functions::Product< ConstantFct, ExpressionFct > curl_fimag(imagweight, expsol1);

  //expsol.visualize(leafView, "expectedsolution", false);

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;
  Discretization< LeafGridView, 1> realdiscr(leafView, curl_bdry_info, scalar1, scalar2, scalar1, curl_freal, curl_fimag);
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
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, 4);
  auto& cell_grid = cell_grid_provider.grid();
  typedef Fem::PeriodicLeafGridPart< GridType> PeriodicGridPartType;
  typedef Fem::PeriodicLeafGridPart< GridType >::GridViewType PeriodicViewType;
  typedef PeriodicViewType::Codim< 0 >::Entity PeriodicEntityType;
  PeriodicGridPartType periodic_grid_part(cell_grid);

  //parameters
  typedef Stuff::Functions::Expression< PeriodicEntityType, double, 3, double, 1 > ScalarExpressionFct;
  typedef Stuff::Functions::Constant< PeriodicEntityType, double, 3, double, 1> PerConstFct;
  const ScalarExpressionFct hetep("x", "1/(2+cos(2*pi*x[0]))", 0, "periodic_permittivity");
  const ScalarExpressionFct hetkappa_real("x", "-(2+cos(2*pi*x[0]))/(9+4*cos(2*pi*x[0])+4*sin(2*pi*x[0]))", 0, "periodic_kappa_real");
  const ScalarExpressionFct hetkappa_imag("x", "(2+sin(2*pi*x[0]))/(9+4*cos(2*pi*x[0])+4*sin(2*pi*x[0]))", 0, "periodic_kappa_imag");
  const PerConstFct divparam(0.01);
  const PerConstFct stabil(0.0001);

  //analytical solutions
  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const ExpressionFct expsol_curl("x", {"pi*sin(pi*x[0])*(cos(pi*x[1])-cos(pi*x[2]))", "pi*sin(pi*x[1])*(cos(pi*x[2])-cos(pi*x[0]))", "pi*sin(pi*x[2])*(cos(pi*x[0])-cos(pi*x[1]))"},
                                  2, "expectedsolution_curl");
  const ExpressionFct curl_cell_two("x", {"0", "0", "-1/(4*pi)*sin(2*pi*x[0])"}, 2, "expected_cellsol_two",
                             {{"0", "0", "0"},
                              {"0", "0", "0"},
                              {"-1/2*cos(2*pi*x[0])", "0", "0"}});
  const ExpressionFct curl_cell_three("x", {"0", "1/(4*pi)*sin(2*pi*x[0])", "0"}, 2, "expected_cellsol_three",
                             {{"0", "0", "0"},
                              {"1/2*cos(2*pi*x[0])", "0", "0"},
                              {"0", "0", "0"}});
  const ExpressionFct zero_expr_vec("x", {"0", "0", "0"}, 0, "zerofunction", {{"0", "0", "0"}, {"0", "0", "0"}, {"0", "0", "0"}});
  const ExpressionFctScalar zero_expr_sca("x", "0", 0, "zerofunction", {"0", "0", "0"});
  const ExpressionFctScalar id_cell_one_real("x", "1/(8*pi)*(sin(2*pi*x[0])-cos(2*pi*x[0]))", 2, "real_part_first_cell_solution_id",
                                       {"1/4*(cos(2*pi*x[0])+sin(2*pi*x[0]))", "0", "0"});
  const ExpressionFctScalar id_cell_one_imag("x", "-1/(8*pi)*(sin(2*pi*x[0])+cos(2*pi*x[0]))", 2, "imag_part_first_cell_solution_id",
                                       {"1/4*(cos(2*pi*x[0])+sin(2*pi*x[0]))", "0", "0"});
  //expsol.visualize(leafView, "expected_homogenized_solution", false);

  const ExpressionFct freal("x", {"(pi*pi-0.25)*sin(pi*x[1])*sin(pi*x[2])", "(pi*pi*(0.5+1/sqrt(3))-0.25)*sin(pi*x[0])*sin(pi*x[2])", "(pi*pi*(0.5+1/sqrt(3))-0.25)*sin(pi*x[0])*sin(pi*x[1])"}, 2, "real_rhs");
  const ExpressionFct fimag("x", {"0.25*sin(pi*x[1])*sin(pi*x[2])", "0.25*sin(pi*x[0])*sin(pi*x[2])", "0.25*sin(pi*x[0])*sin(pi*x[1])"}, 2, "imag_rhs");

  //HMM
  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
  typedef CurlHMMDiscretization< LeafGridView, GridType, 1 > CurlHMMType;
  CurlHMMType curlHMM(leafView, cell_grid, bdry_info, hetep, hetkappa_real, hetkappa_imag, freal, fimag, divparam, stabil, scalar1, scalar1, scalar1);
  std::cout<< "hmm assembly" << std::endl;
  curlHMM.assemble();
  std::cout<< "hmm solving" <<std::endl;
  Dune::Stuff::LA::Container< std::complex< double > >::VectorType sol1;
  curlHMM.solve(sol1);
  CurlHMMType::RealVectorType solreal(sol1.size());
  solreal.backend() = sol1.backend().real();
  CurlHMMType::DiscreteFunctionType sol_real_func(curlHMM.space(), solreal, "solution_real_part");
  typedef Stuff::Functions::Difference< ExpressionFct, CurlHMMType::DiscreteFunctionType > DifferenceFct;
  DifferenceFct myerror(expsol, sol_real_func);

  std::cout<< "corrector computation" <<std::endl;
  CurlHMMType::DiscreteFunctionType macro_sol(curlHMM.space(), solreal);
  std::vector< CurlHMMType::DiscreteFunctionType > macro_solution(2, macro_sol);
  std::map< std::pair< size_t, size_t >, CurlHMMType::CurlCellReconstruction::CellDiscreteFunctionType > curl_corrector;
  std::map< std::pair< size_t, size_t >, CurlHMMType::EllipticCellReconstruction::CellDiscreteFunctionType > id_corrector;
  curlHMM.solve_and_correct(macro_solution, curl_corrector, id_corrector);

  //expected correctors
  std::vector< ExpressionFct > expected_solution_total({expsol, zero_expr_vec});
  std::vector< ExpressionFctScalar > expected_cell_id_one({id_cell_one_real, id_cell_one_imag});
  std::vector< ExpressionFctScalar > expected_cell_id_others({zero_expr_sca, zero_expr_sca});
  std::vector< std::vector< ExpressionFctScalar > > expected_cell_id({expected_cell_id_one, expected_cell_id_others, expected_cell_id_others});

  std::vector< ExpressionFct > expected_curl({expsol_curl, zero_expr_vec});
  std::vector< ExpressionFct > exp_curl_cell_one({zero_expr_vec, zero_expr_vec});
  std::vector< ExpressionFct > exp_curl_cell_two({curl_cell_two, zero_expr_vec});
  std::vector< ExpressionFct > exp_curl_cell_three({curl_cell_three, zero_expr_vec});
  std::vector< std::vector< ExpressionFct > > expected_curl_cell({exp_curl_cell_one, exp_curl_cell_two, exp_curl_cell_three});


  //prolongation
  /*Stuff::Grid::Providers::Cube< GridType > fine_grid_provider(0.0, 1.0, 12);
  auto fine_leafview = fine_grid_provider.grid().leafGridView();
  typedef GDT::ProlongedFunction< CurlHMMType::DiscreteFunctionType, LeafGridView > ProlongedDiscrFct;
  ProlongedDiscrFct prolonged(sol_real_func, fine_leafview); */

  //sol_real_func.visualize("hmm_solution_real_part");

  //error computation
  /*typedef Stuff::Functions::Difference< ExpressionFct, ProlongedDiscrFct > ProlongedDifferenceFct;
  ProlongedDifferenceFct prolonged_error(expsol, prolonged); */

  //H(curl) error
  std::cout<< "compute macro error" <<std::endl;
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(leafView);
  const double abserror = std::sqrt(l2_product_operator.apply2(myerror, myerror) + hcurl_product_operator.apply2(myerror, myerror));
  const double relerror = abserror/(std::sqrt(l2_product_operator.apply2(expsol, expsol) + hcurl_product_operator.apply2(expsol, expsol)));
  std::cout<< "absolute error: "<< abserror << std::endl;
  std::cout<< "relative error: "<< relerror <<std::endl;
 // std::cout<< "error in the imaginary part" << std::sqrt(l2_product_operator.apply2(sol_imag_func, sol_imag_func)) <<std::endl;

  //corrector errors
  std::cout<< "computing corrector errors"<<std::endl;
  std::cout<< "error of id corrector: "<< curlHMM.corrector_error(expected_solution_total, expected_cell_id, id_corrector, "id")<<std::endl;
  //std::cout<< "error of id corrector - real part: "<< curlHMM.corrector_error(expected_solution_total, expected_cell_id, id_corrector, "id_real")<<std::endl;
  //std::cout<< "error of id corrector - imag part: "<< curlHMM.corrector_error(expected_solution_total, expected_cell_id, id_corrector, "id_imag")<<std::endl;
  std::cout<< "error of curl corrector: "<< curlHMM.corrector_error(expected_curl, expected_curl_cell, curl_corrector, "curl")<<std::endl;
  //std::cout<< "error of curl corrector - real part: "<< curlHMM.corrector_error(expected_curl, expected_curl_cell, curl_corrector, "curl_real")<<std::endl;
  //std::cout<< "error of curl correctorm - imag part: "<< curlHMM.corrector_error(expected_curl, expected_curl_cell, curl_corrector, "curl_imag")<<std::endl;


  //Helmholtz decomposition
/*  std::cout<<"computing Helmholtz decomposition" <<std::endl;
  HelmholtzDecomp< LeafGridView, 1 > decomp(fine_leafview, bdry_info, prolonged_error);
  sol.scal(0.0);
  decomp.solve(sol);
  auto decomp_vec = decomp.create_vector();
  decomp_vec.backend() = sol.backend().real();
  typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
  DiscreteFctHelmh phi_fct(decomp.space(), decomp_vec);
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::H1Semi< LeafGridView > h1_semi_prod(leafView);
  std::cout<< "error computation"<<std::endl;
  auto l2totalsquared = l2_product_operator.apply2(myerror, myerror);
  auto l2gradsquared = l2_product_operator.apply2(phi_fct, phi_fct);
  std::cout<< "L2 error of Helmholtz decomposition of error" <<std::endl;
  std::cout<< "total L2 error: " <<std::sqrt(l2totalsquared)<<std::endl;
  std::cout<< "gradient part: "<< std::sqrt(l2gradsquared)<< std::endl;
  //for othogonal to gradients we somehow have to compute  the l2norm of prolonged_error-gradient(phi_fct)
  //std::cout<< "orthogonal to gradients: "<< std::sqrt(l2totalsquared - h1_semi_prod.apply2(phi_fct, phi_fct)) <<std::endl;
*/



  return 0;
}

