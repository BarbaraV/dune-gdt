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

  // some typedefs
  typedef Dune::ALUGrid< 3, 3, simplex, conforming > GridType;
  typedef GridType::LeafGridView LeafGridView;
  typedef LeafGridView::Codim< 0 >::Entity EntityType;

  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 3 > ExpressionFct;
  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 1 > ExpressionFctScalar;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 1 >  ConstantFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 3> VectorFct;

  typedef std::complex< double > complextype;

  const ConstantFct one(1.0);

  try{

  //=============================================================================================================================================================
  //test direct discretization of curl-curl-problem
  //=============================================================================================================================================================

/*
  //instantiate grid
  unsigned int num_cubes = 6;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, num_cubes);
  auto leafView = grid_provider.grid().leafGridView();

  //define functions
  const ExpressionFct expsol_direct("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const double pi = 3.141592653589793;
  const ConstantFct minusone(-1.0);
  const ConstantFct realweight(2*pi*pi-1.0);   //2*pi^2+minusone
  const ConstantFct imagweight(1.0);           //one

  const Stuff::Functions::Product< ConstantFct, ExpressionFct > curl_freal(realweight, expsol_direct);
  const Stuff::Functions::Product< ConstantFct, ExpressionFct > curl_fimag(imagweight, expsol_direct);

  //expsol.visualize(leafView, "expectedsolution", false);

  //discretization
  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;
  Discretization< LeafGridView, 1> realdiscr(leafView, curl_bdry_info, one, minusone, one, curl_freal, curl_fimag);
  std::cout<< "assembling"<<std::endl;
  realdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_direct;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  realdiscr.solve(sol_direct);
  //realdiscr.visualize(sol_direct, "discrete_solution", "discrete_solution");

  //make discrete function (absolute value)
  typedef Discretization< LeafGridView, 1>::ConstDiscreteFunctionType ConstDiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal_direct(sol_direct.size());
  solreal_direct.backend() = sol_direct.backend().cwiseAbs();
  ConstDiscreteFct solasfunc(realdiscr.space(), solreal_direct, "solution_as_discrete_function");

  //error computation
  Stuff::Functions::Difference< ExpressionFct, ConstDiscreteFct > myerror(expsol_direct, solasfunc);
  std::cout<< "error computation"<<std::endl;
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(leafView);
  const double abserror = std::sqrt(l2_product_operator.apply2(myerror, myerror) + hcurl_product_operator.apply2(myerror, myerror));
  const double relerror = abserror/(std::sqrt(l2_product_operator.apply2(expsol_direct, expsol_direct) + hcurl_product_operator.apply2(expsol_direct, expsol_direct)));
  std::cout<< "absolute error: "<< abserror << std::endl;
  std::cout<< "relative error: "<< relerror <<std::endl;
*/

  //=======================================================================================================================================================================
  // HMM Discretization
  //=======================================================================================================================================================================

  //some general typedefs
  typedef Fem::PeriodicLeafGridPart< GridType>                PeriodicGridPartType;
  typedef Fem::PeriodicLeafGridPart< GridType >::GridViewType PeriodicViewType;
  typedef PeriodicViewType::Codim< 0 >::Entity                PeriodicEntityType;

  typedef Stuff::Functions::Expression< PeriodicEntityType, double, 3, double, 1 > PerScalarExpressionFct;
  typedef Stuff::Functions::Constant< PeriodicEntityType, double, 3, double, 1>    PerConstFct;

  //======================================================================================================================================================================
  // Test Case 1: Academic Testscase (with analytical solution)
  //======================================================================================================================================================================


  //parameters
  const PerScalarExpressionFct hetep("x", "1/(2+cos(2*pi*x[0]))", 0, "periodic_permittivity");
  const PerScalarExpressionFct hetkappa_real("x", "-(2+cos(2*pi*x[0]))/(9+4*cos(2*pi*x[0])+4*sin(2*pi*x[0]))", 0, "periodic_kappa_real");
  const PerScalarExpressionFct hetkappa_imag("x", "(2+sin(2*pi*x[0]))/(9+4*cos(2*pi*x[0])+4*sin(2*pi*x[0]))", 0, "periodic_kappa_imag");
  const PerConstFct divparam(0.01);
  const PerConstFct stabil(0.0001);

  //rhs
  const ExpressionFct freal("x", {"(pi*pi-0.25)*sin(pi*x[1])*sin(pi*x[2])", "(pi*pi*(0.5+1/sqrt(3))-0.25)*sin(pi*x[0])*sin(pi*x[2])",
                                  "(pi*pi*(0.5+1/sqrt(3))-0.25)*sin(pi*x[0])*sin(pi*x[1])"}, 2, "real_rhs");
  const ExpressionFct fimag("x", {"0.25*sin(pi*x[1])*sin(pi*x[2])", "0.25*sin(pi*x[0])*sin(pi*x[2])", "0.25*sin(pi*x[0])*sin(pi*x[1])"}, 2, "imag_rhs");


  //analytical solutions
  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 1, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const ExpressionFct expsol_curl("x", {"pi*sin(pi*x[0])*(cos(pi*x[1])-cos(pi*x[2]))", "pi*sin(pi*x[1])*(cos(pi*x[2])-cos(pi*x[0]))", "pi*sin(pi*x[2])*(cos(pi*x[0])-cos(pi*x[1]))"},
                                  1, "expectedsolution_curl");
  const ExpressionFct curl_cell_two("x", {"0", "0", "-1/(4*pi)*sin(2*pi*x[0])"}, 1, "expected_cellsol_two",
                             {{"0", "0", "0"},
                              {"0", "0", "0"},
                              {"-1/2*cos(2*pi*x[0])", "0", "0"}});
  const ExpressionFct curl_cell_three("x", {"0", "1/(4*pi)*sin(2*pi*x[0])", "0"}, 1, "expected_cellsol_three",
                             {{"0", "0", "0"},
                              {"1/2*cos(2*pi*x[0])", "0", "0"},
                              {"0", "0", "0"}});
  const ExpressionFct zero_expr_vec("x", {"0", "0", "0"}, 0, "zerofunction", {{"0", "0", "0"}, {"0", "0", "0"}, {"0", "0", "0"}});
  const ExpressionFctScalar zero_expr_sca("x", "0", 0, "zerofunction", {"0", "0", "0"});
  const ExpressionFctScalar id_cell_one_real("x", "1/(8*pi)*(sin(2*pi*x[0])-cos(2*pi*x[0]))", 1, "real_part_first_cell_solution_id",
                                       {"1/4*(cos(2*pi*x[0])+sin(2*pi*x[0]))", "0", "0"});
  const ExpressionFctScalar id_cell_one_imag("x", "-1/(8*pi)*(sin(2*pi*x[0])+cos(2*pi*x[0]))", 1, "imag_part_first_cell_solution_id",
                                       {"-1/4*(cos(2*pi*x[0])-sin(2*pi*x[0]))", "0", "0"});
  //expsol.visualize(macro_leafView, "expected_homogenized_solution", false);

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


  //reference solution

  //reference grid
  unsigned int num_ref_cubes = 24;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  //delta dependent parameters
  //delta=1/5
  double delta = 1.0/5.0;
  const ExpressionFctScalar hetepdelta("x", "1/(2+cos(2*pi*x[0]*5))", 0, "periodic_permittivity_delta");
  const ExpressionFctScalar hetkappa_real_delta("x", "-(2+cos(2*pi*x[0]*5))/(9+4*cos(2*pi*x[0]*5)+4*sin(2*pi*x[0]*5))", 0, "periodic_kappa_real_delta");
  const ExpressionFctScalar hetkappa_imag_delta("x", "(2+sin(2*pi*x[0]*5))/(9+4*cos(2*pi*x[0]*5)+4*sin(2*pi*x[0]*5))", 0, "periodic_kappa_imag_delta");

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;
  Discretization< LeafGridView, 1> refdiscr(ref_leafView, curl_bdry_info, hetepdelta, hetkappa_real_delta, hetkappa_imag_delta, freal, fimag);
  std::cout<< "number of entities " << ref_leafView.size(0)<<std::endl;
  std::cout<< "number macro dofs " << refdiscr.space().mapper().size()<<std::endl;
  std::cout<< "assembling" << std::endl;
  refdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
  std::cout<< "solving with bicgstab.diagonal" << std::endl;
  //refdiscr.solve(sol_ref);
  typedef Dune::Stuff::LA::Solver< Discretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
  options.set("max_iter", "200000", true);
  SolverType solver(refdiscr.system_matrix());
  solver.apply(refdiscr.rhs_vector(), sol_ref, options);

  //make discrete functions
  Discretization< LeafGridView, 1>::VectorType solrefreal(sol_ref.size());
  solrefreal.backend() = sol_ref.backend().real();
  Discretization< LeafGridView, 1>::VectorType solrefimag(sol_ref.size());
  solrefimag.backend() = sol_ref.backend().imag();
  std::vector< Discretization< LeafGridView, 1>::DiscreteFunctionType > sol_ref_func({Discretization< LeafGridView, 1 >::DiscreteFunctionType(refdiscr.space(), solrefreal),
                                                                                       Discretization< LeafGridView, 1>::DiscreteFunctionType(refdiscr.space(), solrefimag)});

  //refdiscr.visualize(sol_ref, "discrete_solution", "discrete_solution");


  for(unsigned int num_macro_cubes : {4}) {
    //macro grid
    Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(0.0, 1.0, num_macro_cubes);
    auto macro_leafView = macro_grid_provider.grid().leafGridView();
    //micro grid
    unsigned int num_micro_cubes = num_macro_cubes/2;
    Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_micro_cubes);
    auto& cell_grid = cell_grid_provider.grid();

    //HMM
    DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
    typedef CurlHMMDiscretization< LeafGridView, GridType, 1 > CurlHMMType;
    CurlHMMType curlHMM(macro_leafView, cell_grid, bdry_info, hetep, hetkappa_real, hetkappa_imag, freal, fimag, divparam, stabil, one, one, one, true);
    std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_micro_cubes<< " cubes per dim on the micro grid"<< std::endl;
    curlHMM.assemble();
    std::cout<< "hmm solving" <<std::endl;
    Dune::Stuff::LA::Container< complextype >::VectorType sol_hmm;
    curlHMM.solve(sol_hmm);
    CurlHMMType::RealVectorType solreal_hmm(sol_hmm.size());
    solreal_hmm.backend() = sol_hmm.backend().real();
    //CurlHMMType::DiscreteFunctionType sol_real_func(curlHMM.space(), solreal_hmm, "solution_real_part");
    //sol_real_func.visualize("hmm_solution_12_6_real");

    typedef GDT::ProlongedFunction< CurlHMMType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
    typedef Stuff::Functions::Difference< ExpressionFct, CurlHMMType::DiscreteFunctionType > DifferenceFct;
    typedef Stuff::Functions::Difference< ExpressionFct, ProlongedDiscrFct >                 ProlongedDifferenceFct;

    std::cout<< "corrector computation" <<std::endl;
    CurlHMMType::DiscreteFunctionType macro_sol(curlHMM.space(), solreal_hmm);
    std::vector< CurlHMMType::DiscreteFunctionType > macro_solution(2, macro_sol);
    typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::CurlCellDiscreteFctType > CurlCorrectorType;
    typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::EllCellDiscreteFctType > IdCorrectorType;
    std::pair< CurlCorrectorType, IdCorrectorType > correctors(curlHMM.solve_and_correct(macro_solution));

    //errors to the analytical solution
    typedef Stuff::Functions::Difference< ExpressionFct, CurlHMMType::DiscreteFunctionType > DifferenceFct;
    DifferenceFct error_real(expsol, macro_solution[0]);
    DifferenceFct error_imag(zero_expr_vec, macro_solution[1]);
    std::cout<< "compute macro error" <<std::endl;
    GDT::Products::L2< LeafGridView> l2_product_operator_macro(macro_leafView);
    GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_macro(macro_leafView);
    const double abserror_hcurl = std::sqrt(l2_product_operator_macro.apply2(error_real, error_real) + hcurl_product_operator_macro.apply2(error_real, error_real)
                                             + l2_product_operator_macro.apply2(error_imag, error_imag) + hcurl_product_operator_macro.apply2(error_imag, error_imag));
    //const double relerror = abserror_hcurl/(std::sqrt(l2_product_operator_macro.apply2(expsol, expsol) + hcurl_product_operator_macro.apply2(expsol, expsol)));
    std::cout<< "absolute error in Hcurl norm: "<< abserror_hcurl << std::endl;
    std::cout<< "absolute error in L2 norm: " <<std::sqrt(l2_product_operator_macro.apply2(error_real, error_real)
                                                          + l2_product_operator_macro.apply2(error_imag, error_imag))<<std::endl;
    //std::cout<< "relative error: "<< relerror <<std::endl;

    //corrector errors
    std::cout<< "computing corrector errors"<<std::endl;
    std::cout<< "error of id corrector: "<< curlHMM.corrector_error(expected_solution_total, expected_cell_id, correctors.second, "id")<<std::endl;
    std::cout<< "error of curl corrector: "<< curlHMM.corrector_error(expected_curl, expected_curl_cell, correctors.first, "curl")<<std::endl;


    //prolongation for Helmholtz decomsposition on grid with one global refinement
    Stuff::Grid::Providers::Cube< GridType > fine_grid_provider(0.0, 1.0, 2 * num_macro_cubes);
    auto fine_leafView = fine_grid_provider.grid().leafGridView();
    ProlongedDiscrFct prolonged_fine_real(macro_solution[0], fine_leafView);
    ProlongedDiscrFct prolonged_fine_imag(macro_solution[1], fine_leafView);
    ProlongedDifferenceFct prolonged_error_real(expsol, prolonged_fine_real);
    ProlongedDifferenceFct prolonged_error_imag(zero_expr_vec, prolonged_fine_imag);

   //Helmholtz decomposition
    std::cout<<"computing Helmholtz decomposition macro grid with doubled cubes per dim" <<std::endl;
    HelmholtzDecomp< LeafGridView, 1 > decomp_fine(fine_leafView, bdry_info, prolonged_error_real, prolonged_error_imag, one);
    Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_fine;
    decomp_fine.solve(sol_helmholtz_fine);
    auto decomp_fine_vec_real = decomp_fine.create_vector();
    auto decomp_fine_vec_imag = decomp_fine.create_vector();
    decomp_fine_vec_real.backend() = sol_helmholtz_fine.backend().real();
    decomp_fine_vec_imag.backend() = sol_helmholtz_fine.backend().imag();

    //compute error of Helmholtz decomposition
    typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
    DiscreteFctHelmh phi_fine_real(decomp_fine.space(), decomp_fine_vec_real);
    DiscreteFctHelmh phi_fine_imag(decomp_fine.space(), decomp_fine_vec_imag);
    std::cout<< "L2 error of gradient part in Helmholtz decomposition (on the macro grid): "<<          //why does that work?
                 std::sqrt(l2_product_operator_macro.apply2(phi_fine_real, phi_fine_real) + l2_product_operator_macro.apply2(phi_fine_imag, phi_fine_imag))<< std::endl;
    //GDT::Products::L2< LeafGridView> l2_product_operator_fine(fine_leafView);
    //std::cout<< "L2 error of gradient part in Helmholtz decomposition (on the fine grid): "<<
     //            std::sqrt(l2_product_operator_fine.apply2(phi_fine_real, phi_fine_real) + l2_product_operator_fine.apply2(phi_fine_imag, phi_fine_imag))<< std::endl;


    //errors to reference solution
    typedef Stuff::Functions::Difference< Discretization< LeafGridView, 1>::DiscreteFunctionType, ProlongedDiscrFct > RefDifferenceFct;
    ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
    ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);
    RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
    RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);

    std::cout<< "macroscopic errors on reference grid" <<std::endl;
    Dune::GDT::Products::L2< LeafGridView > l2_product_operator_ref(ref_leafView);
    Dune::GDT::Products::HcurlSemi< LeafGridView > curl_product_operator_ref(ref_leafView);
    std::cout<< "L2 error: " << std::sqrt(l2_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                          + l2_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
    std::cout<< "Hcurl seminorm "<< std::sqrt(curl_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                              + curl_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
    std::cout<< "errors to zeroth order approximation" <<std::endl;
    std::cout<< "l2: "<< curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "l2") <<std::endl;
    std::cout<< "hcurl seminorm: "<< curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "hcurlsemi") <<std::endl;

   //Helmholtz decomposition and its error for reference solution
    std::cout<<"computing Helmholtz decomposition on reference grid" <<std::endl;
    HelmholtzDecomp< LeafGridView, 1 > decomp_ref(ref_leafView, bdry_info, prolonged_error_real, prolonged_error_imag, one);
    Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_ref;
    decomp_ref.solve(sol_helmholtz_ref);
    auto decomp_ref_vec_real = decomp_ref.create_vector();
    auto decomp_ref_vec_imag = decomp_ref.create_vector();
    decomp_ref_vec_real.backend() = sol_helmholtz_ref.backend().real();
    decomp_ref_vec_imag.backend() = sol_helmholtz_ref.backend().imag();

    //compute error of Helmholtz decomposition
    typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
    DiscreteFctHelmh phi_ref_real(decomp_ref.space(), decomp_ref_vec_real);
    DiscreteFctHelmh phi_ref_imag(decomp_ref.space(), decomp_ref_vec_imag);
    std::cout<< "L2 error of gradient part in Helmholtz decomposition (on the reference grid): "<<
                 std::sqrt(l2_product_operator_ref.apply2(phi_ref_real, phi_ref_real) + l2_product_operator_ref.apply2(phi_ref_imag, phi_ref_imag))<< std::endl;

 /* //visualization
    if (num_macro_cubes == 12) {
      typedef Dune::GDT::DeltaCorrectorCurl< CurlHMMType::DiscreteFunctionType, CurlHMMType::CurlCellDiscreteFctType, CurlHMMType::EllCellDiscreteFctType > DeltaCorrectorType;
      DeltaCorrectorType corrector_real(correctors.first.macro_function(), correctors.first.cell_solutions(), correctors.second.cell_solutions(), delta, "real");
      DeltaCorrectorType corrector_imag(correctors.first.macro_function(), correctors.first.cell_solutions(), correctors.second.cell_solutions(), delta, "imag");
      corrector_real.visualize(macro_leafView, "delta_corrector_12_real", false);
      corrector_imag.visualize(macro_leafView, "delta_corrector_12_imag", false);
    }
*/

  }//end for loop



  //=========================================================================================================================================================================
  //             Second Testcase from Cao, Zhang, Allegretto, Lin 2010 (Testcase 5.1.1)
  //=========================================================================================================================================================================
/*
  //delta =1/3
  const double delta =1.0/3.0;
  const ExpressionFctScalar hetmu("x", "20/((2+1.5*sin(2*pi*3*x[0]+0.75))*(2+1.5*sin(2*pi*3*x[1]+0.75))*(2+1.5*sin(2*pi*3*x[2]+0.75)))", 2, "oscillatingparam");
  const ConstantFct minusone(-1.0);
  const ConstantFct zero(0.0);
  const PerConstFct divparam(0.01);
  const PerConstFct stabil(0.0001);
  const VectorFct zerovec(0.0);
  const VectorFct freal(30.0);

  //reference grid
  unsigned int num_ref_cubes = 24;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  //reference solution
  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;
  Discretization< LeafGridView, 1> refdiscr(ref_leafView, curl_bdry_info, hetmu, minusone, zero, freal, zerovec);
  std::cout<< "number of entities " << ref_leafView.size(0) <<std::endl;
  std::cout<< "number macro dofs "<< refdiscr.space().mapper().size() <<std::endl;
  std::cout<< "assembling"<<std::endl;
  refdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
  std::cout<< "solving with bicgstab.diagonal" <<std::endl;
  typedef Dune::Stuff::LA::Solver< Discretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
  options.set("max_iter", "200000", true);
  SolverType solver(refdiscr.system_matrix());
  solver.apply(refdiscr.rhs_vector(), sol_ref, options);

  //make discrete functions
  Discretization< LeafGridView, 1>::VectorType solrefreal(sol_ref.size());
  solrefreal.backend() = sol_ref.backend().real();
  Discretization< LeafGridView, 1>::VectorType solrefimag(sol_ref.size());
  solrefimag.backend() = sol_ref.backend().imag();
  std::vector< Discretization< LeafGridView, 1>::DiscreteFunctionType > sol_ref_func({Discretization< LeafGridView, 1 >::DiscreteFunctionType(refdiscr.space(), solrefreal),
                                                                                       Discretization< LeafGridView, 1>::DiscreteFunctionType(refdiscr.space(), solrefimag)});
  //refdiscr.visualize(sol_ref, "discrete_solution", "discrete_solution");

  //HMM solution
  const PerScalarExpressionFct muperiodic("x", "20/((2+1.5*sin(2*pi*x[0]+0.75))*(2+1.5*sin(2*pi*x[1]+0.75))*(2+1.5*sin(2*pi*x[2]+0.75)))", 0, "periodicparam");
  for(unsigned int num_macro_cubes : {4}) {
    //macro grid
    Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(0.0, 1.0, num_macro_cubes);
    auto macro_leafView = macro_grid_provider.grid().leafGridView();
    //micro grid
    unsigned int num_micro_cubes = num_macro_cubes/2;
    Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_micro_cubes);
    auto& cell_grid = cell_grid_provider.grid();

    //HMM
    typedef CurlHMMDiscretization< LeafGridView, GridType, 1 > CurlHMMType;
    CurlHMMType curlHMM(macro_leafView, cell_grid, curl_bdry_info, muperiodic, minusone, zero, freal, zerovec, divparam, stabil, one, one, zero);
    std::cout<< "hmm assembly for " << num_macro_cubes << " cubes per dim on macro grid and "<< num_micro_cubes << " cubes per dim on the micro grid" << std::endl;
    curlHMM.assemble();
    std::cout<< "hmm solving" <<std::endl;
    Dune::Stuff::LA::Container< complextype >::VectorType sol_hmm;
    curlHMM.solve(sol_hmm);

    CurlHMMType::RealVectorType solreal_hmm(sol_hmm.size());
    solreal_hmm.backend() = sol_hmm.backend().real();
    //CurlHMMType::DiscreteFunctionType sol_real_func(curlHMM.space(), solreal_hmm, "solution_real_part");
    //sol_real_func.visualize("hmm_solution_real_12_6");

    typedef GDT::ProlongedFunction< CurlHMMType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
    typedef Stuff::Functions::Difference< ExpressionFct, CurlHMMType::DiscreteFunctionType > DifferenceFct;
    typedef Stuff::Functions::Difference< ExpressionFct, ProlongedDiscrFct >                 ProlongedDifferenceFct;

    std::cout<< "corrector computation" <<std::endl;
    CurlHMMType::DiscreteFunctionType macro_sol(curlHMM.space(), solreal_hmm);
    std::vector< CurlHMMType::DiscreteFunctionType > macro_solution(2, macro_sol);
    typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::CurlCellDiscreteFctType > CurlCorrectorType;
    typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::EllCellDiscreteFctType > IdCorrectorType;
    std::pair< CurlCorrectorType, IdCorrectorType > correctors(curlHMM.solve_and_correct(macro_solution));

    //error computation
    typedef Stuff::Functions::Difference< Discretization< LeafGridView, 1>::DiscreteFunctionType, ProlongedDiscrFct > RefDifferenceFct;
    ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
    ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);
    RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
    RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);

    std::cout<< "macroscopic errors on reference grid"<<std::endl;
    Dune::GDT::Products::L2< LeafGridView > l2_product_operator_ref(ref_leafView);
    Dune::GDT::Products::HcurlSemi< LeafGridView > curl_product_operator_ref(ref_leafView);
    std::cout<< "L2 error: " << std::sqrt(l2_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                          + l2_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
    std::cout<< "Hcurl seminorm "<< std::sqrt(curl_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                              + curl_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
    std::cout<< "errors to zeroth order approximation" <<std::endl;
    std::cout<< "l2: "<< curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "l2") <<std::endl;
    std::cout<< "hcurl seminorm: "<< curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "hcurlsemi") <<std::endl;

    //Helmholtz decomposition
    std::cout<<"computing Helmholtz decomposition on reference grid" <<std::endl;
    HelmholtzDecomp< LeafGridView, 1 > decomp_ref(ref_leafView, curl_bdry_info, reference_error_real, reference_error_imag, one);
    Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_ref;
    decomp_ref.solve(sol_helmholtz_ref);
    auto decomp_ref_vec_real = decomp_ref.create_vector();
    auto decomp_ref_vec_imag = decomp_ref.create_vector();
    decomp_ref_vec_real.backend() = sol_helmholtz_ref.backend().real();
    decomp_ref_vec_imag.backend() = sol_helmholtz_ref.backend().imag();

    //compute error of Helmholtz decomposition
    typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
    DiscreteFctHelmh phi_ref_real(decomp_ref.space(), decomp_ref_vec_real);
    DiscreteFctHelmh phi_ref_imag(decomp_ref.space(), decomp_ref_vec_imag);
    std::cout<< "L2 error of gradient part in Helmholtz decomposition (on the reference grid): "<<
                 std::sqrt(l2_product_operator_ref.apply2(phi_ref_real, phi_ref_real) + l2_product_operator_ref.apply2(phi_ref_imag, phi_ref_imag))<< std::endl;

  }//end for loop
*/



  }//end try block
  catch(...) {
    std::cout<< "something went wrong"<<std::endl;
  }

  return 0;
}
