// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>


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

#include "helmholtz-discretization.hh"
#include "hmm-helmholtz.hh"


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
  typedef Stuff::GlobalLambdaFunction< EntityType, double, 3, double, 1 >  LambdaFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 1 >   ConstantFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 3>    VectorFct;

  typedef std::complex< double > complextype;

  const ConstantFct one(1.0);


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

  typedef GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 > SpaceType;
  typedef Discretization< LeafGridView, 1>::DiscreteFunctionType             DiscreteFctType;

  typedef CurlHMMDiscretization< LeafGridView, GridType, 1 > CurlHMMType;
  typedef Dune::Stuff::LA::Solver< CurlHMMType::MatrixType > HMMSolverType;
  typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::CurlCellDiscreteFctType > CurlCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::EllCellDiscreteFctType >  IdCorrectorType;

  typedef GDT::ProlongedFunction< CurlHMMType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
  typedef Stuff::Functions::Difference< ExpressionFct, DiscreteFctType >                   DifferenceFct;
  typedef Stuff::Functions::Difference< ExpressionFct, ProlongedDiscrFct >                 ProlongedDifferenceFct;
  typedef Stuff::Functions::Difference< DiscreteFctType, ProlongedDiscrFct >               RefDifferenceFct;

  //======================================================================================================================================================================
  // Test Case 1: Academic Testcase (with analytical solution)
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

  //boundary
  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;


  //expected homogenized solution
  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 1, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const ExpressionFct zero_expr_vec("x", {"0", "0", "0"}, 0, "zerofunction", {{"0", "0", "0"}, {"0", "0", "0"}, {"0", "0", "0"}});

 
  //========================================================================================================================================================================
  //   Test Case 1a: Comparison between analytical homogenized solution and HMM approximation
  //========================================================================================================================================================================
/*
  //curl of expected homogenized solution and expected cell problem solutions
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
  const ExpressionFctScalar zero_expr_sca("x", "0", 0, "zerofunction", {"0", "0", "0"});
  const ExpressionFctScalar id_cell_one_real("x", "1/(8*pi)*(sin(2*pi*x[0])-cos(2*pi*x[0]))", 1, "real_part_first_cell_solution_id",
                                       {"1/4*(cos(2*pi*x[0])+sin(2*pi*x[0]))", "0", "0"});
  const ExpressionFctScalar id_cell_one_imag("x", "-1/(8*pi)*(sin(2*pi*x[0])+cos(2*pi*x[0]))", 1, "imag_part_first_cell_solution_id",
                                       {"-1/4*(cos(2*pi*x[0])-sin(2*pi*x[0]))", "0", "0"});

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

  std::ofstream output("output_hmmcurl_analytical_homerror.txt");
  output<< "num_macro_cubes"<< "\t" << "Hcurl norm" << "\t" << "L2 norm" << "\t" << "curl corrector" << "\t" << "id corrector" << "\t" << "L2 Helmholtz" << "\n";

  try{
    for(unsigned int num_macro_cubes : {4, 6, 8, 12}) {
      //macro grid
      Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(0.0, 1.0, num_macro_cubes);
      auto macro_leafView = macro_grid_provider.grid().leafGridView();
      //micro grid
      unsigned int num_micro_cubes = num_macro_cubes/2;
      Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_micro_cubes);
      auto& cell_grid = cell_grid_provider.grid();

      //HMM
      CurlHMMType curlHMM(macro_leafView, cell_grid, curl_bdry_info, hetep, hetkappa_real, hetkappa_imag, freal, fimag, divparam, stabil, one, one, one, true);
      std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_micro_cubes<< " cubes per dim on the micro grid"<< std::endl;
      curlHMM.assemble();
      std::cout<< "hmm solving and corrector computation" <<std::endl;
      CurlHMMType::RealVectorType solreal_hmm;
      Dune::Stuff::Common::Configuration hmm_options = HMMSolverType::options("bicgstab.diagonal");
      hmm_options.set("max_iter", "350000", true);
      if (num_macro_cubes == 12)
        hmm_options.set("precision", "1e-6", true);
      CurlHMMType::DiscreteFunctionType macro_sol(curlHMM.space(), solreal_hmm);
      std::vector< CurlHMMType::DiscreteFunctionType > macro_solution(2, macro_sol);
      std::pair< CurlCorrectorType, IdCorrectorType > correctors(curlHMM.solve_and_correct(macro_solution, hmm_options));

      //visualization
      if(num_macro_cubes == 12){
        macro_solution[0].visualize("hmm_solution_"+std::to_string(int(num_macro_cubes))+"_"+std::to_string(int(num_micro_cubes))+"_real");
        expsol.visualize(macro_leafView, "expected_hom_sol_"+std::to_string(int(num_macro_cubes)), false);
      }

      //errors to the analytical solution
      DifferenceFct error_real(expsol, macro_solution[0]);
      DifferenceFct error_imag(zero_expr_vec, macro_solution[1]);
      std::cout<< "compute macro error" <<std::endl;
      GDT::Products::L2< LeafGridView> l2_product_operator_macro(macro_leafView);
      GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_macro(macro_leafView);  
      const double abserror_curl = std::sqrt(hcurl_product_operator_macro.apply2(error_real, error_real)+ hcurl_product_operator_macro.apply2(error_imag, error_imag));
      const double abserror_l2 = std::sqrt(l2_product_operator_macro.apply2(error_real, error_real) + l2_product_operator_macro.apply2(error_imag, error_imag));
      const double abserror_curl_full = std::sqrt(std::pow(abserror_l2, 2) + std::pow(abserror_curl, 2));

      //corrector errors
      std::cout<< "computing corrector errors"<<std::endl;
      const double curl_corrector_error = curlHMM.corrector_error(expected_curl, expected_curl_cell, correctors.first, "curl");
      const double id_corrector_error = curlHMM.corrector_error(expected_solution_total, expected_cell_id, correctors.second, "id");


      //prolongation for Helmholtz decomsposition on grid with one global refinement
      Stuff::Grid::Providers::Cube< GridType > fine_grid_provider(0.0, 1.0, 2 * num_macro_cubes);
      auto fine_leafView = fine_grid_provider.grid().leafGridView();
      ProlongedDiscrFct prolonged_fine_real(macro_solution[0], fine_leafView);
      ProlongedDiscrFct prolonged_fine_imag(macro_solution[1], fine_leafView);
      ProlongedDifferenceFct prolonged_error_real(expsol, prolonged_fine_real);
      ProlongedDifferenceFct prolonged_error_imag(zero_expr_vec, prolonged_fine_imag);

      //Helmholtz decomposition
      std::cout<<"computing Helmholtz decomposition macro grid with doubled cubes per dim" <<std::endl;
      HelmholtzDecomp< LeafGridView, 1 > decomp_fine(fine_leafView, curl_bdry_info, prolonged_error_real, prolonged_error_imag, one);
      Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_fine;
      decomp_fine.solve(sol_helmholtz_fine);
      //make discrete fct
      Stuff::LA::Container< double >::VectorType decomp_fine_vec_real(sol_helmholtz_fine.size());
      typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
      DiscreteFctHelmh phi_fine_real(decomp_fine.space(), decomp_fine_vec_real);
      std::vector< DiscreteFctHelmh > phi_fine(2, phi_fine_real);
      phi_fine[0].vector().backend() = sol_helmholtz_fine.backend().real();
      phi_fine[1].vector().backend() = sol_helmholtz_fine.backend().imag();
      //compute error of Helmholtz decomposition
      const double helmholtz_error = std::sqrt(l2_product_operator_macro.apply2(phi_fine[0], phi_fine[0]) + l2_product_operator_macro.apply2(phi_fine[1], phi_fine[1]));

      output<< num_macro_cubes << "\t" << abserror_curl_full << "\t" << abserror_l2 << "\t" << curl_corrector_error << "\t" << id_corrector_error << "\t" << helmholtz_error << "\n";
    }//end for loop
  }//end try block
*/

  //========================================================================================================================================================================
  // Test Case 1b: Comparison between discrete reference solution of the heterogenous problem and the HMM approximation for delta=0.2
  // test produces also visualization of HMM lowest order approximation for delta=0.3, 0.2, 0.1
  //========================================================================================================================================================================

  double delta = 0.2;
  const LambdaFct hetepdelta([delta](LambdaFct::DomainType x)
                             {return 1.0/(2+std::cos(2*M_PI*x[0]/delta));},0);
  const LambdaFct hetkappa_real_delta([delta](LambdaFct::DomainType x)
                                      {return -1.0*(2+std::cos(2*M_PI*x[0]/delta))/(9+4*std::cos(2*M_PI*x[0]/delta)+4*std::sin(2*M_PI*x[0]/delta));},0);
  const LambdaFct hetkappa_imag_delta([delta](LambdaFct::DomainType x)
                                      {return (2+std::sin(2*M_PI*x[0]/delta))/(9+4*std::cos(2*M_PI*x[0]/delta)+4*std::sin(2*M_PI*x[0]/delta));},0);

  //reference grid
  unsigned int num_ref_cubes = 24;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  std::ofstream output("output_hmmcurl_analytical_referror"+std::to_string(int(num_ref_cubes))+"_delta"+std::to_string(int(1/delta))+".txt");
  try{
    //reference solution
    Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
    {
    Discretization< LeafGridView, 1> refdiscr(ref_leafView, curl_bdry_info, hetepdelta, hetkappa_real_delta, hetkappa_imag_delta, freal, fimag);
    std::cout<< "number of entities " << ref_leafView.size(0)<<std::endl;
    std::cout<< "number macro dofs " << refdiscr.space().mapper().size()<<std::endl;
    std::cout<< "assembling" << std::endl;
    refdiscr.assemble();
    std::cout<< "solving with bicgstab.diagonal" << std::endl;
    Dune::Stuff::Common::Configuration options = HMMSolverType::options("bicgstab.diagonal");
    options.set("max_iter", "400000", true);
    options.set("precision", "1e-6", true);
    refdiscr.solve(sol_ref, options);
    }
    //make discrete functions
    Stuff::LA::Container< double >::VectorType solreal_ref(sol_ref.size());
    std::vector< DiscreteFctType > sol_ref_func(2, DiscreteFctType(SpaceType(ref_leafView), solreal_ref));
    sol_ref_func[0].vector().backend() = sol_ref.backend().real();
    sol_ref_func[1].vector().backend() = sol_ref.backend().imag();
    sol_ref_func[0].visualize("ref_discrete_solution_delta"+std::to_string(int(1/delta))+"_"+std::to_string(int(num_ref_cubes)));

    //error reference and analytical solution
    {
    DifferenceFct error_real(expsol, sol_ref_func[0]);
    DifferenceFct error_imag(zero_expr_vec, sol_ref_func[1]);
    std::cout<< "compute homogenization error" <<std::endl;
    GDT::Products::L2< LeafGridView> l2_product_operator_macro(ref_leafView);
    GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_macro(ref_leafView);
    const double abserror_curl = std::sqrt(hcurl_product_operator_macro.apply2(error_real, error_real)+ hcurl_product_operator_macro.apply2(error_imag, error_imag));
    const double abserror_l2 = std::sqrt(l2_product_operator_macro.apply2(error_real, error_real) + l2_product_operator_macro.apply2(error_imag, error_imag));
    const double abserror_curl_full = std::sqrt(std::pow(abserror_l2, 2) + std::pow(abserror_curl, 2));
    output << "homogenization error" << "\n" << "Hcurl:"<< "\t" << abserror_curl_full << "\t" << "L2:" << "\t" << abserror_l2 << "\n" << "\n";
    output << "errors reference solution to HMM" << "\n" << "num_macro_cubes" << "\t" << "L2" << "\t" << "Hcurlsemi" << "\t" << "L2 corrector" << "\t" << "Hcurl semi corrector" << "\n";
    }

    for(unsigned int num_macro_cubes : {4, 6, 8, 12}) {
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

      std::cout<< "hmm solving and corrector computation" <<std::endl;
      CurlHMMType::RealVectorType solreal_hmm;
      Dune::Stuff::Common::Configuration hmm_options = HMMSolverType::options("bicgstab.diagonal");
      hmm_options.set("max_iter", "250000", true);
      if(num_macro_cubes == 12)
        hmm_options.set("precision", "1e-6", true);
      CurlHMMType::DiscreteFunctionType macro_sol(curlHMM.space(), solreal_hmm);
      std::vector< CurlHMMType::DiscreteFunctionType > macro_solution(2, macro_sol);
      std::pair< CurlCorrectorType, IdCorrectorType > correctors(curlHMM.solve_and_correct(macro_solution, hmm_options));

      //errors to reference solution
      ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
      ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);
      RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
      RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);

      std::cout<< "macroscopic errors on reference grid" <<std::endl;
      Dune::GDT::Products::L2< LeafGridView > l2_product_operator_ref(ref_leafView);
      Dune::GDT::Products::HcurlSemi< LeafGridView > curl_product_operator_ref(ref_leafView);
      const double l2 = std::sqrt(l2_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                            + l2_product_operator_ref.apply2(reference_error_imag, reference_error_imag));
      const double hcurlsemi = std::sqrt(curl_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                                + curl_product_operator_ref.apply2(reference_error_imag, reference_error_imag));
      std::cout<< "errors to zeroth order approximation" <<std::endl;
      const double l2_correc = curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "l2");
      const double hcurlsemi_correc = curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "hcurlsemi");
      output << num_macro_cubes << "\t" << l2 << "\t" << hcurlsemi << "\t" << l2_correc << "\t" << hcurlsemi_correc << "\n";

      //visualization
      if (num_macro_cubes == 12) {
        typedef Dune::GDT::DeltaCorrectorCurl< CurlHMMType::DiscreteFunctionType, CurlHMMType::CurlCellDiscreteFctType, CurlHMMType::EllCellDiscreteFctType > DeltaCorrectorType;
        //choose the delta's you wnat to visualize by adding the values into the for loop
        for (double delta1 : {0.3, 0.2, 0.15}) {
          DeltaCorrectorType corrector_real(correctors.first.macro_function(), correctors.first.cell_solutions(), correctors.second.cell_solutions(), delta1, "real");
          std::cout<< "visualization for delta "<< delta1 <<std::endl;
          corrector_real.visualize(macro_leafView, "delta_corrector_delta0"+std::to_string(int(100*delta1))+"_"+std::to_string(int(num_macro_cubes))+"_real", false);
        }
      }

    }//end for loop
  }//end try block



  //=========================================================================================================================================================================
  //             Second Testcase from Cao, Zhang, Allegretto, Lin 2010 (Testcase 5.1.1)
  //=========================================================================================================================================================================

/*  
  typedef Stuff::GlobalLambdaFunction< EntityType, double, 3, double, 3, 3 > MatrixLambdaFct;

  //parameters
  const PerScalarExpressionFct muperiodic("x", "20/((2+1.5*sin(2*pi*x[0]+0.75))*(2+1.5*sin(2*pi*x[1]+0.75))*(2+1.5*sin(2*pi*x[2]+0.75)))", 0, "periodicparam");
  const ConstantFct minusone(-1.0);
  const ConstantFct zero(0.0);
  const PerConstFct divparam(0.01);
  const PerConstFct stabil(0.0001);
  const VectorFct zerovec(0.0);
  //rhs
  const VectorFct freal(30.0);
  //boundary
  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;

  double delta = 1.0/3.0;
  const LambdaFct hetmu([delta](LambdaFct::DomainType x)
                        {return 20/((2+1.5*std::sin(2*M_PI*x[0]/delta+0.75))*(2+1.5*std::sin(2*M_PI*x[1]/delta+0.75))*(2+1.5*std::sin(2*M_PI*x[2]/delta+0.75)));}, 0);

  //==========================================================================================================================================================
  //  TestCase 2: Comparison of the HMM approximation to a) the reference homogenized solution and b) the reference solution with delta=1/3
  //==========================================================================================================================================================

  //reference grid
  unsigned int num_ref_cubes = 24;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  //compute homogenized reference solution
  Dune::FieldMatrix< double, 3, 3 > mueff;
  Dune::Stuff::LA::Container< std::complex< double > >::VectorType sol_hom_ref;
  try {
    //cell grid
    unsigned int num_ref_cell_cubes = num_ref_cubes;
    Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
    auto& cell_grid_ref = cell_grid_provider_ref.grid();

    //alternative computation of effective parameters
    {
    SpaceType coarse_space(ref_leafView);
    CurlHMMType::CurlCellReconstruction curl_cell(coarse_space, cell_grid_ref, muperiodic, divparam, stabil);
    std::cout<< "computing effective curl matrix" <<std::endl;
    mueff = curl_cell.effective_matrix();
    //build piece-wise constant function
    const MatrixLambdaFct mu_eff_fct([mueff](LambdaFct::DomainType xx){return mueff;}, 0);

    //assemble and solve homogenized system
    Discretization< LeafGridView, 1, true > homdiscr(ref_leafView, curl_bdry_info, mu_eff_fct, minusone, zero, freal, zerovec);
    std::cout<< "reference homogenized solution, assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
    std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
    homdiscr.assemble();
    std::cout<<"solving with bicgstab.diagonal"<<std::endl;
    Dune::Stuff::Common::Configuration hom_options = HMMSolverType::options("bicgstab.diagonal");
    hom_options.set("precision", "1e-6", true);
    hom_options.set("max_iter", "200000", true);
    homdiscr.solve(sol_hom_ref, hom_options);
    } //end anonymous space just for memory reasons

    //make discrete functions
    Dune::Stuff::LA::Container< double >::VectorType solhom_real_ref(sol_hom_ref.size());
    std::vector< DiscreteFctType > sol_hom_ref_func(2, DiscreteFctType(SpaceType(ref_leafView), solhom_real_ref));
    sol_hom_ref_func[0].vector().backend() = sol_hom_ref.backend().real();
    sol_hom_ref_func[1].vector().backend() = sol_hom_ref.backend().imag();

    //reference solution
    Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
    {
    Discretization< LeafGridView, 1> refdiscr(ref_leafView, curl_bdry_info, hetmu, minusone, zero, freal, zerovec);
    std::cout<< "number of entities " << ref_leafView.size(0) <<std::endl;
    std::cout<< "number macro dofs "<< refdiscr.space().mapper().size() <<std::endl;
    std::cout<< "assembling"<<std::endl;
    refdiscr.assemble();
    std::cout<< "solving with bicgstab.diagonal" <<std::endl;
    Dune::Stuff::Common::Configuration options = HMMSolverType::options("bicgstab.diagonal");
    options.set("max_iter", "500000", true);
    options.set("precision", "1e-6", true);
    refdiscr.solve(sol_ref, options);
    }
    //make discrete functions
    Discretization< LeafGridView, 1>::VectorType solrefreal(sol_ref.size());
    solrefreal.backend() = sol_ref.backend().real();
    Discretization< LeafGridView, 1>::VectorType solrefimag(sol_ref.size());
    solrefimag.backend() = sol_ref.backend().imag();
    std::vector< DiscreteFctType > sol_ref_func({DiscreteFctType(SpaceType(ref_leafView), solrefreal),
                                                 DiscreteFctType(SpaceType(ref_leafView), solrefimag)});
    sol_ref_func[0].visualize("ref_discrete_solution_delta"+std::to_string(int(1/delta))+"_"+std::to_string(int(num_ref_cubes)));

    std::ofstream output("output_hmmcurl_CZAL_error"+std::to_string(int(num_ref_cubes))+"_delta"+std::to_string(int(1/delta))+".txt");
    //error reference and analytical solution
    {
    typedef Stuff::Functions::Difference< DiscreteFctType, DiscreteFctType > DiscrDifferenceFct;
    DiscrDifferenceFct error_real(sol_hom_ref_func[0], sol_ref_func[0]);
    DiscrDifferenceFct error_imag(sol_hom_ref_func[1], sol_ref_func[1]);
    std::cout<< "compute homogenization error" <<std::endl;
    GDT::Products::L2< LeafGridView> l2_product_operator_macro(ref_leafView);
    GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_macro(ref_leafView);
    const double abserror_curl = std::sqrt(hcurl_product_operator_macro.apply2(error_real, error_real)+ hcurl_product_operator_macro.apply2(error_imag, error_imag));
    const double abserror_l2 = std::sqrt(l2_product_operator_macro.apply2(error_real, error_real) + l2_product_operator_macro.apply2(error_imag, error_imag));
    const double abserror_curl_full = std::sqrt(std::pow(abserror_l2, 2) + std::pow(abserror_curl, 2));
    output << "homogenization error" << "\n" << "Hcurl:"<< "\t" << abserror_curl_full << "\t" << "L2:" << "\t" << abserror_l2 << "\n" << "\n";
    output << "errors reference solutions to HMM" << "\n" << "num_macro_cubes" << "\t" << "L2 hom" << "\t" << "Hcurl semi hom" << "\t" << "L2 Helmholtz hom" 
                                                  << "\t" << "L2ref" << "\t" << "Hcurl semi ref" <<"\t" << "Hcurl semi correc" << "\t" << "L2 Helmholtz ref" << "\n";
    }

    for(unsigned int num_macro_cubes : {4, 6, 8, 12}) {
      //macro grid
      Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(0.0, 1.0, num_macro_cubes);
      auto macro_leafView = macro_grid_provider.grid().leafGridView();
      //micro grid
      unsigned int num_micro_cubes = num_macro_cubes/2;
      Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_micro_cubes);
      auto& cell_grid = cell_grid_provider.grid();

      //HMM
      CurlHMMType curlHMM(macro_leafView, cell_grid, curl_bdry_info, muperiodic, minusone, zero, freal, zerovec, divparam, stabil, one, one, zero, false);
      std::cout<< "hmm assembly for " << num_macro_cubes << " cubes per dim on macro grid and "<< num_micro_cubes << " cubes per dim on the micro grid" << std::endl;
      curlHMM.assemble();

      std::cout<< "hmm solving and corrector computation" <<std::endl;
      CurlHMMType::RealVectorType solreal_hmm;
      Dune::Stuff::Common::Configuration hmm_options = HMMSolverType::options("lu.sparse");
      DiscreteFctType macro_sol(curlHMM.space(), solreal_hmm);
      std::vector< DiscreteFctType > macro_solution(2, macro_sol);
      typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::CurlCellDiscreteFctType > CurlCorrectorType;
      typedef Dune::GDT::PeriodicCorrector< CurlHMMType::DiscreteFunctionType, CurlHMMType::EllCellDiscreteFctType > IdCorrectorType;
      std::pair< CurlCorrectorType, IdCorrectorType > correctors(curlHMM.solve_and_correct(macro_solution, hmm_options));

      //define errors
      ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
      ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);
      RefDifferenceFct hom_reference_error_real(sol_hom_ref_func[0], prolonged_ref_real);
      RefDifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], prolonged_ref_imag);
      RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
      RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);

      Dune::GDT::Products::L2< LeafGridView > l2_product_operator(ref_leafView);
      Dune::GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(ref_leafView);

      //error to reference homogenized solution
      std::cout<< "computing errors to homogenized reference solution" <<std::endl;
      const double l2_hom = std::sqrt(l2_product_operator.apply2(hom_reference_error_real, hom_reference_error_real)
                                        + l2_product_operator.apply2(hom_reference_error_imag, hom_reference_error_imag));
      const double hcurl_semi_hom = std::sqrt(hcurl_product_operator.apply2(hom_reference_error_real, hom_reference_error_real)
                                            + hcurl_product_operator.apply2(hom_reference_error_imag, hom_reference_error_imag));

      //error to (heterogeneous) reference solution
      std::cout<< "macroscopic errors on reference grid"<<std::endl;
      const double l2_ref = std::sqrt(l2_product_operator.apply2(reference_error_real, reference_error_real)
                                          + l2_product_operator.apply2(reference_error_imag, reference_error_imag));
      const double hcurl_semi_ref = std::sqrt(hcurl_product_operator.apply2(reference_error_real, reference_error_real)
                                              + hcurl_product_operator.apply2(reference_error_imag, reference_error_imag));
      std::cout<< "errors to zeroth order approximation" <<std::endl;
      const double hcurl_semi_correc = curlHMM.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "hcurlsemi");

      //Helmholtz decomposition of the errors
      Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_hom;
      Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_ref;
      std::cout<< "computing Helmholtz decompositions" <<std::endl;
      HelmholtzDecomp< LeafGridView, 1 > decomp_hom(ref_leafView, curl_bdry_info, hom_reference_error_real, hom_reference_error_imag, one);
      HelmholtzDecomp< LeafGridView, 1 > decomp_ref(ref_leafView, curl_bdry_info, reference_error_real, reference_error_imag, one);
      decomp_hom.solve(sol_helmholtz_hom);
      decomp_ref.solve(sol_helmholtz_ref);
      //make discrete fcts
      Stuff::LA::Container< double >::VectorType decomp_hom_vec_real(sol_helmholtz_hom.size());
      typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
      DiscreteFctHelmh phi_hom_real(decomp_hom.space(), decomp_hom_vec_real);
      std::vector< DiscreteFctHelmh > phi_hom(2, phi_hom_real);
      std::vector< DiscreteFctHelmh > phi_ref(2, phi_hom_real);
      phi_hom[0].vector().backend() = sol_helmholtz_hom.backend().real();
      phi_hom[1].vector().backend() = sol_helmholtz_hom.backend().imag();
      phi_ref[0].vector().backend() = sol_helmholtz_ref.backend().real();
      phi_ref[1].vector().backend() = sol_helmholtz_ref.backend().imag();
      //compute errors of Helmholtz decomposition
      const double helmholtz_error_hom = std::sqrt(l2_product_operator.apply2(phi_hom[0], phi_hom[0]) + l2_product_operator.apply2(phi_hom[1], phi_hom[1]));
      const double helmholtz_error_ref = std::sqrt(l2_product_operator.apply2(phi_ref[0], phi_ref[0]) + l2_product_operator.apply2(phi_ref[1], phi_ref[1]));

      output<< num_macro_cubes << "\t" << l2_hom << "\t" << hcurl_semi_hom << "\t" << helmholtz_error_hom
                               << "\t" << l2_ref << "\t" << hcurl_semi_ref << "\t" << hcurl_semi_correc << "\t" << helmholtz_error_ref << "\n";

      //visualization
      if(num_macro_cubes == 12){
        macro_solution[0].visualize("hmm_solution_"+std::to_string(int(num_macro_cubes))+"_"+std::to_string(int(num_micro_cubes))+"_real");
      }
    }//end for loop
  }//end try block
*/

  catch(Dune::Exception& ee) {
    std::cout<< ee.what() <<std::endl;
  }

  catch(...) {
    std::cout<< "something went wrong"<<std::endl;
  }


  return 0;
}
