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

#include <dune/gdt/discretefunction/prolonged.hh>
#include <dune/gdt/products/weightedl2.hh>
#include <dune/gdt/products/h1.hh>

#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>

#include "helmholtz-discretization.hh"
#include "hmm-helmholtz.hh"

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

  //instantiate  reference grid
 /* unsigned int num_cubes = 256;
  const double left_outer = 0.0;
  const double right_outer = 1.0;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_cubes);
  auto leafView = grid_provider.grid().leafGridView();

  const double wavenumber = 28.0;
  const LambdaFct bdry_real([wavenumber, left_outer, right_outer](LambdaFct::DomainType x){if (std::abs(x[0] - right_outer) < 1e-12)
                                                                    return -2*wavenumber*std::sin(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12 || std::abs(x[1] - right_outer) < 1e-12)
                                                                    return -1*wavenumber*std::sin(wavenumber*x[0]);
                                                                  else
                                                                    return 0.0;}, 0);
  const LambdaFct bdry_imag([wavenumber, left_outer, right_outer](LambdaFct::DomainType x){if (std::abs(x[0] - right_outer) < 1e-12)
                                                                    return -2*wavenumber*std::cos(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12 || std::abs(x[1] - right_outer) < 1e-12)
                                                                    return -1*wavenumber*std::cos(wavenumber*x[0]);
                                                                  else
                                                                    return 0.0;}, 0);

  //expected solution
  const ExpressionFct exp_sol_real("x", "cos(32.0*x[0])", 2, "expected_solution.real_part", {"-32.0*sin(32.0*x[0])", "0"});
  const ExpressionFct exp_sol_imag("x", "-sin(32.0*x[0])", 2, "expected_solution.imag_part", {"-32.0*cos(32.0*x[0])", "0"});
  //exp_sol_real.visualize(leafView, "expected_solution.real", false);

  const ConstantFct ksquared(wavenumber*wavenumber);


  //discretization
  std::cout<< "wavenumber "<< wavenumber <<std::endl;
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > bdry_info;
  HelmholtzDiscretization< LeafGridView, 1> discr_direct(leafView, bdry_info, one, zero, one, zero, wavenumber, bdry_real, bdry_imag, zero, zero);
  std::cout<< "assembling on grid with "<< num_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entitites "<< leafView.size(0) << " and number of reference dofs: "<< discr_direct.space().mapper().size() <<std::endl;
  discr_direct.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_direct;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  //discr_direct.solve(sol_direct);
  typedef Dune::Stuff::LA::Solver< HelmholtzDiscretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
  options.set("max_iter", "50000", true);
  SolverType solver(discr_direct.system_matrix());
  solver.apply(discr_direct.rhs_vector(), sol_direct, options);
 // discr_direct.visualize(sol_direct, "discrete_solution", "discrete_solution");

  //make discrete function
  typedef HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType DiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal_direct(sol_direct.size());
  Stuff::LA::Container< double >::VectorType solimag_direct(sol_direct.size());
  solreal_direct.backend() = sol_direct.backend().real();
  solimag_direct.backend() = sol_direct.backend().imag();
  DiscreteFct solasfct_real(discr_direct.space(), solreal_direct, "discrete_function");
  DiscreteFct solasfct_imag(discr_direct.space(), solimag_direct, "discrete_function");


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
*/

  //=======================================================================================================================================================================
  // HMM Discretization
  //=======================================================================================================================================================================

  //some typedefs
  typedef Dune::Fem::PeriodicLeafGridPart< GridType >::GridViewType PeriodicViewType;
  typedef PeriodicViewType::template Codim< 0 >::Entity             PeriodicEntityType;

  //======================================================================================================================================================================
  // TestCase1: Incoming plane wave from the right, quadratic scatterer (0.375, 0.625)^2 with inclusions (0.25, 0.75)^2
  //======================================================================================================================================================================

  //parameters
  const double left_inner = 0.25;
  const double right_inner = 0.75;
  const double d_right = 0.75;
  const double d_left = 0.25;
  const double delta = 1.0/8.0;
  ConstantFct a_diel(3.0);
  ConstantFct a_incl_real(2.0);
  ConstantFct a_incl_imag(-0.001);
  ConstantFct stabil(0.0001);
  //filters
  const std::function< bool(const PeriodicViewType& , const PeriodicEntityType& ) > filter_inclusion
          = [d_left, d_right](const PeriodicViewType& /*cell_grid_view*/, const PeriodicEntityType& periodic_entity) -> bool
            {const auto xx = periodic_entity.geometry().center();
             return !(xx[0] >= d_left && xx[0] <= d_right && xx[1] >= d_left && xx[1] <= d_right);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [left_inner, right_inner](const LeafGridView& /*grid_view*/, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [left_inner, right_inner](const LeafGridView& /*grid_view*/, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return !(xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner);};

  //reference solution
  //instantiate  reference grid
  unsigned int num_ref_cubes = 128;
  const double left_outer = 0.0;
  const double right_outer = 1.0;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  //delta dependent parameters
  double intpart;
  const LambdaFct a_real([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                            {if (x[0] >= left_inner && x[0] <= right_inner && x[1] >= left_inner && x[1] <= right_inner) { //inside scatterer
                               if (std::modf(x[0]/delta, &intpart) >= d_left && std::modf(x[0]/delta, &intpart) <= d_right
                                       && std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right)
                                 return delta*delta*2.0;   //a_incl_real
                               return 3.0;   //a_diel
                               }
                             return 1.0;}, 0);
 // a_real.visualize(ref_leafView, "parameter_real", false);
  const LambdaFct a_imag([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                            {if (x[0] >= left_inner && x[0] <= right_inner && x[1] >= left_inner && x[1] <= right_inner) {
                               if (std::modf(x[0]/delta, &intpart) >= d_left && std::modf(x[0]/delta, &intpart) <= d_right
                                       && std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right)
                                 return -0.001*delta*delta;  //a_incl_imag
                               return 0.0;
                               }
                             return 0.0;}, 0);
 // a_imag.visualize(ref_leafView, "parameter_imag", false);

  for (const double wavenumber : {16.0}) {
  //boundary functions
  const LambdaFct bdry_real([wavenumber, left_outer, right_outer](LambdaFct::DomainType x){if (std::abs(x[0] - right_outer) < 1e-12)
                                                                    return -2*wavenumber*std::sin(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12 || std::abs(x[1] - right_outer) < 1e-12)
                                                                    return -1*wavenumber*std::sin(wavenumber*x[0]);
                                                                  else
                                                                    return 0.0;}, 0);
  const LambdaFct bdry_imag([wavenumber, left_outer, right_outer](LambdaFct::DomainType x){if (std::abs(x[0] - right_outer) < 1e-12)
                                                                    return -2*wavenumber*std::cos(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12 || std::abs(x[1] - right_outer) < 1e-12)
                                                                    return -1*wavenumber*std::cos(wavenumber*x[0]);
                                                                  else
                                                                    return 0.0;}, 0);

  //compute reference solution
  std::cout<< "wavenumber "<< wavenumber <<std::endl;
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > ref_bdry_info;
  HelmholtzDiscretization< LeafGridView, 1> refdiscr(ref_leafView, ref_bdry_info, a_real, a_imag, one, zero, wavenumber, bdry_real, bdry_imag, zero, zero);
  std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< refdiscr.space().mapper().size() <<std::endl;
  refdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  //refdiscr.solve(sol_ref);
  typedef Dune::Stuff::LA::Solver< HelmholtzDiscretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration ref_options = SolverType::options("bicgstab.diagonal");
  ref_options.set("max_iter", "250000", true);
  SolverType ref_solver(refdiscr.system_matrix());
  ref_solver.apply(refdiscr.rhs_vector(), sol_ref, ref_options);
  refdiscr.visualize(sol_ref, "discrete_solution_"+std::to_string((int(wavenumber)))+"_delta8_192", "discrete_solution_"+std::to_string(int(wavenumber)));
  //make discrete function
  typedef HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType DiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal_direct(sol_ref.size());
  Stuff::LA::Container< double >::VectorType solimag_direct(sol_ref.size());
  solreal_direct.backend() = sol_ref.backend().real();
  solimag_direct.backend() = sol_ref.backend().imag();
  std::vector< HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType > sol_ref_func({HelmholtzDiscretization< LeafGridView, 1 >::DiscreteFunctionType(refdiscr.space(), solreal_direct),
                                                                                               HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType(refdiscr.space(), solimag_direct)});



  //reference homogenized solution
/*  unsigned int num_ref_cell_cubes = num_ref_cubes;
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
  auto& cell_grid_ref = cell_grid_provider_ref.grid();
  //grid for the inclusions
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_cell_cubes/2);
  auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();

  //alternative computation of effective parameters
  HelmholtzDiscretization< LeafGridView, 1>::SpaceType coarse_space(ref_leafView);
  HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::EllipticCellProblem elliptic_cell(coarse_space, cell_grid_ref, a_diel, stabil, filter_inclusion);
  ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
  HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg);
  auto a_eff = elliptic_cell.effective_matrix();
  std::cout<< "effective inverse permittivity " <<std::endl;
  std::cout<< a_eff <<std::endl;
  auto mu_eff = incl_cell.effective_param();
  std::cout<< "effective permeability " << mu_eff <<std::endl;

  //build piece-wise constant functions
  const LambdaFct a_eff_fct([a_eff, left_inner, right_inner](LambdaFct::DomainType xx){if (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner)
                                                                                         return a_eff[0][0];
                                                                                       else return 1.0;}, 0);
  const LambdaFct mu_eff_real_fct([mu_eff, left_inner, right_inner](LambdaFct::DomainType xx){if (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner)
                                                                                               return mu_eff.real();
                                                                                             else return 1.0;}, 0);
  const LambdaFct mu_eff_imag_fct([mu_eff, left_inner, right_inner](LambdaFct::DomainType xx){if (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner)
                                                                                               return mu_eff.imag();
                                                                                             else return 0.0;}, 0);

  //assemble and solve homogenized system
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hom_bdry_info;
  HelmholtzDiscretization< LeafGridView, 1> homdiscr(ref_leafView, hom_bdry_info, a_eff_fct, zero, mu_eff_real_fct, mu_eff_imag_fct, wavenumber, bdry_real, bdry_imag, zero, zero);
  std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
  homdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_hom;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  //homdiscr.solve(sol_hom);
  typedef Dune::Stuff::LA::Solver< HelmholtzDiscretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration hom_options = SolverType::options("bicgstab.diagonal");
  hom_options.set("max_iter", "100000", true);
  SolverType hom_solver(homdiscr.system_matrix());
  hom_solver.apply(homdiscr.rhs_vector(), sol_hom, hom_options);
  homdiscr.visualize(sol_hom, "homogenized_solution_k"+std::to_string((int(wavenumber)))+"_256", "discrete_solution_"+std::to_string(int(wavenumber)));
  //make discrete function
  typedef HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType DiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal_hom(sol_hom.size());
  Stuff::LA::Container< double >::VectorType solimag_hom(sol_hom.size());
  solreal_hom.backend() = sol_hom.backend().real();
  solimag_hom.backend() = sol_hom.backend().imag();
  std::vector< HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType > sol_hom_ref_func({HelmholtzDiscretization< LeafGridView, 1 >::DiscreteFunctionType(homdiscr.space(), solreal_hom),
                                                                                               HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType(homdiscr.space(), solimag_hom)});

*/

  for (unsigned int num_macro_cubes : {8, 12, 16, 24, 32, 48, 64, 96}) {
  Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(left_outer, right_outer, num_macro_cubes);
  auto macro_leafView = macro_grid_provider.grid().leafGridView();
  unsigned int num_cell_cubes = num_macro_cubes;
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_cell_cubes);
  auto& cell_grid = cell_grid_provider.grid();
  //grid for the inclusions
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_left, d_right, num_cell_cubes/2);
  auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();

  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hmmbdry_info;

  typedef HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1 > HMMHelmholtzType;
  HMMHelmholtzType hmmhelmholtz(macro_leafView, cell_grid, inclusion_leafView, hmmbdry_info, a_diel, a_incl_real, a_incl_imag, wavenumber, bdry_real, bdry_imag,
                                filter_scatterer, filter_outside, filter_inclusion, stabil, one, one, one);

  std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_cell_cubes<< " cubes per dim on the micro grid"<< std::endl;
  hmmhelmholtz.assemble();
  std::cout<< "hmm solving" <<std::endl;
  Dune::Stuff::LA::Container< complextype >::VectorType sol_hmm;
  hmmhelmholtz.solve(sol_hmm);
  HMMHelmholtzType::RealVectorType solreal_hmm(sol_hmm.size());
  solreal_hmm.backend() = sol_hmm.backend().real();

  if(num_macro_cubes == 96) {
    HMMHelmholtzType::DiscreteFunctionType sol_real_func(hmmhelmholtz.space(), solreal_hmm, "solution_real_part");
    sol_real_func.visualize("hmm_solution_k"+std::to_string((int(wavenumber)))+"_"+std::to_string(num_macro_cubes)+"_real");
  }

  typedef GDT::ProlongedFunction< HMMHelmholtzType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
  typedef Stuff::Functions::Difference< ExpressionFct, HMMHelmholtzType::DiscreteFunctionType > DifferenceFct;
  typedef Stuff::Functions::Difference< ExpressionFct, ProlongedDiscrFct >                 ProlongedDifferenceFct;

  std::cout<< "corrector computation" <<std::endl;
  HMMHelmholtzType::DiscreteFunctionType macro_sol(hmmhelmholtz.space(), solreal_hmm);
  std::vector< HMMHelmholtzType::DiscreteFunctionType > macro_solution(2, macro_sol);
  typedef Dune::GDT::PeriodicCorrector< HMMHelmholtzType::DiscreteFunctionType, HMMHelmholtzType::EllipticCellDiscreteFctType > EllipticCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMHelmholtzType::DiscreteFunctionType, HMMHelmholtzType::InclusionCellDiscreteFctType > InclusionCorrectorType;
  std::pair< EllipticCorrectorType, InclusionCorrectorType > correctors(hmmhelmholtz.solve_and_correct(macro_solution));

  typedef Stuff::Functions::Difference< HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType, ProlongedDiscrFct > RefDifferenceFct;
  ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
  ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);

  //errors to reference solution
  RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
  RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);

  std::cout<< "macroscopic errors on reference grid" <<std::endl;
  Dune::GDT::Products::L2< LeafGridView > l2_product_operator_ref(ref_leafView);
  Dune::GDT::Products::H1Semi< LeafGridView > h1_product_operator_ref(ref_leafView);
  std::cout<< "L2 error: " << std::sqrt(l2_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                        + l2_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
  std::cout<< "H1 seminorm "<< std::sqrt(h1_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                            + h1_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
  std::cout<< "errors to zeroth order approximation" <<std::endl;
  std::cout<< "L2: "<< hmmhelmholtz.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "l2") <<std::endl;
 // std::cout<< "H1 seminorm: "<< hmmhelmholtz.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "h1semi") <<std::endl;

  if (num_macro_cubes == 96) {
    std::cout<< "visualization" <<std::endl;
    typedef Dune::GDT::DeltaCorrectorHelmholtz< HMMHelmholtzType::DiscreteFunctionType, HMMHelmholtzType::EllipticCellDiscreteFctType, HMMHelmholtzType::InclusionCellDiscreteFctType > DeltaCorrectorType;
    DeltaCorrectorType corrector_real(correctors.first.macro_function(), correctors.first.cell_solutions(), correctors.second.cell_solutions(), filter_scatterer, filter_inclusion, wavenumber, delta, "real");
    corrector_real.visualize(ref_leafView, "delta_corrector_k"+std::to_string((int(wavenumber)))+"_"+std::to_string(num_macro_cubes)+"_real", false);
  }

  //errors to homogenized solution
/*  RefDifferenceFct hom_reference_error_real(sol_hom_ref_func[0], prolonged_ref_real);
  RefDifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], prolonged_ref_imag);
  Dune::GDT::Products::L2< LeafGridView > l2_product_operator_hom(ref_leafView);
  Dune::GDT::Products::H1Semi< LeafGridView > h1_product_operator_hom(ref_leafView);
  std::cout<< "L2 error: " << std::sqrt(l2_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                        + l2_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag)) <<std::endl;
  std::cout<< "H1 seminorm "<< std::sqrt(h1_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                            + h1_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag)) <<std::endl;
*/
  }//end for loop num_macro_cubes

  }//end for loop wavenumber


  } //end try block

  catch(...) {
    std::cout<< "something went wrong"<<std::endl;
  }

  return 0;
}
