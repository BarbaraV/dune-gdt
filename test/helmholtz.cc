// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <dune/common/parallel/mpihelper.hh>

#include <dune/grid/alugrid.hh>

#include <dune/stuff/common/ranges.hh>
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

  //============================================================================================
  // test direct discretization on a Helmholtz problem
  //============================================================================================

  //instantiate  reference grid
 /* unsigned int num_cubes = 256;
  const double left_outer = 0.0;
  const double right_outer = 1.0;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_cubes);
  auto leafView = grid_provider.grid().leafGridView();

  const double wavenumber = 32.0;
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
  typedef Dune::Stuff::LA::Solver< HelmholtzDiscretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
  options.set("max_iter", "50000", true);
  discr_direct.solve(sol_direct, options);
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

  //some general typedefs
  typedef Dune::Fem::PeriodicLeafGridPart< GridType >::GridViewType                PeriodicViewType;
  typedef PeriodicViewType::template Codim< 0 >::Entity                            PeriodicEntityType;
  typedef Stuff::Functions::Constant< PeriodicEntityType, double, 2, double, 1>    PerConstFct;

  typedef Dune::Stuff::LA::Solver< HelmholtzDiscretization< LeafGridView, 1>::MatrixTypeComplex > SolverType;
  typedef HelmholtzDiscretization< LeafGridView, 1>::DiscreteFunctionType                         DiscreteFct;
  typedef HelmholtzDiscretization< LeafGridView, 1>::SpaceType                                    SpaceType;

  typedef HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1 > HMMHelmholtzType;

  typedef Dune::GDT::PeriodicCorrector< HMMHelmholtzType::DiscreteFunctionType, HMMHelmholtzType::EllipticCellDiscreteFctType > EllipticCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMHelmholtzType::DiscreteFunctionType, HMMHelmholtzType::InclusionCellDiscreteFctType > InclusionCorrectorType;

  typedef GDT::ProlongedFunction< HMMHelmholtzType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
  typedef Stuff::Functions::Difference< DiscreteFct, DiscreteFct >                              DifferenceFct;
  typedef Stuff::Functions::Difference< DiscreteFct, ProlongedDiscrFct >                        RefDifferenceFct;

  //======================================================================================================================================================================
  // TestCase: Computational domain (0.25, 0.75)^2;
  //           Scatterer (0.375, 0.625)^2 or (0.375, 0.625)x(0.25, 0.75) (see below);
  //           Inclusions (0.25, 0.75)^2 in the unit cube
  //======================================================================================================================================================================

  //macroscopic geometry
  const double left_outer = 0.25;
  const double right_outer = 0.75;
  const double left_inner = 0.375;
  const double right_inner = 0.625;
  const double size_scatterer = right_inner - left_inner;
  const FieldVector< double, 2 > cube_center(0.5);

  //inclusion geometry
  const double d_right = 0.75;
  const double d_left = 0.25;
  const double size_inclusion = d_right - d_left;

  //filter returning true when NOT in the inclusion
  const std::function< bool(const PeriodicViewType& , const PeriodicEntityType& ) > filter_inclusion
          = [size_inclusion, cube_center](const PeriodicViewType& cell_grid_view, const PeriodicEntityType& periodic_entity) -> bool
            {const auto xx = periodic_entity.geometry().center();
             return !((xx-cube_center).infinity_norm() <= 0.5*size_inclusion);};

  //material parameters
  PerConstFct a_diel(10.0);
  PerConstFct a_incl_real(10.0);
  PerConstFct a_incl_imag(-0.01);
  PerConstFct stabil(0.0001);

  //=====================================================================================================================================================================
  //Example 1: Dependence of mu_eff on the wave number
  //=====================================================================================================================================================================
/*
  //grid for the inclusions
  unsigned int num_ref_incl_cubes = 256;
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_incl_cubes);
  auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();

  //vectors which store the wave numbers for the computation
  auto range1 = Stuff::Common::valueRange(15.0, 27.0, 1.0);
  auto range2 = Stuff::Common::valueRange(27.0, 29.0, 0.1);
  auto range3 = Stuff::Common::valueRange(29.0, 62.0, 1.0);
  auto range4 = Stuff::Common::valueRange(62.0, 63.6, 0.1);
  auto range5 = Stuff::Common::valueRange(64.0, 69.0, 1.0);

  std::ofstream output("output_hmmhelmholtz_mueff_"+std::to_string(int(num_ref_incl_cubes))+".txt");
  output << "k" << "\t" << "Re(mu_eff)" << "\t" << "Im(mu_eff)" << "\n";

  try {
    for (auto& range : {range1, range2, range3, range4, range5}) {
      for (auto wavenumber : range) {
        std::cout<< "computing mu_eff for wavenumber " << wavenumber <<std::endl;
        ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
        HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg);
        auto mu_eff = incl_cell.effective_param();
        output << wavenumber << "\t" << mu_eff.real() << "\t" << mu_eff.imag() << "\n";
      }//end for loop wavenumber
    }//end for loop ranges
  }//end try block
*/

  //=====================================================================================================================================================================
  // Example 2: Comparison of the HMM approximation to a homogenized reference solution and the resolution condition
  // TestCase: quadratic scatterer, incoming plane wave from the right
  //=====================================================================================================================================================================

  //filters for the scatterer
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [size_scatterer, cube_center](const LeafGridView& grid_view, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [size_scatterer, cube_center](const LeafGridView& grid_view, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return !((xx-cube_center).infinity_norm() <= 0.5*size_scatterer);};

  //select the wave number by commenting out the corresponding line
  double wavenumber;
  wavenumber = 34.0;
  //wavenumber = 48.0;
  //wavenumber = 68.0;

  //bdry condition
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

  //instantiate  reference grids
  unsigned int num_ref_cubes = 512;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  unsigned int num_ref_cell_cubes = num_ref_cubes;
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
  auto& cell_grid_ref = cell_grid_provider_ref.grid();
  //grid for the inclusions
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_cell_cubes/2);
  auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();

  try {
    // computation of effective parameters
    SpaceType coarse_space(ref_leafView);
    HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::EllipticCellProblem elliptic_cell(coarse_space, cell_grid_ref, a_diel, stabil, filter_inclusion);
    ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
    HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg);
    std::cout<< "computing a_eff" << std::endl;
    auto a_eff = elliptic_cell.effective_matrix();
    std::cout << "computing mu_eff" << std::endl;
    auto mu_eff = incl_cell.effective_param();

    //build piece-wise constant functions
    const LambdaFct a_eff_fct([a_eff, size_scatterer, cube_center](LambdaFct::DomainType xx){if ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                                                                               return a_eff[0][0];
                                                                                             else return 1.0;}, 0);
    const LambdaFct mu_eff_real_fct([mu_eff, size_scatterer, cube_center](LambdaFct::DomainType xx){if ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                                                                                      return mu_eff.real();
                                                                                                    else return 1.0;}, 0);
    const LambdaFct mu_eff_imag_fct([mu_eff, size_scatterer, cube_center](LambdaFct::DomainType xx){if ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                                                                                      return mu_eff.imag();
                                                                                                    else return 0.0;}, 0);

    //assemble and solve homogenized system
    DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hom_bdry_info;
    HelmholtzDiscretization< LeafGridView, 1> homdiscr(ref_leafView, hom_bdry_info, a_eff_fct, zero, mu_eff_real_fct, mu_eff_imag_fct, wavenumber, bdry_real, bdry_imag, zero, zero);
    std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
    std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
    homdiscr.assemble();
    Dune::Stuff::LA::Container< complextype >::VectorType sol_hom;
    std::cout<<"solving with bicgstab.ilut"<<std::endl;
    Dune::Stuff::Common::Configuration hom_options = SolverType::options("bicgstab.ilut");
    hom_options.set("max_iter", "100000", true);
    homdiscr.solve(sol_hom, hom_options);
    //make discrete function
    Stuff::LA::Container< double >::VectorType solhom_real_ref(sol_hom.size());
    std::vector< DiscreteFct > sol_hom_ref_func(2, DiscreteFct(SpaceType(ref_leafView), solhom_real_ref));
    sol_hom_ref_func[0].vector().backend() = sol_hom.backend().real();
    sol_hom_ref_func[1].vector().backend() = sol_hom.backend().imag();

    std::ofstream output("output_hmmhelmholtz_homerror_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_ref_cubes))+".txt");
    output<< "num_macro_cubes" << "\t" << "L2" << "\t" << "H1 seminorm" << "\n";

    std::vector< unsigned int > num_macro_cubes_range({8, 12, 16, 24, 32, 48, 64, 96});
    if (wavenumber == 68.0)
      num_macro_cubes_range = {8, 16, 24, 32, 48, 64, 96};

    for (unsigned int num_macro_cubes : num_macro_cubes_range) {
      Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(left_outer, right_outer, num_macro_cubes);
      auto macro_leafView = macro_grid_provider.grid().leafGridView();
      unsigned int num_cell_cubes = num_macro_cubes;
      Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_cell_cubes);
      auto& cell_grid = cell_grid_provider.grid();
      //grid for the inclusions
      Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_left, d_right, num_cell_cubes/2);
      auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();

      DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hmmbdry_info;
      HMMHelmholtzType hmmhelmholtz(macro_leafView, cell_grid, inclusion_leafView, hmmbdry_info, a_diel, a_incl_real, a_incl_imag, wavenumber, bdry_real, bdry_imag,
                                    filter_scatterer, filter_outside, filter_inclusion, stabil, one, one, one);

      std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_cell_cubes<< " cubes per dim on the micro grid"<< std::endl;
      hmmhelmholtz.assemble();
      std::cout<< "hmm solving and corrector computation" <<std::endl;
      HMMHelmholtzType::RealVectorType solreal_hmm;
      HMMHelmholtzType::DiscreteFunctionType macro_sol(hmmhelmholtz.space(), solreal_hmm);
      std::vector< HMMHelmholtzType::DiscreteFunctionType > macro_solution(2, macro_sol);
      std::pair< EllipticCorrectorType, InclusionCorrectorType > correctors(hmmhelmholtz.solve_and_correct(macro_solution));

      ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
      ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);

      //error computation
      std::cout<< "error computation" <<std::endl;
      RefDifferenceFct hom_reference_error_real(sol_hom_ref_func[0], prolonged_ref_real);
      RefDifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], prolonged_ref_imag);
      Dune::GDT::Products::L2< LeafGridView > l2_product_operator_hom(ref_leafView);
      Dune::GDT::Products::H1Semi< LeafGridView > h1_product_operator_hom(ref_leafView);
      const double l2 = std::sqrt(l2_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                   + l2_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag));
      const double h1_semi = std::sqrt(h1_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                        + h1_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag));

      output<< num_macro_cubes << "\t" << l2 << "\t" << h1_semi << "\n";
    }//end for loop
  }//end try block


  //==============================================================================================================================================================================
  //Example 3: Errors between reference solution, reference homogenized solution and HMM approximation(s)
  //           Confirmation of convergence rates
  //           Comparison of transmission and band gap
  //==============================================================================================================================================================================
/*
  const double delta = 1.0/32.0;
  double intpart;

  //select wave number
  double wavenumber;
  //wavenumber = 29.0;   //band gap
  wavenumber = 38.0;  //transmission

  //some identifiers
  std::string scatterer_shape;
  std::string bdry_cond;
*/
  //==============================================================================
  // TestCases: -Quadratic scatterer with plane wave in x_0-direction
  //            -Quadratic scatterer with plane wave in direction (0.6, 0.8)
  //            -stripe-shape scatterer with plane wave in direction (0.6, 0.8)
  //==============================================================================
/*
  //quadratic scatterer
  //---------------------

  //filters for the scatterer
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [size_scatterer, cube_center](const LeafGridView& grid_view, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [size_scatterer, cube_center](const LeafGridView& grid_view, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return !((xx-cube_center).infinity_norm() <= 0.5*size_scatterer);};

  //delta dependent "diffusion" parameter
  const LambdaFct a_real([size_scatterer, cube_center, delta, &intpart, size_inclusion](LambdaFct::DomainType x)
                            {if ((x-cube_center).infinity_norm() <= 0.5*size_scatterer) { //inside scatterer
                               LambdaFct::DomainType y;
                               y[0] = std::modf(x[0]/delta, &intpart);
                               y[1] = std::modf(x[1]/delta, &intpart);
                               if ((y-cube_center).infinity_norm() <= 0.5*size_inclusion)
                                 return delta*delta*10.0;   //a_incl_real
                               return 10.0;   //a_diel
                             }
                             return 1.0;}, 0);
  const LambdaFct a_imag([size_scatterer, cube_center, delta, &intpart, size_inclusion](LambdaFct::DomainType x)
                            {if ((x-cube_center).infinity_norm() <= 0.5*size_scatterer) { //inside scatterer
                               LambdaFct::DomainType y;
                               y[0] = std::modf(x[0]/delta, &intpart);
                               y[1] = std::modf(x[1]/delta, &intpart);
                               if ((y-cube_center).infinity_norm() <= 0.5*size_inclusion)
                                 return -0.01*delta*delta; //a_incl_imag
                               return 0.0;
                             }
                             return 0.0;}, 0);
  scatterer_shape = "quadr";*/
  //-----------------------------------------------------------------------------------------------------------
/*
  //stripe-shaped scatterer
  //----------------------
  //filters for the scatterer
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [size_scatterer, cube_center](const LeafGridView& grid_view, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return (std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [size_scatterer, cube_center](const LeafGridView& grid_view, const EntityType& macro_entity)
            {const auto xx = macro_entity.geometry().center();
             return !(std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer);};

  //delta dependent "diffusion" parameter
  const LambdaFct a_real([size_scatterer, cube_center, delta, &intpart, size_inclusion](LambdaFct::DomainType x)
                            {if (std::abs(x[0]-cube_center[0]) <= 0.5*size_scatterer) { //inside scatterer
                               LambdaFct::DomainType y;
                               y[0] = std::modf(x[0]/delta, &intpart);
                               y[1] = std::modf(x[1]/delta, &intpart);
                               if ((y-cube_center).infinity_norm() <= 0.5*size_inclusion)
                                 return delta*delta*10.0;   //a_incl_real
                               return 10.0;   //a_diel
                             }
                             return 1.0;}, 0);
  const LambdaFct a_imag([size_scatterer, cube_center, delta, &intpart, size_inclusion](LambdaFct::DomainType x)
                            {if (std::abs(x[0]-cube_center[0]) <= 0.5*size_scatterer) { //inside scatterer
                               LambdaFct::DomainType y;
                               y[0] = std::modf(x[0]/delta, &intpart);
                               y[1] = std::modf(x[1]/delta, &intpart);
                               if ((y-cube_center).infinity_norm() <= 0.5*size_inclusion)
                                 return -0.01*delta*delta; //a_incl_imag
                               return 0.0;
                             }
                             return 0.0;}, 0);
  scatterer_shape = "stripe";*/
  //-------------------------------------------------------------------------------------------------------------------
/*
  //plane wave in x_0 direction
  //----------------------------

  //bdry condition
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
  bdry_cond = "plane";*/
  //---------------------------------------------------------------------------------------------------------------------------------------------------
/*
  //plane wave in (0.6, 0.8) direction
  //-----------------------------------

  //bdry condition
  Dune::FieldVector< double, 2 > wave_normal;
  wave_normal[0] = 0.6;
  wave_normal[1] = 0.8;

  const LambdaFct bdry_real([wavenumber, left_outer, right_outer, wave_normal](LambdaFct::DomainType x){if (std::abs(x[0] - right_outer) < 1e-12)
                                                                    return -(wave_normal[0]+1)*wavenumber*std::sin(wavenumber*x.dot(wave_normal));
                                                                  if (std::abs(x[0] - left_outer) < 1e-12)
                                                                    return wavenumber*(wave_normal[0] - 1)*std::sin(wavenumber*x.dot(wave_normal));
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    return (wave_normal[1]-1)*wavenumber*std::sin(wavenumber*x.dot(wave_normal));
                                                                  if (std::abs(x[1] - right_outer) < 1e-12)
                                                                    return -(wave_normal[1]+1)*wavenumber*std::sin(wavenumber*x.dot(wave_normal));}, 0);
  const LambdaFct bdry_imag([wavenumber, left_outer, right_outer, wave_normal](LambdaFct::DomainType x){if (std::abs(x[0] - right_outer) < 1e-12)
                                                                    return -(wave_normal[0]+1)*wavenumber*std::cos(wavenumber*x.dot(wave_normal));
                                                                  if (std::abs(x[0] - left_outer) < 1e-12)
                                                                    return wavenumber*(wave_normal[0] - 1)*std::cos(wavenumber*x.dot(wave_normal));
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    return (wave_normal[1]-1)*wavenumber*std::cos(wavenumber*x.dot(wave_normal));
                                                                  if (std::abs(x[1] - right_outer) < 1e-12)
                                                                    return -(wave_normal[1]+1)*wavenumber*std::cos(wavenumber*x.dot(wave_normal));}, 0); 
  bdry_cond = "oblique";*/
  //------------------------------------------------------------------------------------------------------------------------------------------------------
/*
  //reference solution
  //instantiate  reference grid
  unsigned int num_ref_cubes = 512;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  std::ofstream output("output_hmmhelmholtz_error_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_ref_cubes))+"_"+scatterer_shape+"_"+bdry_cond+".txt");

  try {
    //compute reference solution
    std::cout<< "wavenumber "<< wavenumber <<std::endl;
    DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > ref_bdry_info;
    HelmholtzDiscretization< LeafGridView, 1> refdiscr(ref_leafView, ref_bdry_info, a_real, a_imag, one, zero, wavenumber, bdry_real, bdry_imag, zero, zero);
    std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
    std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< refdiscr.space().mapper().size() <<std::endl;
    refdiscr.assemble();
    Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
    std::cout<<"solving with bicgstab.ilut"<<std::endl;
    Dune::Stuff::Common::Configuration ref_options = SolverType::options("bicgstab.ilut");
    ref_options.set("max_iter", "200000", true);
    ref_options.set("precision", "1e-6", true);
    refdiscr.solve(sol_ref, ref_options);
    //make discrete function
    Stuff::LA::Container< double >::VectorType sol_real_ref(sol_ref.size());
    std::vector< DiscreteFct > sol_ref_func(2, DiscreteFct(SpaceType(ref_leafView), sol_real_ref));
    sol_ref_func[0].vector().backend() = sol_ref.backend().real();
    sol_ref_func[1].vector().backend() = sol_ref.backend().imag();
    sol_ref_func[0].visualize("ref_discrete_solution_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_ref_cubes))+"_"+scatterer_shape+"_"+bdry_cond+"_real");

    //compute reference homogenized solution
    unsigned int num_ref_cell_cubes = num_ref_cubes;
    Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
    auto& cell_grid_ref = cell_grid_provider_ref.grid();
    //grid for the inclusions
    Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_cell_cubes/2);
    auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();

    // computation of effective parameters
    SpaceType coarse_space(ref_leafView);
    HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::EllipticCellProblem elliptic_cell(coarse_space, cell_grid_ref, a_diel, stabil, filter_inclusion);
    ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
    HMMHelmholtzDiscretization< LeafGridView, GridType, LeafGridView, 1>::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg);
    std::cout<< "computing a_eff" << std::endl;
    auto a_eff = elliptic_cell.effective_matrix();
    std::cout << "computing mu_eff" << std::endl;
    auto mu_eff = incl_cell.effective_param();

    //build piece-wise constant functions
    const LambdaFct a_eff_fct([a_eff, size_scatterer, cube_center, scatterer_shape](LambdaFct::DomainType xx)
                                                                                      {if ((scatterer_shape == "quadr" && (xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                                                                            || (scatterer_shape == "stripe" &&std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer))
                                                                                         return a_eff[0][0];
                                                                                       else return 1.0;}, 0);
    const LambdaFct mu_eff_real_fct([mu_eff, size_scatterer, cube_center, scatterer_shape](LambdaFct::DomainType xx)
                                                                                             {if ((scatterer_shape == "quadr" && (xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                                                                                   || (scatterer_shape == "stripe" &&std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer))
                                                                                                return mu_eff.real();
                                                                                              else return 1.0;}, 0);
    const LambdaFct mu_eff_imag_fct([mu_eff, size_scatterer, cube_center, scatterer_shape](LambdaFct::DomainType xx)
                                                                                             {if ((scatterer_shape == "quadr" && (xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                                                                                   || (scatterer_shape == "stripe" &&std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer))
                                                                                                return mu_eff.imag();
                                                                                              else return 0.0;}, 0);

    //assemble and solve homogenized system
    DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hom_bdry_info;
    HelmholtzDiscretization< LeafGridView, 1> homdiscr(ref_leafView, hom_bdry_info, a_eff_fct, zero, mu_eff_real_fct, mu_eff_imag_fct, wavenumber, bdry_real, bdry_imag, zero, zero);
    std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
    std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
    homdiscr.assemble();
    Dune::Stuff::LA::Container< complextype >::VectorType sol_hom;
    std::cout<<"solving with bicgstab.ilut"<<std::endl;
    Dune::Stuff::Common::Configuration hom_options = SolverType::options("bicgstab.ilut");
    hom_options.set("max_iter", "100000", true);
    homdiscr.solve(sol_hom, hom_options);
    //make discrete function
    Stuff::LA::Container< double >::VectorType solhom_real_ref(sol_hom.size());
    std::vector< DiscreteFct > sol_hom_ref_func(2, DiscreteFct(SpaceType(ref_leafView), solhom_real_ref));
    sol_hom_ref_func[0].vector().backend() = sol_hom.backend().real();
    sol_hom_ref_func[1].vector().backend() = sol_hom.backend().imag();
    sol_hom_ref_func[0].visualize("ref_hom_solution_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_ref_cubes))+"_"+scatterer_shape+"_"+bdry_cond+"_real");

    //compute homogenization error
    std::cout<< "homogenization error computation" <<std::endl;
    DifferenceFct hom_reference_error_real(sol_hom_ref_func[0], sol_ref_func[0]);
    DifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], sol_ref_func[1]);
    Dune::GDT::Products::L2< LeafGridView > l2_product_operator_hom(ref_leafView);
    Dune::GDT::Products::H1Semi< LeafGridView > h1_product_operator_hom(ref_leafView);
    const double l2 = std::sqrt(l2_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                 + l2_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag));
    const double h1_semi = std::sqrt(h1_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                      + h1_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag));

    output<< "Homogenization error" << "\n" << "L2: " << "\t" << l2 << "\t" << "H1 seminorm: " << "\t" << h1_semi << "\n" << "\n";

    output << "errors reference solutions to HMM" << "\n" << "num_macro_cubes" << "\t" << "L2 hom" << "\t" << "H1 semi hom" << "\t" << "L2 ref" << "\t"  << "L2 corrector" << "\n";

    for (unsigned int num_macro_cubes : {8, 16, 24, 32, 48, 64}) {
      Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(left_outer, right_outer, num_macro_cubes);
      auto macro_leafView = macro_grid_provider.grid().leafGridView();
      unsigned int num_cell_cubes = num_macro_cubes;
      Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_cell_cubes);
      auto& cell_grid = cell_grid_provider.grid();
      //grid for the inclusions
      Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_left, d_right, num_cell_cubes/2);
      auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();

      DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hmmbdry_info;
      HMMHelmholtzType hmmhelmholtz(macro_leafView, cell_grid, inclusion_leafView, hmmbdry_info, a_diel, a_incl_real, a_incl_imag, wavenumber, bdry_real, bdry_imag,
                                    filter_scatterer, filter_outside, filter_inclusion, stabil, one, one, one);

      std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_cell_cubes<< " cubes per dim on the micro grid"<< std::endl;
      hmmhelmholtz.assemble();
      std::cout<< "hmm solving and corrector computation" <<std::endl;
      HMMHelmholtzType::RealVectorType solreal_hmm;
      HMMHelmholtzType::DiscreteFunctionType macro_sol(hmmhelmholtz.space(), solreal_hmm);
      std::vector< HMMHelmholtzType::DiscreteFunctionType > macro_solution(2, macro_sol);
      std::pair< EllipticCorrectorType, InclusionCorrectorType > correctors(hmmhelmholtz.solve_and_correct(macro_solution));

      //prepare error functions
      ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
      ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);
      RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
      RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);
      RefDifferenceFct hom_reference_error_real(sol_hom_ref_func[0], prolonged_ref_real);
      RefDifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], prolonged_ref_imag);

      Dune::GDT::Products::L2< LeafGridView > l2_product_operator_ref(ref_leafView);
      Dune::GDT::Products::H1Semi< LeafGridView > h1_product_operator_ref(ref_leafView);

      //errors to homogenized solution
      std::cout<< "errors to homogenized solution" <<std::endl;
      const double l2_hom = std::sqrt(l2_product_operator_ref.apply2(hom_reference_error_real, hom_reference_error_real)
                                      + l2_product_operator_ref.apply2(hom_reference_error_imag, hom_reference_error_imag));
      const double h1_semi_hom = std::sqrt(h1_product_operator_ref.apply2(hom_reference_error_real, hom_reference_error_real)
                                           + h1_product_operator_ref.apply2(hom_reference_error_imag, hom_reference_error_imag));

      //errors to reference solution
      std::cout<< "macroscopic error on reference grid" <<std::endl;
      double l2_ref = std::sqrt(l2_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                 + l2_product_operator_ref.apply2(reference_error_imag, reference_error_imag));
      std::cout<< "error to zeroth order approximation" <<std::endl;
      double l2_correc = hmmhelmholtz.reference_error(sol_ref_func, correctors.first, correctors.second, delta, "l2");

      output<< num_macro_cubes << "\t" << l2_hom << "\t" << h1_semi_hom << "\t" << l2_ref << "\t" << l2_correc << "\n";

      if (num_macro_cubes == 64) {
        std::cout<< "visualization" <<std::endl;
        macro_solution[0].visualize("hmm_sol_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_macro_cubes))+"_"+scatterer_shape+"_"+bdry_cond+"_real");
        typedef Dune::GDT::DeltaCorrectorHelmholtz< HMMHelmholtzType::DiscreteFunctionType, HMMHelmholtzType::EllipticCellDiscreteFctType,
                                                    HMMHelmholtzType::InclusionCellDiscreteFctType > DeltaCorrectorType;
        DeltaCorrectorType corrector_real(correctors.first.macro_function(), correctors.first.cell_solutions(), correctors.second.cell_solutions(),
                                          filter_scatterer, filter_inclusion, wavenumber, delta, "real");
        corrector_real.visualize(ref_leafView, "delta_correc_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_macro_cubes))+"_on"+std::to_string(int(num_ref_cubes))
                                                               +"_"+scatterer_shape+"_"+bdry_cond+"_real", false);
      }

    }//end for loop num_macro_cubes
  }//end try block
*/

  catch(Dune::Exception& ee) {
    std::cout<< ee.what() <<std::endl;
  }

  catch(...) {
    std::cout<< "something went wrong" <<std::endl;
  }

  return 0;
}
