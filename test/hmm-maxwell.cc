// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <vector>
#include <string>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/exceptions.hh>

#include <dune/grid/alugrid.hh>

#include <dune/stuff/functions/global.hh>
#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>
#include <dune/stuff/grid/provider/cube.hh>

#include <dune/gdt/discretefunction/prolonged.hh>
#include <dune/gdt/discretefunction/corrector.hh>
#include <dune/gdt/products/weightedl2.hh>
#include <dune/gdt/products/hcurl.hh>
#include <dune/gdt/localevaluation/divdiv.hh>
#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/operators/projections.hh>

#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>

#include "hmm-maxwell.hh"
#include "curlcurldiscretization.hh"
#include "hmm-discretization.hh"

using namespace Dune;

int main(int argc, char** argv) {
  Fem::MPIManager::initialize(argc, argv);

  // some typedefs
  typedef ALUGrid< 3, 3, simplex, conforming > GridType;
  typedef GridType::LeafGridView LeafGridView;
  typedef LeafGridView::Codim< 0 >::Entity EntityType;

  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 1 >  ConstantFct;
  typedef Stuff::GlobalLambdaFunction< EntityType, double, 3, double, 1 > LambdaFct;
  typedef Stuff::GlobalLambdaFunction< EntityType, double, 3, double, 3 > VectorLambdaFct;
  typedef Stuff::GlobalLambdaFunction< EntityType, double, 3, double, 3, 3 > MatrixLambdaFct;
  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 1 > ExpressionFct;
  typedef std::complex< double > complextype;

  const ConstantFct one(1.0);
  const ConstantFct zero(0.0);
  Stuff::Functions::Constant< EntityType, double, 3, double, 3, 3 >  zero_matrix(0.0);

  //===============================================================================================================================
  // HMM Discretization
  //===============================================================================================================================

  //some general typedefs
  typedef Dune::Fem::PeriodicLeafGridPart< GridType >::GridViewType 		   PeriodicViewType;
  typedef PeriodicViewType::template Codim< 0 >::Entity             		   PeriodicEntityType;
  typedef Stuff::Functions::Constant< PeriodicEntityType, double, 3, double, 1>    PerConstFct;

  typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, false, false >::MatrixTypeComplex > SolverType;
  typedef ScatteringDiscretization< LeafGridView, 1, false, false >::DiscreteFunctionType                         DiscreteFct;
  typedef ScatteringDiscretization< LeafGridView, 1, false, false >::SpaceType                                    SpaceType;

  typedef HMMMaxwellDiscretization< LeafGridView, GridType, LeafGridView, 1 > HMMMaxwellType;

  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::CurlCellDiscreteFctType >      CurlCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::IdCellDiscreteFctType >        IdCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::InclusionCellDiscreteFctType > InclCorrectorType;

  typedef GDT::ProlongedFunction< HMMMaxwellType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
  typedef Stuff::Functions::Difference< DiscreteFct, DiscreteFct >                            DifferenceFct;
  typedef Stuff::Functions::Difference< DiscreteFct, ProlongedDiscrFct >                      RefDifferenceFct;

  //======================================================================================================================================================================
  // TestCase: Computational domain (0, 1)^3;
  //           Scatterer (0.25, 0.75)^3 or (0.25, 0.75)x(0, 1)^2 (see below);
  //           Inclusions (0.25, 0.75)^3 in the unit cube
  //======================================================================================================================================================================

  //macroscopic geometry
  const double left_outer = 0.0;
  const double right_outer = 1.0;
  const double left_inner = 0.25;
  const double right_inner = 0.75;
  const double size_scatterer = right_inner - left_inner;
  const FieldVector< double, 3 > cube_center(0.5);

  //inclusion geometry
  double d_right = 0.75;
  double d_left = 0.25;
  const double size_inclusion = d_right - d_left;

  //filter returning true when NOT in the inclusion
  const std::function< bool(const PeriodicViewType& , const PeriodicEntityType& ) > filter_inclusion
          = [size_inclusion, cube_center](const PeriodicViewType& cell_grid_view, const PeriodicEntityType& periodic_entity) -> bool
            {const auto xx = periodic_entity.geometry().center();
             return !((xx-cube_center).infinity_norm() <= 0.5*size_inclusion);};

  //material parameters
  PerConstFct a_diel(1.0);
  PerConstFct a_incl_real(1.0);
  PerConstFct a_incl_imag(-0.01);
  PerConstFct stabil(0.0001);

  const double div_param = 0.01;
  const double radius = 0.1;
  const double exponent = 1.0;
  LambdaFct divparam2([div_param, exponent, radius, d_left, d_right](LambdaFct::DomainType x) {
                      auto ret = distance_to_cube(x, radius, d_left, d_right);
                      return div_param * std::pow(ret, 2 * exponent);
                      }
                      , 1);

  //=====================================================================================================================================================================
  //Example 1: Dependence of mu_eff on the wave number
  //=====================================================================================================================================================================
/*
  //cell grid
  unsigned int num_ref_cell_cubes = 32;
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_ref_cell_cubes);
  auto& cell_grid = cell_grid_provider.grid();

  //grid for the inclusions
  unsigned int num_ref_incl_cubes = num_ref_cell_cubes/2;
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_incl_cubes);
  auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;

  //mu_eff outside the inclusions is independent from the wavenumber, so take any wavenumber
  std::ofstream output("output_hmmmaxwell_mueff_"+std::to_string(int(num_ref_cell_cubes))+".txt");
  {
    auto leafView = cell_grid_provider.grid().leafGridView();
    SpaceType coarse_space(leafView);
    const double frequency = 1.0;
    ConstantFct k_squared(frequency*frequency);
    HMMMaxwellType::IdCellProblem id_cell(coarse_space, cell_grid, k_squared, stabil, filter_inclusion);
    std::cout<< "computing mu_eff outside inclusion" << std::endl;
    auto mu_eff_out = id_cell.effective_matrix();
    output << "mu_eff outside the inclusion (diagonal entry)" << "\t" << mu_eff_out[0][0] << "\n" << "\n" << "mu_eff inside the inclusions (diagonal entries)" << "\n";
  }//anonymous namespace for memory reasons 

  //vectors which store the wave numbers for the computation
  auto range1 = Stuff::Common::valueRange(5.0, 8.5, 1.0);
  auto range2 = Stuff::Common::valueRange(8.5, 9.6, 0.1);
  auto range3 = Stuff::Common::valueRange(10.0, 19.0, 1.0);
  auto range4 = Stuff::Common::valueRange(19.0, 21.0, 0.1);
  auto range5 = Stuff::Common::valueRange(21.0, 26.0, 1.0);

  output << "k" << "\t" << "Re(mu_eff)" << "\t" << "Im(mu_eff)" << "\n";

  try {
    for (auto& range : {range1, range2, range3, range4, range5}) {
      for (auto wavenumber : range) {
        std::cout<< "computing mu_eff for wavenumber " << wavenumber <<std::endl;
        ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
        HMMMaxwellType::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg, curl_bdry_info);
        auto mu_eff = incl_cell.effective_matrix();
        output << wavenumber << "\t" << mu_eff[0][0][0] << "\t" << mu_eff[1][0][0] << "\n";
      }//end for loop wavenumber
    }//end for loop ranges
  }//end try block
*/

  //===================================================================================================================================================================
  // Example 2: Errors between homogenized reference solution and HMM approximation(s)
  //            Confirmation of convergence rates
  //===================================================================================================================================================================
/*
  //select wavenumber
  double wavenumber;
  wavenumber = 9.0; //frequency band gap
  //wavenumber = 12.0; //transmission
  std::cout<< "wavenumber " << wavenumber << std::endl;

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;

  //==================================================================================================
  // Test Case: -Quadratic scatterer with plane wave in x_0 direction, polarized in x_1 direction
  //==================================================================================================

  //filters for in- and outside scatterer
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [size_scatterer, cube_center](const LeafGridView& cell_grid_view, const EntityType& entity) -> bool
            {const auto xx = entity.geometry().center();
             return ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [size_scatterer, cube_center](const LeafGridView& cell_grid_view, const EntityType& entity) -> bool
            {const auto xx = entity.geometry().center();
             return !((xx-cube_center).infinity_norm() <= 0.5*size_scatterer);};
  //------------------------------------------------------------------------------------------------------------------------------

  //plane wave in x_0 direction, polarized in x_1 direction
  //-----------------------------------------------------------

  //bdry condition
  const VectorLambdaFct bdry_real([wavenumber, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[2] - right_outer) < 1e-12 || std::abs(x[2]-left_outer) < 1e-12)
                                                                    ret[1] = -1*wavenumber*std::sin(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    ret[0] = -1*wavenumber*std::sin(wavenumber*x[0]);
           							  if (std::abs(x[1] - right_outer) < 1e-12)
								    ret[0] = wavenumber*std::sin(wavenumber*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[1] = -2*wavenumber*std::sin(wavenumber*x[0]);
                                                                  return ret;}, 0);
  const VectorLambdaFct bdry_imag([wavenumber, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[2] - right_outer) < 1e-12 || std::abs(x[2]-left_outer) < 1e-12)
                                                                    ret[1] = -1*wavenumber*std::cos(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    ret[0] = -1*wavenumber*std::cos(wavenumber*x[0]);
           							  if (std::abs(x[1] - right_outer) < 1e-12)
								    ret[0] = wavenumber*std::cos(wavenumber*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[1] = -2*wavenumber*std::cos(wavenumber*x[0]);
                                                                  return ret;}, 0);
  //---------------------------------------------------------------------------------------------------------------------------------------------------

  //instantiate reference grid
  unsigned int num_ref_cubes = 48;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();

  std::ofstream output("output_hmmmaxwell_error_k"+std::to_string(int(wavenumber))+"_"+std::to_string(int(num_ref_cubes))+".txt");
  output<< "num_macro_cubes" << "\t" << "L2" << "\t" << "Hcurl seminorm" << "\t" << "L2 helmholtz" << "\n";

  try {
    //homogenized reference solution
    Dune::FieldMatrix< double, 3, 3 > a_eff;
    Dune::FieldMatrix< double, 3, 3 > mu_eff_out;
    std::vector< Dune::FieldMatrix< double, 3, 3 > > mu_eff_in(2, mu_eff_out);

    //computation of effective parameters
    {
      GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 > coarse_space(ref_leafView);
      unsigned int num_ref_cell_cubes = 32;
      //outside inclusion
      {
        Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
        auto& cell_grid_ref = cell_grid_provider_ref.grid();
        ConstantFct k_squared(wavenumber*wavenumber);
        HMMMaxwellType::IdCellProblem id_cell(coarse_space, cell_grid_ref, k_squared, stabil, filter_inclusion);
        HMMMaxwellType::CurlCellProblem curl_cell(coarse_space, cell_grid_ref, a_diel, divparam2, stabil, filter_inclusion);
        std::cout<< "computing effective inverse permittivity " <<std::endl;
        a_eff = curl_cell.effective_matrix();
        std::cout << "computing effective permeability outside inclusions" <<std::endl;
        mu_eff_out = id_cell.effective_matrix();
      }//outside inclusion 
      //inside inclusion
      {
        Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_cell_cubes/2);
        auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();
        ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
        HMMMaxwellType::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg, curl_bdry_info);
        std::cout<< "computing effective permeability inside inclusion" << std::endl;
        mu_eff_in = incl_cell.effective_matrix();
      }//inside inclusion 
    } //effective parameters

    //build piece-wise constant functions
    const MatrixLambdaFct a_eff_fct([a_eff, cube_center, size_scatterer](LambdaFct::DomainType xx)
                                      {if ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                         return a_eff;
                                       else return Stuff::Functions::internal::unit_matrix< double, 3 >();}, 0);
    const MatrixLambdaFct mu_eff_real_fct([mu_eff_out, mu_eff_in, cube_center, size_scatterer](LambdaFct::DomainType xx)
                                          {if ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer){
                                             auto ret = mu_eff_out;
                                             ret +=mu_eff_in[0];
                                             return ret;
                                           }
                                           else return Stuff::Functions::internal::unit_matrix< double, 3 >();}, 0);
    const MatrixLambdaFct mu_eff_imag_fct([mu_eff_in, cube_center, size_scatterer](LambdaFct::DomainType xx)
                                          {if ((xx-cube_center).infinity_norm() <= 0.5*size_scatterer)
                                             return mu_eff_in[1];
                                           else return Dune::FieldMatrix< double, 3, 3>();}, 0);

    //assemble and solve homogenized system
    Dune::Stuff::LA::Container< complextype >::VectorType sol_hom;
    {
      DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hom_bdry_info;
      ScatteringDiscretization< LeafGridView, 1, true, true > homdiscr(ref_leafView, hom_bdry_info, a_eff_fct, zero_matrix, wavenumber, mu_eff_real_fct, mu_eff_imag_fct, bdry_real, bdry_imag);
      std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
      std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
      homdiscr.assemble();
      std::cout<<"solving with bicgstab.diagonal"<<std::endl;
      Dune::Stuff::Common::Configuration hom_options = SolverType::options("bicgstab.diagonal");
      hom_options.set("max_iter", "200000", true);
      hom_options.set("precision", "1e-6", true);
      homdiscr.solve(sol_hom, hom_options);
    }
    //make discrete function
    Stuff::LA::Container< double >::VectorType solreal_hom(sol_hom.size());
    std::vector< DiscreteFct > sol_hom_ref_func(2, DiscreteFct(SpaceType(ref_leafView), solreal_hom));
    sol_hom_ref_func[0].vector().backend() = sol_hom.backend().real();
    sol_hom_ref_func[1].vector().backend() = sol_hom.backend().imag();

    //visualization
    {
      auto adapter = std::make_shared<Dune::Stuff::Functions::VisualizationAdapter<LeafGridView, 3, 1>>(sol_hom_ref_func[0]);
      std::unique_ptr<VTKWriter<LeafGridView>> vtk_writer = DSC::make_unique<VTKWriter<LeafGridView>>(ref_leafView, VTK::conforming);
      vtk_writer->addVertexData(adapter);
      vtk_writer->write("hom_ref_sol_k"+std::to_string((int(wavenumber)))+"_"+std::to_string(num_ref_cubes)+"_real", VTK::appendedraw); 
    }

    //HMM
    for (unsigned int num_macro_cubes : {4, 8, 12, 16}) {
      //grids
      Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(left_outer, right_outer, num_macro_cubes);
      auto macro_leafView = macro_grid_provider.grid().leafGridView();
      unsigned int num_cell_cubes = num_macro_cubes;
      Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_cell_cubes);
      auto& cell_grid = cell_grid_provider.grid();
      //grid for the inclusions
      Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_left, d_right, num_cell_cubes/2);
      auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();

      DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hmmbdry_info;
      HMMMaxwellType hmmmaxwell(macro_leafView, cell_grid, inclusion_leafView, hmmbdry_info, curl_bdry_info, a_diel, a_incl_real, a_incl_imag, wavenumber, bdry_real, bdry_imag,
                                filter_scatterer, filter_outside, filter_inclusion, divparam2, stabil, one, one, one);

      std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_cell_cubes<< " cubes per dim on the micro grid"<< std::endl;
      hmmmaxwell.assemble();
      std::cout<< "hmm solving and corrector computation" <<std::endl;
      HMMMaxwellType::RealVectorType solreal_hmm;
      Dune::Stuff::Common::Configuration hmm_options = SolverType::options("bicgstab.diagonal");
      hmm_options.set("max_iter", "350000", true);
      hmm_options.set("precision", "1e-6", true);
      DiscreteFct macro_sol(hmmmaxwell.space(), solreal_hmm);
      std::vector< DiscreteFct > macro_solution(2, macro_sol);
      std::tuple< CurlCorrectorType, IdCorrectorType, InclCorrectorType > correctors(hmmmaxwell.solve_and_correct(macro_solution, hmm_options));
 
      ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
      ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);

      //errors to homogenized solution
      std::cout<< "computing errors to homogenized reference solution" <<std::endl;
      RefDifferenceFct hom_reference_error_real(sol_hom_ref_func[0], prolonged_ref_real);
      RefDifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], prolonged_ref_imag);
      Dune::GDT::Products::L2< LeafGridView > l2_product_operator_hom(ref_leafView);
      Dune::GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_hom(ref_leafView);
      double l2_hom = std::sqrt(l2_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                          + l2_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag));
      double hcurl_semi_hom = std::sqrt(hcurl_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                              + hcurl_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag));

      //Helmholtz decomposition of the error
      double l2_helmh_hom;
      {
        std::cout<< "computing Helmholtz decomposition" <<std::endl;
        HelmholtzDecomp< LeafGridView, 1 > decomp_hom(ref_leafView, curl_bdry_info, hom_reference_error_real, hom_reference_error_imag, one);
        Dune::Stuff::LA::Container< complextype >::VectorType sol_helmholtz_hom;
        decomp_hom.solve(sol_helmholtz_hom);
        auto decomp_hom_vec_real = decomp_hom.create_vector();
        auto decomp_hom_vec_imag = decomp_hom.create_vector();
        decomp_hom_vec_real.backend() = sol_helmholtz_hom.backend().real();
        decomp_hom_vec_imag.backend() = sol_helmholtz_hom.backend().imag();

        //compute error of Helmholtz decomposition
        typedef HelmholtzDecomp< LeafGridView, 1 >::DiscreteFctType DiscreteFctHelmh;
        DiscreteFctHelmh phi_hom_real(decomp_hom.space(), decomp_hom_vec_real);
        DiscreteFctHelmh phi_hom_imag(decomp_hom.space(), decomp_hom_vec_imag);
        std::cout<< "error of Helmholtz decomposition "<< std::endl;
        l2_helmh_hom = std::sqrt(l2_product_operator_hom.apply2(phi_hom_real, phi_hom_real) + l2_product_operator_hom.apply2(phi_hom_imag, phi_hom_imag));
      }

      output<< num_macro_cubes<< "\t" << l2_hom << "\t" << hcurl_semi_hom << "\t" << l2_helmh_hom << "\n";

    }//end for loop num_macro_cubes
  }//end try block
*/

  //================================================================================================
  //Example 3: Comparison of transmission and band gap wave number
  //           Visualization of HMM- and zeroth order approximation on a reference mesh
  //     NOTE: This can be used/enlarged to a comparison to a heterogeneous reference solution
  //================================================================================================

  double delta = 1.0/8.0;

  //select wavenumber
  double wavenumber;
  //wavenumber = 9.0; //frequency band gap
  wavenumber = 12.0; //transmission
  //wavenumber = 19.0;
  std::cout<< "wavenumber " << wavenumber << std::endl;

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;

  //==================================================================================================
  // Test Case: -Quadratic scatterer with plane wave in x_0 direction, polarized in x_1 direction
  //==================================================================================================

  //filters for in- and outside scatterer
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [size_scatterer, cube_center](const LeafGridView& cell_grid_view, const EntityType& entity) -> bool
            {const auto xx = entity.geometry().center();
             return (std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [size_scatterer, cube_center](const LeafGridView& cell_grid_view, const EntityType& entity) -> bool
            {const auto xx = entity.geometry().center();
             return !(std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer);};
  //------------------------------------------------------------------------------------------------------------------------------

  //plane wave in x_0 direction, polarized in x_1 direction
  //-----------------------------------------------------------

  //bdry condition
/*  const VectorLambdaFct bdry_real([wavenumber, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[2] - right_outer) < 1e-12 || std::abs(x[2]-left_outer) < 1e-12)
                                                                    ret[1] = -1*wavenumber*std::sin(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    ret[0] = -1*wavenumber*std::sin(wavenumber*x[0]);
           							  if (std::abs(x[1] - right_outer) < 1e-12)
								    ret[0] = wavenumber*std::sin(wavenumber*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[1] = -2*wavenumber*std::sin(wavenumber*x[0]);
                                                                  return ret;}, 0);
  const VectorLambdaFct bdry_imag([wavenumber, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[2] - right_outer) < 1e-12 || std::abs(x[2]-left_outer) < 1e-12)
                                                                    ret[1] = -1*wavenumber*std::cos(wavenumber*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    ret[0] = -1*wavenumber*std::cos(wavenumber*x[0]);
           							  if (std::abs(x[1] - right_outer) < 1e-12)
								    ret[0] = wavenumber*std::cos(wavenumber*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[1] = -2*wavenumber*std::cos(wavenumber*x[0]);
                                                                  return ret;}, 0); */

//in x_0 direction, polarized in x_2 direction
  const VectorLambdaFct bdry_real([wavenumber, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[1] - right_outer) < 1e-12 || std::abs(x[1]-left_outer) < 1e-12)
                                                                    ret[2] = -1*wavenumber*std::sin(wavenumber*x[0]);
                                                                  if (std::abs(x[2] - left_outer) < 1e-12)
                                                                    ret[0] = wavenumber*std::sin(wavenumber*x[0]);
           							  if (std::abs(x[2] - right_outer) < 1e-12)
								    ret[0] = -1*wavenumber*std::sin(wavenumber*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[2] = -2*wavenumber*std::sin(wavenumber*x[0]);
                                                                  return ret;}, 0);
  const VectorLambdaFct bdry_imag([wavenumber, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[1] - right_outer) < 1e-12 || std::abs(x[1]-left_outer) < 1e-12)
                                                                    ret[2] = -1*wavenumber*std::cos(wavenumber*x[0]);
                                                                  if (std::abs(x[2] - left_outer) < 1e-12)
                                                                    ret[0] = wavenumber*std::cos(wavenumber*x[0]);
           							  if (std::abs(x[2] - right_outer) < 1e-12)
								    ret[0] = -1*wavenumber*std::cos(wavenumber*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[2] = -2*wavenumber*std::cos(wavenumber*x[0]);
                                                                  return ret;}, 0);
  //---------------------------------------------------------------------------------------------------------------------------------------------------

  //instantiate reference grid

  try{
//filter returning true when NOT in the inclusion
  const std::function< bool(const PeriodicViewType& , const PeriodicEntityType& ) > filter_inclusion1
          = [size_inclusion, cube_center](const PeriodicViewType& cell_grid_view, const PeriodicEntityType& periodic_entity) -> bool
            {const auto xx = periodic_entity.geometry().center();
             return !((std::abs(xx[1]-cube_center[1]) <= 0.5*size_inclusion));};  //&& (std::abs(xx[1]-cube_center[1]) <= 0.5*size_inclusion) 

/* const std::function< bool(const PeriodicViewType& , const PeriodicEntityType& ) > filter_inclusion1
          = [size_inclusion, cube_center](const PeriodicViewType& cell_grid_view, const PeriodicEntityType& periodic_entity) -> bool
            {const auto xx = periodic_entity.geometry().center();
             return !((std::abs(xx[2]-cube_center[2]) <= 0.5*size_inclusion)&& (std::abs(xx[1]-cube_center[1]) <= 0.5*size_inclusion));}; */


    unsigned int num_ref_cubes = 64;
    Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
    auto ref_leafView = grid_provider.grid().leafGridView(); 

/*  {
    //maybe also a homogenized reference solution and the homogenization error are computed (see above)
    unsigned int num_ref_cubes = 48;
    Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
    auto ref_leafView = grid_provider.grid().leafGridView();

    //ConstantFct divparam(div_param);

    //homogenized reference solution
    Dune::FieldMatrix< double, 3, 3 > a_eff;
    Dune::FieldMatrix< double, 3, 3 > mu_eff_out;
    std::vector< Dune::FieldMatrix< double, 3, 3 > > mu_eff_in(2, mu_eff_out);
*/
   /* a_eff[0][0] = 0.5;
    mu_eff_out[1][1] = 0.5;
    mu_eff_out[2][2] = 0.5;
    //mu_eff_in[0][1][1] = -0.271697;
    //mu_eff_in[0][2][2] = -0.271697;
    mu_eff_in[1][1][1] = 0.0291414;
    mu_eff_in[1][2][2] = 0.0291414;*/

    //computation of effective parameters
/*    {
      GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 > coarse_space(ref_leafView);
      unsigned int num_ref_cell_cubes = 24;
      //outside inclusion
      {
        Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
        auto& cell_grid_ref = cell_grid_provider_ref.grid();
        ConstantFct k_squared(wavenumber*wavenumber);
        HMMMaxwellType::IdCellProblem id_cell(coarse_space, cell_grid_ref, k_squared, stabil, filter_inclusion1);
        HMMMaxwellType::CurlCellProblem curl_cell(coarse_space, cell_grid_ref, a_diel, divparam2, stabil, filter_inclusion1);
        std::cout<< "computing effective inverse permittivity " <<std::endl;
        a_eff = curl_cell.effective_matrix();
	std::cout<< a_eff << std::endl;
        std::cout << "computing effective permeability outside inclusions" <<std::endl;
        mu_eff_out = id_cell.effective_matrix();
	std::cout<< mu_eff_out<< std::endl;
      }//outside inclusion 
      //inside inclusion
      {
	//!!!!!!!!!!!!!!!!!!!!
	FieldVector< double, 3 > d_lower_left(0.0);
	d_lower_left[1] = d_left;
	FieldVector< double, 3 > d_upper_right(1.0);
	d_upper_right[1] = d_right;
	std::vector< unsigned int > num_ref_incl_cubes(3, num_ref_cell_cubes);
	num_ref_incl_cubes[1] = num_ref_cell_cubes/2;
        Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_lower_left, d_upper_right, num_ref_incl_cubes);
        auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();
        ConstantFct k_squared_neg(-1*wavenumber*wavenumber);
        HMMMaxwellType::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg, curl_bdry_info);
        std::cout<< "computing effective permeability inside inclusion" << std::endl;
        mu_eff_in = incl_cell.effective_matrix();
	std::cout<< mu_eff_in <<std::endl;
      }//inside inclusion 
    } //effective parameters 

    //build piece-wise constant functions
    const MatrixLambdaFct a_eff_fct([a_eff, cube_center, size_scatterer](LambdaFct::DomainType xx)
                                      {if (std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer)
                                         return a_eff;
                                       else return Stuff::Functions::internal::unit_matrix< double, 3 >();}, 0);
    const MatrixLambdaFct mu_eff_real_fct([mu_eff_out, mu_eff_in, cube_center, size_scatterer](LambdaFct::DomainType xx)
                                          {if (std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer){
                                             auto ret = mu_eff_out;
                                             ret +=mu_eff_in[0];
                                             return ret;
                                           }
                                           else return Stuff::Functions::internal::unit_matrix< double, 3 >();}, 0);
    const MatrixLambdaFct mu_eff_imag_fct([mu_eff_in, cube_center, size_scatterer](LambdaFct::DomainType xx)
                                          {if (std::abs(xx[0]-cube_center[0]) <= 0.5*size_scatterer)
                                             return mu_eff_in[1];
                                           else return Dune::FieldMatrix< double, 3, 3>();}, 0);

    //assemble and solve homogenized system
    Dune::Stuff::LA::Container< complextype >::VectorType sol_hom;
    {
      DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hom_bdry_info;
      ScatteringDiscretization< LeafGridView, 1, true, true > homdiscr(ref_leafView, hom_bdry_info, a_eff_fct, zero_matrix, wavenumber, mu_eff_real_fct, mu_eff_imag_fct, bdry_real, bdry_imag);
      std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
      std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
      homdiscr.assemble();
      std::cout<<"solving with bicgstab.diagonal"<<std::endl;
      Dune::Stuff::Common::Configuration hom_options = SolverType::options("bicgstab.diagonal");
      hom_options.set("max_iter", "200000", true);
      hom_options.set("precision", "1e-6", true);
      homdiscr.solve(sol_hom, hom_options);
    }
    //make discrete function
    Stuff::LA::Container< double >::VectorType solreal_hom(sol_hom.size());
    std::vector< DiscreteFct > sol_hom_ref_func(2, DiscreteFct(SpaceType(ref_leafView), solreal_hom));
    sol_hom_ref_func[0].vector().backend() = sol_hom.backend().real();
    sol_hom_ref_func[1].vector().backend() = sol_hom.backend().imag();

    //visualization
    {
      auto adapter = std::make_shared<Dune::Stuff::Functions::VisualizationAdapter<LeafGridView, 3, 1>>(sol_hom_ref_func[0]);
      std::unique_ptr<VTKWriter<LeafGridView>> vtk_writer = DSC::make_unique<VTKWriter<LeafGridView>>(ref_leafView, VTK::conforming);
      vtk_writer->addVertexData(adapter);
      vtk_writer->write("hom_ref_sol_k"+std::to_string((int(wavenumber)))+"_"+std::to_string(num_ref_cubes)+"_real", VTK::appendedraw); 
    }
}//end bracket homogenized reference mesh
*/

/*    { unsigned int num_ref_cubes = 64;
    Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
    auto ref_leafView = grid_provider.grid().leafGridView();

    double intpart;

    //!!!!! modify d_left, d_right
    //d_left = 0.375;
    //d_right = 0.625;

    //here one could compute a (heterogeneous) reference solution with the following delta dependent parameters
    const LambdaFct a_real([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                              {if (x[0] >= left_inner && x[0] <= right_inner) { //inside scatterer
                                 if (std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right) //
                                   return delta*delta*1.0;   //a_incl_real
                                 return 1.0;   //a_diel
                                 }
                               return 1.0;}, 0);  
    const LambdaFct a_imag([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                              {if (x[0] >= left_inner && x[0] <= right_inner) {
                                 if (std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right) //
                                   return -0.01*delta*delta;  //a_incl_imag
                                 return 0.0;
                                 }
                               return 0.0;}, 0); 
*/
/*    const LambdaFct a_real([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                              {if (x[0] >= left_inner && x[0] <= right_inner) { //inside scatterer
                                 if (!(std::modf(x[0]/delta, &intpart) >= d_left && std::modf(x[0]/delta, &intpart) <= d_right
                                       && std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right))
                                   return delta*delta*1.0;   //a_incl_real
                                 return 1.0;   //a_diel
                                 }
                               return 1.0;}, 0);  
    const LambdaFct a_imag([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                              {if (x[0] >= left_inner && x[0] <= right_inner) {
                                 if (!(std::modf(x[0]/delta, &intpart) >= d_left && std::modf(x[0]/delta, &intpart) <= d_right
                                       && std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right))
                                   return -0.01*delta*delta;  //a_incl_imag
                                 return 0.0;
                                 }
                               return 0.0;}, 0);
    a_real.visualize(ref_leafView, "parameter_real", false);
*/
/*
    //reference solution
    Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
    {
    DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > ref_bdry_info;
    ScatteringDiscretization< LeafGridView, 1, false, false > refdiscr(ref_leafView, ref_bdry_info, a_real, a_imag, wavenumber, one, zero, bdry_real, bdry_imag);
    //std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
    std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< refdiscr.space().mapper().size() <<std::endl;
    refdiscr.assemble();
    std::cout<<"solving "<<std::endl;
    typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, false, false >::MatrixTypeComplex > SolverType;
    Dune::Stuff::Common::Configuration ref_options = SolverType::options("bicgstab.diagonal");
    ref_options.set("max_iter", "500000", true);
    ref_options.set("precision", "1e-4", true);
    ref_options.set("post_check_solves_system", "0", true);
    refdiscr.solve(sol_ref, ref_options);
    }
    //make discrete function
    Stuff::LA::Container< double >::VectorType solreal_ref(sol_ref.size());
    std::vector< ScatteringDiscretization< LeafGridView, 1, false, false>::DiscreteFunctionType > 
                                             sol_ref_func(2, ScatteringDiscretization< LeafGridView, 1, false, false >::DiscreteFunctionType
                                                                   (GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 >(ref_leafView), solreal_ref));
    sol_ref_func[0].vector().backend() = sol_ref.backend().real();
    sol_ref_func[1].vector().backend() = sol_ref.backend().imag();
  
    auto adapter = std::make_shared<Dune::Stuff::Functions::VisualizationAdapter<LeafGridView, 3, 1>>(sol_ref_func[0]);
    std::unique_ptr<VTKWriter<LeafGridView>> vtk_writer = DSC::make_unique<VTKWriter<LeafGridView>>(ref_leafView, VTK::conforming);
    vtk_writer->addVertexData(adapter);
    vtk_writer->write("reference_solution_k"+std::to_string(int(wavenumber))+"_"+std::to_string(num_ref_cubes), VTK::appendedraw); 
  }//end bracket for reference mesh
*/
    //HMM
    for (unsigned int num_macro_cubes : {16}) {//if this test is used for comparison with the reference solution, a series of meshes should be studied
      //grids
      Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(left_outer, right_outer, num_macro_cubes);
      auto macro_leafView = macro_grid_provider.grid().leafGridView();
      unsigned int num_cell_cubes = num_macro_cubes;
      Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_cell_cubes);
      auto& cell_grid = cell_grid_provider.grid();
      //grid for the inclusions
      //Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_left, d_right, num_cell_cubes/2);
      //auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();
	FieldVector< double, 3 > d_lower_left(0.0);
	d_lower_left[1] = d_left;
	FieldVector< double, 3 > d_upper_right(1.0);
	d_upper_right[1] = d_right;
	std::vector< unsigned int > num_incl_cubes(3, num_cell_cubes);
	num_incl_cubes[1] = num_cell_cubes/2;
        Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_lower_left, d_upper_right, num_incl_cubes);
        auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();

      DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hmmbdry_info;
      HMMMaxwellType hmmmaxwell(macro_leafView, cell_grid, inclusion_leafView, hmmbdry_info, curl_bdry_info, a_diel, a_incl_real, a_incl_imag, wavenumber, bdry_real, bdry_imag,
                                filter_scatterer, filter_outside, filter_inclusion1, divparam2, stabil, one, one, one);

      std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_cell_cubes<< " cubes per dim on the micro grid"<< std::endl;
      hmmmaxwell.assemble();
      std::cout<< "hmm solving and corrector computation" <<std::endl;
      HMMMaxwellType::RealVectorType solreal_hmm;
      Dune::Stuff::Common::Configuration hmm_options = SolverType::options("bicgstab.diagonal");
      hmm_options.set("max_iter", "350000", true);
      hmm_options.set("precision", "1e-6", true);
      DiscreteFct macro_sol(hmmmaxwell.space(), solreal_hmm);
      std::vector< DiscreteFct > macro_solution(2, macro_sol);
      std::tuple< CurlCorrectorType, IdCorrectorType, InclCorrectorType > correctors(hmmmaxwell.solve_and_correct(macro_solution, hmm_options));

      //if a reference solution is computed, error computations should follow here

      //visualizations
      if(num_macro_cubes == 16) {
        std::cout<< "visualization" <<std::endl;
        //hmm solution
        auto adapter1 = std::make_shared<Dune::Stuff::Functions::VisualizationAdapter<LeafGridView, 3, 1>>(macro_solution[0]);
        std::unique_ptr<VTKWriter<LeafGridView>> vtk_writer1 = DSC::make_unique<VTKWriter<LeafGridView>>(macro_leafView, VTK::conforming);
        vtk_writer1->addVertexData(adapter1);
        vtk_writer1->write("hmm_solution_k"+std::to_string(int(wavenumber))+"_"+std::to_string(num_macro_cubes)+"_real_conform", VTK::appendedraw);
        //corrector
        typedef Dune::GDT::DeltaCorrectorMaxwell< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::CurlCellDiscreteFctType, HMMMaxwellType::IdCellDiscreteFctType,
                                                  HMMMaxwellType::InclusionCellDiscreteFctType > DeltaCorrectorType;
        DeltaCorrectorType corrector_real(std::get<0>(correctors).macro_function(), std::get<0>(correctors).cell_solutions(), std::get<1>(correctors).cell_solutions(),
                                          std::get<2>(correctors).cell_solutions(), filter_scatterer, filter_inclusion1, wavenumber, delta, "real");
        auto adapter2 = std::make_shared<Dune::Stuff::Functions::VisualizationAdapter<LeafGridView, 3, 1>>(corrector_real);
        std::unique_ptr<VTKWriter<LeafGridView>> vtk_writer2 = DSC::make_unique<VTKWriter<LeafGridView>>(ref_leafView, VTK::conforming);
        vtk_writer2->addVertexData(adapter2);
        vtk_writer2->write("delta_corrector_k"+std::to_string(int(wavenumber))+"_"+std::to_string(num_macro_cubes)+"_conform", VTK::appendedraw);
      }

    }//end for loop num_macro_cubes 
  }//end try block

//----------------------------------------------------------------------------------
//old stuff
/*
  //delta dependent parameters
  double intpart;
  const LambdaFct a_real([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                            {if (x[0] >= left_inner && x[0] <= right_inner && x[1] >= left_inner && x[1] <= right_inner && x[2] >= left_inner && x[2] <= right_inner) { //inside scatterer
                               if (std::modf(x[0]/delta, &intpart) >= d_left && std::modf(x[0]/delta, &intpart) <= d_right
                                       && std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right
                                       && std::modf(x[2]/delta, &intpart) >= d_left && std::modf(x[2]/delta, &intpart) <= d_right)
                                 return delta*delta*1.0;   //a_incl_real
                               return 1.0;   //a_diel
                               }
                             return 1.0;}, 0);  
  const LambdaFct a_imag([left_inner, right_inner, delta, &intpart, d_left, d_right](LambdaFct::DomainType x)
                            {if (x[0] >= left_inner && x[0] <= right_inner && x[1] >= left_inner && x[1] <= right_inner && x[2] >= left_inner && x[2] <= right_inner) {
                               if (std::modf(x[0]/delta, &intpart) >= d_left && std::modf(x[0]/delta, &intpart) <= d_right
                                       && std::modf(x[1]/delta, &intpart) >= d_left && std::modf(x[1]/delta, &intpart) <= d_right
                                       && std::modf(x[2]/delta, &intpart) >= d_left && std::modf(x[2]/delta, &intpart) <= d_right)
                                 return -0.01*delta*delta;  //a_incl_imag
                               return 0.0;
                               }
                             return 0.0;}, 0);
*/
/*  try {

  for (const double frequency : {120.0} ){
  std::cout<< "wavenumber "<< frequency <<std::endl;

  //instantiate  reference grid
  for (unsigned int num_ref_cubes : {2} ){
  std::cout<< "number of reference cubes " << num_ref_cubes <<std::endl;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);

  auto& ref_grid = grid_provider.grid();
  for (size_t ii = 0; ii<0 ;++ii){
    auto help_leafView = ref_grid.leafGridView();
    for (const auto& entity : Dune::Stuff::Common::entityRange(help_leafView) ) {
      int mark = 1;*/
  /*    //loop over vertices
      for (auto vv : Stuff::Common::valueRange(entity.template count<3>()) ) {
        const auto vertex_ptr = entity.template subEntity< 3 >(boost::numeric_cast<int>(vv));
        const auto& vertex = *vertex_ptr;
        const auto vertex_position = vertex.geometry().center();
        if (distance_to_cube(vertex_position, 0.001, left_inner, right_inner) < 1)
          mark = 1;
        //if inside scatterer
        if (filter_scatterer(help_leafView, entity)) {
          FieldVector< double, 3> vertex_yy_position(vertex_position);
          vertex_yy_position /= delta;
          double intpart1;
          for (size_t ii = 0; ii < 3; ++ii) {
            auto fracpart = std::modf(vertex_yy_position[ii], &intpart1);
            if (fracpart < 0)
              vertex_yy_position[ii] = 1 + fracpart;
            else
              vertex_yy_position[ii] = fracpart;
          }
          if (distance_to_cube(vertex_yy_position, 0.001, d_left, d_right) < 1)
            mark = 1;
        } //if inside scatterer 
      } //loop over vertices */
/*      if (mark > 0)
        ref_grid.mark(mark, entity);
    }
    ref_grid.preAdapt();
    ref_grid.adapt();
    ref_grid.postAdapt();
  }

  auto ref_leafView = ref_grid.leafGridView();
*/
  //test
/*  Dune::FieldVector< double, 3> control_point;
  control_point[0] = 3.0/(16*4);
  control_point[1] = 2.0/(16*4);
  control_point[2] = 1.0/(16*4);
  SpaceType space(ref_leafView);
  Dune::Stuff::LA::Container<double, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType sys_matrix(space.mapper().size(), space.mapper().size(), space.compute_volume_pattern());
  ConstantFct fct(-1000);
  Dune::GDT::LocalOperator::Codim0Integral< Dune::GDT::LocalEvaluation::Product< ConstantFct > > integral(fct);
  Dune::GDT::LocalAssembler::Codim0Matrix< Dune::GDT::LocalOperator::Codim0Integral< Dune::GDT::LocalEvaluation::Product< ConstantFct> > > assembler(integral);
  size_t kk = 0;
  for (const auto& entity: DSC::entityRange(ref_leafView) ) {
    const auto center = entity.geometry().center();
    bool output = false;
    //std::cout<< center  <<std::endl;
    for (auto vv : Stuff::Common::valueRange(entity.template count<3>()) ) {
      const auto vertex_ptr = entity.template subEntity< 3 >(boost::numeric_cast<int>(vv));
      const auto& vertex = *vertex_ptr;
      const auto vertex_position = vertex.geometry().center();
      if (distance_to_cube(vertex_position, 0.001, left_inner, right_inner) < 1) 
        output = true;
    }
   // if (output && 5<= kk < 10) {
      SpaceType::BaseFunctionSetType base(space.base_function_set(entity));
      auto value = base.evaluate(entity.geometry().local(center));
      auto jacob = base.jacobian(entity.geometry().local(center));
      //std::cout<< value <<std::endl;
      ++kk;
      Dune::DynamicMatrix< double > ret(base.size(), base.size());
      std::vector<Dune::DynamicMatrix<double> > tmp_ret(1, ret);
      integral.apply(base, base, ret, tmp_ret);
     // std::vector< std::vector< Dune::DynamicMatrix<double > > > tmp_tmp_ret(2, tmp_ret);
      //Dune::DynamicVector<size_t> indx(base.size());
      //std::vector<Dune::DynamicVector< size_t > > tmp_indx(2, indx);
      //assembler.assembleLocal(space, space, entity, sys_matrix, tmp_tmp_ret, tmp_indx);
 //     if (output){
        if( kk>= 13 && kk<=17 ) {
        //std::cout<< sys_matrix.backend().norm() <<std::endl;
       // std::cout<< ret.frobenius_norm() <<std::endl;
       std::cout<< jacob << std::endl;
      //std::cout<< space.mapper().globalIndices(entity)<<std::endl;
    }*/
 //       ++kk;
 //   }
  /*  if ((center-control_point).two_norm() < 1e-12) {
      SpaceType::BaseFunctionSetType base(space.base_function_set(entity));
      auto value = base.evaluate(entity.geometry().local(center));
      ConstantFct fct(-10000);
      Dune::GDT::LocalOperator::Codim0Integral< Dune::GDT::LocalEvaluation::Product< ConstantFct > > integral(fct);
      Dune::DynamicMatrix< double > ret(base.size(), base.size());
      std::vector<Dune::DynamicMatrix<double> > tmp_ret(1, ret);
      integral.apply(base, base, ret, tmp_ret);
      std::cout<< ret <<std::endl;
    }*/
 // }
 // std::cout<< sys_matrix.backend().norm()<<std::endl;


  //reference solution
/*  Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
  {
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > ref_bdry_info;
  ScatteringDiscretization< LeafGridView, 1, false, false > refdiscr(ref_leafView, ref_bdry_info, a_real, a_imag, frequency, one, zero, bdry_real, bdry_imag);
  //std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< refdiscr.space().mapper().size() <<std::endl;
  refdiscr.assemble();
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, false, false >::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration ref_options = SolverType::options("bicgstab.diagonal");
  ref_options.set("max_iter", "400000", true);
  ref_options.set("precision", "1e-4", true);
  ref_options.set("post_check_solves_system", "0", true);
  refdiscr.solve(sol_ref, ref_options);
  }
  //make discrete function
  Stuff::LA::Container< double >::VectorType solreal_ref(sol_ref.size());
  std::vector< ScatteringDiscretization< LeafGridView, 1, false, false>::DiscreteFunctionType > 
                                           sol_ref_func(2, ScatteringDiscretization< LeafGridView, 1, false, false >::DiscreteFunctionType
                                                                 (GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 >(ref_leafView), solreal_ref));
  sol_ref_func[0].vector().backend() = sol_ref.backend().real();
  sol_ref_func[1].vector().backend() = sol_ref.backend().imag();

  sol_ref_func[0].visualize("reference_solution_k"+std::to_string(int(frequency))+"_"+std::to_string(num_ref_cubes));
*/

//scattering discretization
 /* DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > bdry_info;
  ScatteringDiscretization< LeafGridView, 1, false, false > discr_direct(ref_leafView, bdry_info, zero, zero, frequency, one, zero, bdry_real, bdry_imag);
  std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entitites "<< ref_leafView.size(0) << " and number of reference dofs: "<< discr_direct.space().mapper().size() <<std::endl;
  discr_direct.assemble();
  std::cout<< discr_direct.system_matrix().sup_norm() <<std::endl;*/
/*  Dune::Stuff::LA::Container< complextype >::VectorType sol_direct;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, false, false >::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
  options.set("max_iter", "50000", true);
  discr_direct.solve(sol_direct, options);
  Stuff::LA::Container< double >::VectorType solreal_direct(sol_direct.size());
  ScatteringDiscretization< LeafGridView, 1, true, true >::DiscreteFunctionType sol_direct_func
                                                                 (GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 >(ref_leafView), solreal_direct);
  sol_direct_func.vector().backend() = sol_direct.backend().real();
  auto adapter = std::make_shared<Dune::Stuff::Functions::VisualizationAdapter<LeafGridView, 3, 1>>(sol_direct_func);
    std::unique_ptr<VTKWriter<LeafGridView>> vtk_writer = DSC::make_unique<VTKWriter<LeafGridView>>(ref_leafView, VTK::conforming);
    vtk_writer->addVertexData(adapter);
    vtk_writer->write("discrete_solution", VTK::appendedraw);  
  //discr_direct.visualize(sol_direct, "discrete_solution", "discrete_solution");
*/


  //HMM
/*  for (unsigned int num_macro_cubes : {4, 8, 12}) {
  Stuff::Grid::Providers::Cube< GridType > macro_grid_provider(left_outer, right_outer, num_macro_cubes);
  auto macro_leafView = macro_grid_provider.grid().leafGridView();
  unsigned int num_cell_cubes = num_macro_cubes;   //
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, num_cell_cubes);
  auto& cell_grid = cell_grid_provider.grid();
  //grid for the inclusions
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider(d_left, d_right, num_cell_cubes/2);
  auto inclusion_leafView = inclusion_grid_provider.grid().leafGridView();

  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hmmbdry_info;

  typedef HMMMaxwellDiscretization< LeafGridView, GridType, LeafGridView, 1 > HMMMaxwellType;
  HMMMaxwellType hmmmaxwell(macro_leafView, cell_grid, inclusion_leafView, hmmbdry_info, curl_bdry_info, a_diel, a_incl_real, a_incl_imag, frequency, bdry_real, bdry_imag,
                            filter_scatterer, filter_outside, filter_inclusion, divparam2, stabil, one, one, one);

  std::cout<< "hmm assembly for " << num_macro_cubes<< " cubes per dim on macro grid and "<< num_cell_cubes<< " cubes per dim on the micro grid"<< std::endl;
  hmmmaxwell.assemble();
  std::cout<< "hmm solving and corrector computation" <<std::endl;
  HMMMaxwellType::RealVectorType solreal_hmm;
  Dune::Stuff::LA::Container< complextype >::VectorType sol_hmm;
  typedef Dune::Stuff::LA::Solver< HMMMaxwellType::MatrixType > HMMSolverType;
  Dune::Stuff::Common::Configuration hmm_options = HMMSolverType::options("bicgstab.diagonal");
  hmm_options.set("max_iter", "350000", true);
  hmm_options.set("precision", "1e-6", true);
  hmm_options.set("post_check_solves_system", "0", true);
  HMMMaxwellType::DiscreteFunctionType macro_sol(hmmmaxwell.space(), solreal_hmm);
  std::vector< HMMMaxwellType::DiscreteFunctionType > macro_solution(2, macro_sol);
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::CurlCellDiscreteFctType > CurlCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::IdCellDiscreteFctType > IdCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::InclusionCellDiscreteFctType > InclCorrectorType;
  std::tuple< CurlCorrectorType, IdCorrectorType, InclCorrectorType > correctors(hmmmaxwell.solve_and_correct(macro_solution, hmm_options));

  typedef GDT::ProlongedFunction< HMMMaxwellType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
  typedef Stuff::Functions::Difference< ScatteringDiscretization< LeafGridView, 1, true, true>::DiscreteFunctionType, ProlongedDiscrFct > RefDifferenceFct;

  ProlongedDiscrFct prolonged_ref_real(macro_solution[0], ref_leafView);
  ProlongedDiscrFct prolonged_ref_imag(macro_solution[1], ref_leafView);

  //errors to reference solution
  RefDifferenceFct reference_error_real(sol_ref_func[0], prolonged_ref_real);
  RefDifferenceFct reference_error_imag(sol_ref_func[1], prolonged_ref_imag);

  std::cout<< "macroscopic errors on reference grid" <<std::endl;
  Dune::GDT::Products::L2< LeafGridView > l2_product_operator_ref(ref_leafView);
  Dune::GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_ref(ref_leafView);
  std::cout<< "L2 error: " << std::sqrt(l2_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                        + l2_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
  std::cout<< "Hcurl seminorm "<< std::sqrt(hcurl_product_operator_ref.apply2(reference_error_real, reference_error_real)
                                            + hcurl_product_operator_ref.apply2(reference_error_imag, reference_error_imag)) <<std::endl;
  std::cout<< "error to zeroth order approximation" <<std::endl;
  std::cout<< "L2: "<< hmmmaxwell.reference_error(sol_ref_func, std::get<0>(correctors), std::get<1>(correctors), std::get<2>(correctors), delta, "l2") <<std::endl;


  }//end for loop num_macro_cubes
*/
/*  }//end for loop num_ref_cubes

  }//end for loop frequency

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
