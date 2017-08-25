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
  ConstantFct divparam1(0.001);
  ConstantFct stabil(0.0001);


  const double div_param = 0.01;
  const double radius = 0.1;
  const double exponent = 1.0;


  typedef Dune::Fem::PeriodicLeafGridPart< GridType >::GridViewType PeriodicViewType;
  typedef PeriodicViewType::template Codim< 0 >::Entity             PeriodicEntityType;

  //parameters
  const double d_right = 0.75;
  const double d_left = 0.25;
  const double left_outer = 0.0;
  const double right_outer = 1.0;
  const double left_inner = 0.25;
  const double right_inner = 0.75;
  const double delta = 1.0/8.0;
  ConstantFct a_diel(1.0);
 // ConstantFct ep_diel(1.0);
  ConstantFct a_incl_real(1.0);
  ConstantFct a_incl_imag(-0.01);

  LambdaFct divparam2([div_param, exponent, radius, d_left, d_right](LambdaFct::DomainType x) {
                      auto ret = distance_to_cube(x, radius, d_left, d_right);
                      return div_param * std::pow(ret, 2 * exponent);
                      }
                      , 1);

  //filter for OUTSIDE of inclusion
  const std::function< bool(const PeriodicViewType& , const PeriodicEntityType& ) > filter_inclusion
          = [d_left, d_right](const PeriodicViewType& /*cell_grid_view*/, const PeriodicEntityType& periodic_entity) -> bool
            {const auto xx = periodic_entity.geometry().center();
             return !(xx[0] >= d_left && xx[0] <= d_right && xx[1] >= d_left && xx[1] <= d_right && xx[2] >= d_left && xx[2] <= d_right);};
//filters for in- and outside scatterer
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_scatterer
          = [left_inner, right_inner](const LeafGridView& /*cell_grid_view*/, const EntityType& entity) -> bool
            {const auto xx = entity.geometry().center();
             return (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner && xx[2] >= left_inner && xx[2] <= right_inner);};
  const std::function< bool(const LeafGridView& , const EntityType& ) > filter_outside
          = [left_inner, right_inner](const LeafGridView& /*cell_grid_view*/, const EntityType& entity) -> bool
            {const auto xx = entity.geometry().center();
             return !(xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner && xx[2] >= left_inner && xx[2] <= right_inner);};

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

  try {

  for (const double frequency : {12.0} ){
  std::cout<< "wavenumber "<< frequency <<std::endl;
  const VectorLambdaFct bdry_real([frequency, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[2] - right_outer) < 1e-12 || std::abs(x[2]-left_outer) < 1e-12)
                                                                    ret[1] = -1*frequency*std::sin(frequency*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    ret[0] = -1*frequency*std::sin(frequency*x[0]);
           							  if (std::abs(x[1] - right_outer) < 1e-12)
								    ret[0] = frequency*std::sin(frequency*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[1] = -2*frequency*std::sin(frequency*x[0]);
                                                                  return ret;}, 0);
  const VectorLambdaFct bdry_imag([frequency, left_outer, right_outer](VectorLambdaFct::DomainType x){
                                                                  Dune::FieldVector< double, 3 > ret(0.0);
                                                                  if (std::abs(x[2] - right_outer) < 1e-12 || std::abs(x[2]-left_outer) < 1e-12)
                                                                    ret[1] = -1*frequency*std::cos(frequency*x[0]);
                                                                  if (std::abs(x[1] - left_outer) < 1e-12)
                                                                    ret[0] = -1*frequency*std::cos(frequency*x[0]);
           							  if (std::abs(x[1] - right_outer) < 1e-12)
								    ret[0] = frequency*std::cos(frequency*x[0]);
 								  if (std::abs(x[0] - right_outer) < 1e-12)
								    ret[1] = -2*frequency*std::cos(frequency*x[0]);
                                                                  return ret;}, 0);
  
  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;

  //instantiate  reference grid
  for (unsigned int num_ref_cubes : {48} ){
  std::cout<< "number of reference cubes " << num_ref_cubes <<std::endl;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(left_outer, right_outer, num_ref_cubes);
  auto ref_leafView = grid_provider.grid().leafGridView();


  //reference solution
  Dune::Stuff::LA::Container< complextype >::VectorType sol_ref;
  {
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > ref_bdry_info;
  ScatteringDiscretization< LeafGridView, 1, false, false > refdiscr(ref_leafView, ref_bdry_info, a_real, a_imag, frequency, one, zero, bdry_real, bdry_imag);
  std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< refdiscr.space().mapper().size() <<std::endl;
  refdiscr.assemble();
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  //homdiscr.solve(sol_hom);
  typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, false, false >::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration ref_options = SolverType::options("bicgstab.diagonal");
  ref_options.set("max_iter", "400000", true);
  ref_options.set("precision", "1e-4", true);
  ref_options.set("post_check_solves_system", "0", true);
  SolverType ref_solver(refdiscr.system_matrix());
  ref_solver.apply(refdiscr.rhs_vector(), sol_ref, ref_options);
  refdiscr.visualize(sol_ref, "reference_solution_k"+std::to_string((int(frequency)))+"_"+std::to_string(num_ref_cubes), "discrete_solution_"+std::to_string(int(frequency)));
  }
  //make discrete function
  Stuff::LA::Container< double >::VectorType solreal_ref(sol_ref.size());
  std::vector< ScatteringDiscretization< LeafGridView, 1, false, false>::DiscreteFunctionType > 
                                           sol_ref_func(2, ScatteringDiscretization< LeafGridView, 1, false, false >::DiscreteFunctionType
                                                                 (GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 >(ref_leafView), solreal_ref));
  sol_ref_func[0].vector().backend() = sol_ref.backend().real();
  sol_ref_func[1].vector().backend() = sol_ref.backend().imag();


  //homogenized reference solution
/*  Dune::FieldMatrix< double, 3, 3 > a_eff;
  Dune::FieldMatrix< double, 3, 3 > mu_eff_out;
  std::vector< Dune::FieldMatrix< double, 3, 3 > > mu_eff_in(2, mu_eff_out);

  //computations of effective parameters
  {
  GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 > coarse_space(ref_leafView);
  unsigned int num_ref_cell_cubes = 32;
  //outside inclusion
  {
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider_ref(0.0, 1.0, num_ref_cell_cubes);
  auto& cell_grid_ref = cell_grid_provider_ref.grid();
  ConstantFct k_squared(frequency*frequency);
  HMMMaxwellDiscretization< LeafGridView, GridType, LeafGridView, 1>::IdCellProblem id_cell(coarse_space, cell_grid_ref, k_squared, stabil, filter_inclusion);
  //normal variant with curl cell matrix
  HMMMaxwellDiscretization< LeafGridView, GridType, LeafGridView, 1>::CurlCellProblem curl_cell(coarse_space, cell_grid_ref, a_diel, divparam2, stabil, filter_inclusion);
  a_eff = curl_cell.effective_matrix();
  std::cout<< "effective inverse permittivity " <<std::endl;
  std::cout<< a_eff <<std::endl;
  //variant 2: via inverse elliptic effective matrix
  //typedef GDT::Operators::CurlEllipticCellReconstruction< GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 >, GridType > CurlEllCellReconType;
  //CurlEllCellReconType curlellipticcell(coarse_space, cell_grid_ref, ep_diel, filter_inclusion);
  //a_eff = curlellipticcell.effective_matrix();
  //std::cout<< "effective matrix via inversion of elliptic effective matrix " <<std::endl;
  //std::cout<< a_eff <<std::endl;
  mu_eff_out = id_cell.effective_matrix();
  std::cout<< "effective permeability outside the inclusion " << mu_eff_out <<std::endl; 
  }//outside inclusion 
  //inside inclusion
  {
  Stuff::Grid::Providers::Cube< GridType > inclusion_grid_provider_ref(d_left, d_right, num_ref_cell_cubes/2);
  auto inclusion_ref_leafView = inclusion_grid_provider_ref.grid().leafGridView();
  ConstantFct k_squared_neg(-1*frequency*frequency);
  HMMMaxwellDiscretization< LeafGridView, GridType, LeafGridView, 1>::InclusionCellProblem incl_cell(inclusion_ref_leafView, a_incl_real, a_incl_imag, k_squared_neg, curl_bdry_info);
  mu_eff_in = incl_cell.effective_matrix();
  std::cout<< "effective permeability inside the inclusion, real part " << mu_eff_in[0] <<std::endl;
  std::cout<< "effective permeability inside the inclusion, imag part " << mu_eff_in[1] <<std::endl;
  }//inside inclusion 
  } //effective parameters
*/
  //manual filling of the effective matrices
/*  for (size_t ii = 0; ii < 3; ++ii){
    for (size_t jj = 0; jj < 3; ++jj) {
      if (ii == jj) {
        a_eff[ii][jj] = 0.333542;
        mu_eff_out[ii][jj] = 0.813064;
        mu_eff_in[0][ii][jj] = -2.65547;
        mu_eff_in[1][ii][jj] = 1.6605;
      }
      else {
        mu_eff_in[0][ii][jj] = 0.0882852;
        mu_eff_in[1][ii][jj] = -0.0779266;
      }
    }
  }
*/
  //build piece-wise constant functions
/*  const MatrixLambdaFct a_eff_fct([a_eff, left_inner, right_inner](LambdaFct::DomainType xx)
                                           {if (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner && xx[2]>= left_inner && xx[2]<= right_inner)
                                              return a_eff;
                                            else return Stuff::Functions::internal::unit_matrix< double, 3 >();}, 0);
  const MatrixLambdaFct mu_eff_real_fct([mu_eff_out, mu_eff_in, left_inner, right_inner](LambdaFct::DomainType xx)
                                        {if (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner && xx[2]>= left_inner && xx[2]<= right_inner){
                                           auto ret = mu_eff_out;
                                           ret +=mu_eff_in[0];
                                           return ret;
                                         }
                                         else return Stuff::Functions::internal::unit_matrix< double, 3 >();}, 0);
  const MatrixLambdaFct mu_eff_imag_fct([mu_eff_in, left_inner, right_inner](LambdaFct::DomainType xx)
                                       {if (xx[0] >= left_inner && xx[0] <= right_inner && xx[1] >= left_inner && xx[1] <= right_inner && xx[2]>= left_inner && xx[2]<= right_inner)
                                          return mu_eff_in[1];
                                        else return Dune::FieldMatrix< double, 3, 3>();}, 0);

  //assemble and solve homogenized system
  Dune::Stuff::LA::Container< complextype >::VectorType sol_hom;
  {
  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > hom_bdry_info;
  ScatteringDiscretization< LeafGridView, 1, true, true > homdiscr(ref_leafView, hom_bdry_info, a_eff_fct, zero_matrix, frequency, mu_eff_real_fct, mu_eff_imag_fct, bdry_real, bdry_imag);
  std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entities "<< ref_leafView.size(0) << " and number of reference dofs: "<< homdiscr.space().mapper().size() <<std::endl;
  homdiscr.assemble();
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  //homdiscr.solve(sol_hom);
  typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, true, true >::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration hom_options = SolverType::options("bicgstab.diagonal");
  hom_options.set("max_iter", "200000", true);
  hom_options.set("precision", "1e-6", true);
  SolverType hom_solver(homdiscr.system_matrix());
  hom_solver.apply(homdiscr.rhs_vector(), sol_hom, hom_options);
  homdiscr.visualize(sol_hom, "homogenized_solution_k"+std::to_string((int(frequency)))+"_"+std::to_string(num_ref_cubes), "discrete_solution_"+std::to_string(int(frequency)));
  }
  //make discrete function
  Stuff::LA::Container< double >::VectorType solreal_hom(sol_hom.size());
  std::vector< ScatteringDiscretization< LeafGridView, 1, true, true>::DiscreteFunctionType > 
                                           sol_hom_ref_func(2, ScatteringDiscretization< LeafGridView, 1, true, true >::DiscreteFunctionType
                                                                 (GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3, 1 >(ref_leafView), solreal_hom));
  sol_hom_ref_func[0].vector().backend() = sol_hom.backend().real();
  sol_hom_ref_func[1].vector().backend() = sol_hom.backend().imag();
*/
  //variant 1: with filter in CurlCellReconstruction
 // typedef GDT::Operators::CurlCellReconstruction< CoarseSpaceType, GridType > CurlCellReconType;
  //CurlCellReconType curlcell(coarse_space, cell_grid_ref, a_diel, divparam2, stabil, filter_inclusion);
  //auto a_eff1 = curlcell.effective_matrix();
  //std::cout<< "effective matrix with filter in CurlCellReconstruction " <<std::endl;
  //std::cout<< a_eff1 <<std::endl;

  //variant 2: via inverse elliptic effective matrix
  //typedef GDT::Operators::CurlEllipticCellReconstruction< CoarseSpaceType, GridType > CurlEllCellReconType;
  //CurlEllCellReconType curlellipticcell(coarse_space, cell_grid_ref, ep_diel, filter_inclusion);
  //auto a_eff2 = curlellipticcell.effective_matrix();
  //std::cout<< "effective matrix via inversion of elliptic effective matrix " <<std::endl;
  //std::cout<< a_eff2 <<std::endl;

 // typedef GDT::MaxwellInclusionCell< LeafGridView > InclusionCellType;
  //InclusionCellType maxwellinclusioncell(inclusion_ref_leafView, a_incl_real, a_incl_imag, om_squared_neg);
 // InclusionCellType::CellSolutionStorageType cell_solutions(3);
 // for (auto& it : cell_solutions) {
 //   std::vector<GDT::DiscreteFunction< InclusionCellType::SpaceType, InclusionCellType::RealVectorType > > 
 //            it1(2, GDT::DiscreteFunction< InclusionCellType::SpaceType, InclusionCellType::RealVectorType >(maxwellinclusioncell.cell_space()));
 //   it = DSC::make_unique< InclusionCellType::CellDiscreteFunctionType >(it1);
 // }
 // maxwellinclusioncell.compute_cell_solutions(cell_solutions);
  //cell_solutions[0]->operator[](0).visualize("cell_solution_x_real");
  //cell_solutions[2]->operator[](0).visualize("cell_solution_z_real");
  //auto effective_matrix_incl = maxwellinclusioncell.effective_matrix();
  //std::cout<< "effective matrix inside the inclusion, real part "<<std::endl;
  //std::cout<< effective_matrix_incl[0]<<std::endl;
  //std::cout<< "effective matrix inside the inclusion, imag part "<<std::endl;
  //std::cout<< effective_matrix_incl[1]<<std::endl;
 // if (frequency == 7.0 ){
 // GDT::Operators::IdCellReconstruction< CoarseSpaceType, GridType > idcell(coarse_space, cell_grid_ref, om_squared, stabil, filter_inclusion);
 // std::cout<< "effective matrix outside the inclusions "<<std::endl;
 // std::cout<< idcell.effective_matrix() <<std::endl;
 // }

/*  DSG::BoundaryInfos::AllNeumann< LeafGridView::Intersection > bdry_info;
  ScatteringDiscretization< LeafGridView, 1, false, false > discr_direct(ref_leafView, bdry_info, one, zero, frequency, one, zero, bdry_real, bdry_imag);
  std::cout<< "assembling on grid with "<< num_ref_cubes<< " cubes per direction"<<std::endl;
  std::cout<< "number of reference entitites "<< ref_leafView.size(0) << " and number of reference dofs: "<< discr_direct.space().mapper().size() <<std::endl;
  discr_direct.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol_direct;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  //discr_direct.solve(sol_direct);
  typedef Dune::Stuff::LA::Solver< ScatteringDiscretization< LeafGridView, 1, false, false >::MatrixTypeComplex > SolverType;
  Dune::Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
  options.set("max_iter", "50000", true);
  SolverType solver(discr_direct.system_matrix());
  solver.apply(discr_direct.rhs_vector(), sol_direct, options);
  discr_direct.visualize(sol_direct, "discrete_solution", "discrete_solution");
*/


  //HMM
  for (unsigned int num_macro_cubes : {8, 12, 16}) {
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
  std::cout<< "hmm solving" <<std::endl;
  Dune::Stuff::LA::Container< complextype >::VectorType sol_hmm;
  //hmmmaxwell.solve(sol_hmm);
  typedef Dune::Stuff::LA::Solver< HMMMaxwellType::MatrixType > HMMSolverType;
  Dune::Stuff::Common::Configuration hmm_options = HMMSolverType::options("bicgstab.diagonal");
  hmm_options.set("max_iter", "350000", true);
  hmm_options.set("precision", "1e-6", true);
  hmm_options.set("post_check_solves_system", "0", true);
  HMMSolverType hmm_solver(hmmmaxwell.system_matrix());
  hmm_solver.apply(hmmmaxwell.rhs_vector(), sol_hmm, hmm_options);
  HMMMaxwellType::RealVectorType solreal_hmm(sol_hmm.size());
  solreal_hmm.backend() = sol_hmm.backend().real();

  if(num_macro_cubes == 16) {
    HMMMaxwellType::DiscreteFunctionType sol_real_func(hmmmaxwell.space(), solreal_hmm, "solution_real_part");
    sol_real_func.visualize("hmm_solution_k"+std::to_string((int(frequency)))+"_"+std::to_string(num_macro_cubes)+"_real");
  }

  typedef GDT::ProlongedFunction< HMMMaxwellType::DiscreteFunctionType, LeafGridView >        ProlongedDiscrFct;
  typedef Stuff::Functions::Difference< ScatteringDiscretization< LeafGridView, 1, true, true>::DiscreteFunctionType, ProlongedDiscrFct > RefDifferenceFct;

  std::cout<< "corrector computation" <<std::endl;
  HMMMaxwellType::DiscreteFunctionType macro_sol(hmmmaxwell.space(), solreal_hmm);
  std::vector< HMMMaxwellType::DiscreteFunctionType > macro_solution(2, macro_sol);
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::CurlCellDiscreteFctType > CurlCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::IdCellDiscreteFctType > IdCorrectorType;
  typedef Dune::GDT::PeriodicCorrector< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::InclusionCellDiscreteFctType > InclCorrectorType;
  std::tuple< CurlCorrectorType, IdCorrectorType, InclCorrectorType > correctors(hmmmaxwell.solve_and_correct(macro_solution));

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

  if (num_macro_cubes == 16) {
    std::cout<< "visualization" <<std::endl;
    typedef Dune::GDT::DeltaCorrectorMaxwell< HMMMaxwellType::DiscreteFunctionType, HMMMaxwellType::CurlCellDiscreteFctType, HMMMaxwellType::IdCellDiscreteFctType,
                                              HMMMaxwellType::InclusionCellDiscreteFctType > DeltaCorrectorType;
    DeltaCorrectorType corrector_real(std::get<0>(correctors).macro_function(), std::get<0>(correctors).cell_solutions(), std::get<1>(correctors).cell_solutions(),
                                      std::get<2>(correctors).cell_solutions(), filter_scatterer, filter_inclusion, frequency, delta, "real");
    corrector_real.visualize(ref_leafView, "delta_corrector_k"+std::to_string((int(frequency)))+"_"+std::to_string(num_macro_cubes), false);
  }

  //errors to homogenized solution
/*  std::cout<< "computing errors to homogenized reference solution" <<std::endl;
  RefDifferenceFct hom_reference_error_real(sol_hom_ref_func[0], prolonged_ref_real);
  RefDifferenceFct hom_reference_error_imag(sol_hom_ref_func[1], prolonged_ref_imag);
  Dune::GDT::Products::L2< LeafGridView > l2_product_operator_hom(ref_leafView);
  Dune::GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator_hom(ref_leafView);
  std::cout<< "L2 error: " << std::sqrt(l2_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                        + l2_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag)) <<std::endl;
  std::cout<< "Hcurl seminorm "<< std::sqrt(hcurl_product_operator_hom.apply2(hom_reference_error_real, hom_reference_error_real)
                                            + hcurl_product_operator_hom.apply2(hom_reference_error_imag, hom_reference_error_imag)) <<std::endl;

  //Helmholtz decomposition of the error
  {
  std::cout<< "computing Helmholtz decomposition" <<std::endl;
  //DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > curl_bdry_info;
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
  std::cout<< "L2 error of gradient part in Helmholtz decomposition (on the reference grid): "<<
               std::sqrt(l2_product_operator_hom.apply2(phi_hom_real, phi_hom_real) + l2_product_operator_hom.apply2(phi_hom_imag, phi_hom_imag))<< std::endl;

  }
*/
  }//end for loop num_macro_cubes

  }//end for loop num_ref_cubes

  }//end for loop frequency

  }//end try block

  catch(Dune::Exception& ee) {
    std::cout<< ee.what() <<std::endl;
  }

  catch(...) {
    std::cout<< "something went wrong" <<std::endl;
  }

  return 0;
}
