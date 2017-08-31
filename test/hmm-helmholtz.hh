// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_TEST_HMM_HELMHOLTZ_HH
#define DUNE_GDT_TEST_HMM_HELMHOLTZ_HH

#if !HAVE_DUNE_PDELAB
# error "This one requires dune-pdelab!"
#endif

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>

#include <dune/common/timer.hh>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/common/ranges.hh>
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localevaluation/product.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/operators/elliptic-cg.hh>
#include <dune/gdt/operators/cellreconstruction.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

//for error computation
#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/h1.hh>
#include <dune/gdt/discretefunction/corrector.hh>

namespace Dune {
namespace GDT {


template< class GridViewImp >
class HelmholtzInclusionCell
{
public:
  typedef GridViewImp                                       PeriodicViewType;
  typedef typename GridViewImp::template Codim< 0 >::Entity EntityType;
  typedef typename GridViewImp::ctype                       DomainFieldType;
  static const size_t                                       dimDomain = GridViewImp::dimension;

  typedef std::complex< double > complextype;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType      MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType      VectorType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, double, 1 > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType >            DiscreteFunctionType;
  typedef std::vector< DiscreteFunctionType >                             CellDiscreteFunctionType;
  typedef std::vector< std::shared_ptr< CellDiscreteFunctionType > >      CellSolutionStorageType;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, double, 1 > ScalarFct;
  typedef Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, double, 1 >          ConstantFct;

  typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > >  EllipticOperator;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ConstantFct > > IdOperator;

  typedef LocalAssembler::Codim0Matrix< EllipticOperator > EllipticAssembler;
  typedef LocalAssembler::Codim0Matrix< IdOperator >       IdAssembler;


  HelmholtzInclusionCell(const GridViewImp& cell_gridview, const ScalarFct& a_real, const ScalarFct& a_imag,
                         const ConstantFct& wavenumber_squared_neg)
    : cell_space_(cell_gridview)
    , a_real_(a_real)
    , a_imag_(a_imag)
    , wavenumber_squared_neg_(wavenumber_squared_neg)
    , system_matrix_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_assembler_(cell_space_)
    , system_matrix_real_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_matrix_imag_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , elliptic_operator_real_(a_real_)
    , elliptic_operator_imag_(a_imag_)
    , id_operator_(wavenumber_squared_neg_)
    , elliptic_assembler_real_(elliptic_operator_real_)
    , elliptic_assembler_imag_(elliptic_operator_imag_)
    , id_assembler_(id_operator_)
  {
    system_assembler_.add(elliptic_assembler_real_, system_matrix_real_);
    system_assembler_.add(elliptic_assembler_imag_, system_matrix_imag_);
    system_assembler_.add(id_assembler_, system_matrix_real_);
  }

  const SpaceType& cell_space() const
  {
    return cell_space_;
  }

  void compute_cell_solutions(CellSolutionStorageType& cell_solutions) const
  {
    assert(cell_solutions.size() > 0);
    //clear return argument
    for (auto& localSol : cell_solutions) {
      assert(localSol->size() > 1);
      localSol->operator[](0).vector() *= 0;
      localSol->operator[](1).vector() *= 0;
    }
    //assemble rhs and system
    ConstantFct one(1.0);
    VectorType rhs_vector_real(cell_space_.mapper().size());
    VectorTypeComplex rhs_vector_total(cell_space_.mapper().size());
    auto rhs_functional_real = Dune::GDT::Functionals::make_l2_volume(one, rhs_vector_real, cell_space_);
    system_assembler_.add(*rhs_functional_real);
    Spaces::DirichletConstraints< typename GridViewImp::Intersection >
           dirichlet_constraints(DSG::BoundaryInfos::AllDirichlet< typename GridViewImp::Intersection >(), cell_space_.mapper().size());
    system_assembler_.add(dirichlet_constraints);
    system_assembler_.assemble();
    dirichlet_constraints.apply(system_matrix_real_, rhs_vector_real);
    dirichlet_constraints.apply(system_matrix_imag_);
    //make complex matrix and vector
    rhs_vector_total.backend() = rhs_vector_real.backend().template cast< complextype >();
    complextype im(0.0, 1.0);
    system_matrix_.backend() = system_matrix_imag_.backend().template cast< complextype >();
    system_matrix_.scal(im);
    system_matrix_.backend() += system_matrix_real_.backend().template cast< complextype >();
    //solve
    assert(cell_solutions.size() == 1);
    auto& current_solution = *cell_solutions[0];
    VectorTypeComplex tmp_solution(rhs_vector_total.size());
    if(!rhs_vector_total.valid())
      DUNE_THROW(Dune::InvalidStateException, "RHS vector invalid!");
    Stuff::LA::Solver< MatrixTypeComplex > solver(system_matrix_);
    solver.apply(rhs_vector_total, tmp_solution, "bicgstab.diagonal");
    if(!tmp_solution.valid())
      DUNE_THROW(Dune::InvalidStateException, "Solution vector invalid!");
    //make discrete functions
    assert(current_solution.size() > 1);
    current_solution[0].vector().backend() = tmp_solution.backend().real();
    current_solution[1].vector().backend() = tmp_solution.backend().imag();
  } //compute_cell_solutions

  template< class FunctionType >
  typename FunctionType::RangeType average(FunctionType& function) const
  {
    typename FunctionType::RangeType result(0.0);
    //integrate
    for (const auto& entity : DSC::entityRange(cell_space_.grid_view()) ) {
      const auto localparam = function.local_function(entity);
      const size_t int_order = localparam->order();
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(int_order));
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        auto evaluation_result = localparam->evaluate(x);
        evaluation_result *= (quadrature_weight * integration_factor);
        result += evaluation_result;
      } //loop over quadrature points
    } //loop over entities
    return result;
  } //average

  complextype effective_param() const
  {
    CellSolutionStorageType cell_solution(1);
    for (auto& it : cell_solution){
      std::vector< DiscreteFunctionType > it1(2, DiscreteFunctionType(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    compute_cell_solutions(cell_solution);
    typedef Dune::Stuff::Functions::Product< ConstantFct, DiscreteFunctionType > ProductFct;
    ProductFct real_integrand(wavenumber_squared_neg_, cell_solution[0]->operator[](0));
    ProductFct imag_integrand(wavenumber_squared_neg_, cell_solution[0]->operator[](1));
    double real_result = 1 - average(real_integrand);
    double imag_result = -1*average(imag_integrand);
    return complextype(real_result, imag_result);
  }

private:
  const SpaceType                      cell_space_;
  const ScalarFct&                     a_real_;
  const ScalarFct&                     a_imag_;
  const ConstantFct&                   wavenumber_squared_neg_;
  mutable MatrixTypeComplex            system_matrix_;
  mutable SystemAssembler< SpaceType > system_assembler_;
  mutable MatrixType                   system_matrix_real_;
  mutable MatrixType                   system_matrix_imag_;
  mutable EllipticOperator             elliptic_operator_real_;
  mutable EllipticOperator             elliptic_operator_imag_;
  mutable IdOperator                   id_operator_;
  mutable EllipticAssembler            elliptic_assembler_real_;
  mutable EllipticAssembler            elliptic_assembler_imag_;
  mutable IdAssembler                  id_assembler_;
};


} // namespace GDT
} //namespace Dune


template< class MacroGridViewType, class CellGridType, class InclusionGridViewType, int polynomialOrder >
class HMMHelmholtzDiscretization {
public:
  typedef typename MacroGridViewType::ctype                                   MacroDomainFieldType;
  typedef typename MacroGridViewType::template Codim<0>::Entity               MacroEntityType;
  typedef double                                                              RangeFieldType;
  static const size_t       dimDomain = MacroGridViewType::dimension;
  static const size_t       dimRange = 1;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename MacroGridViewType::Intersection >                            BoundaryInfoType;
  typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >                 MacroConstFct;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >        MacroScalarFct;
  typedef std::function< bool(const MacroGridViewType&, const MacroEntityType&) >                                         MacroFilterType;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 RealMatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 RealVectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;

  typedef Dune::GDT::Spaces::CG::PdelabBased< MacroGridViewType, polOrder, double, dimRange > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, RealVectorType >                            DiscreteFunctionType;

  typedef Dune::GDT::Operators::EllipticCellReconstruction< SpaceType, CellGridType >                   EllipticCellProblem;
  typedef Dune::GDT::HelmholtzInclusionCell< InclusionGridViewType >                                    InclusionCellProblem;
  typedef typename EllipticCellProblem::PeriodicEntityType                                              CellEntityType;
  typedef typename EllipticCellProblem::DomainFieldType                                                 CellDomainFieldType;
  typedef typename EllipticCellProblem::ScalarFct                                                       CellScalarFct;
  typedef typename EllipticCellProblem::FilterType                                                      CellFilterType;
  typedef Dune::Stuff::Functions::Constant< CellEntityType, CellDomainFieldType, dimDomain, double, 1 > CellConstFct;
  typedef typename InclusionCellProblem::ConstantFct                                                    InclusionConstantFct;

  typedef Dune::GDT::DiscreteFunction< typename EllipticCellProblem::CellSpaceType, RealVectorType >      EllipticCellDiscreteFctType;
  typedef Dune::GDT::DiscreteFunction< typename InclusionCellProblem::SpaceType, RealVectorType >         InclusionCellDiscreteFctType;

  typedef typename EllipticCellProblem::CellSolutionStorageType   AllEllipticSolutionsStorageType;
  typedef typename InclusionCellProblem::CellSolutionStorageType  AllInclusionSolutionsStorageType;

  HMMHelmholtzDiscretization(const MacroGridViewType& macrogridview,
                             CellGridType& cellgrid,
                             const InclusionGridViewType& inclusion_gridview,
                             const BoundaryInfoType& info,
                             const CellScalarFct& a_diel,
                             const CellScalarFct& a_incl_real,
                             const CellScalarFct& a_incl_imag,
                             const double& wavenumber,
                             const MacroScalarFct& bdry_real,
                             const MacroScalarFct& bdry_imag,
                             MacroFilterType filter_scatterer,
                             MacroFilterType filter_outside,
                             CellFilterType filter_inclusion,
                             const CellScalarFct& stabil,
                             const MacroScalarFct& a_diel_macro,
                             const MacroScalarFct& a_incl_real_macro,
                             const MacroScalarFct& a_incl_imag_macro)
    : coarse_space_(macrogridview)
    , bdry_info_(info)
    , macro_a_diel_(a_diel_macro)
    , macro_a_incl_real_(a_incl_real_macro)
    , macro_a_incl_imag_(a_incl_imag_macro)
    , bdry_real_(bdry_real)
    , bdry_imag_(bdry_imag)
    , periodic_a_diel_(a_diel)
    , periodic_a_incl_real_(a_incl_real)
    , periodic_a_incl_imag_(a_incl_imag)
    , stabil_param_(stabil)
    , wavenumber_(wavenumber)
    , k_squared_neg_(-1*wavenumber_*wavenumber_)
    , filter_scatterer_(filter_scatterer)
    , filter_outside_(filter_outside)
    , filter_inclusion_(filter_inclusion)
    , elliptic_cell_(coarse_space_, cellgrid, periodic_a_diel_, stabil_param_, filter_inclusion_)
    , inclusion_cell_(inclusion_gridview, periodic_a_incl_real_, periodic_a_incl_imag_, k_squared_neg_)
    , is_assembled_(false)
    , system_matrix_real_(0,0)
    , system_matrix_imag_(0,0)
    , system_matrix_(0,0)
    , rhs_vector_real_(0)
    , rhs_vector_imag_(0)
    , rhs_vector_(0)
  {}

  const SpaceType& space() const
  {
    return coarse_space_;
  }

  const typename EllipticCellProblem::CellSpaceType& ell_cell_space() const
  {
    return elliptic_cell_.cell_space();
  }

  const typename InclusionCellProblem::SpaceType& curl_cell_space() const
  {
    return inclusion_cell_.cell_space();
  }


  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = coarse_space_.compute_volume_pattern();
      system_matrix_real_ = RealMatrixType(coarse_space_.mapper().size(), coarse_space_.mapper().size(), sparsity_pattern);
      system_matrix_imag_ = RealMatrixType(coarse_space_.mapper().size(), coarse_space_.mapper().size(), sparsity_pattern);
      system_matrix_ = MatrixType(coarse_space_.mapper().size(), coarse_space_.mapper().size());
      rhs_vector_real_ = RealVectorType(coarse_space_.mapper().size());
      rhs_vector_imag_ = RealVectorType(coarse_space_.mapper().size());
      rhs_vector_ = VectorType(coarse_space_.mapper().size());
      SystemAssembler< SpaceType > walker(coarse_space_);

      //rhs
      auto bdry_functional_real = Functionals::make_l2_face(bdry_real_, rhs_vector_real_, coarse_space_,
                                                            new Stuff::Grid::ApplyOn::NeumannIntersections< MacroGridViewType >(bdry_info_));
      walker.add(*bdry_functional_real);
      auto bdry_functional_imag = Functionals::make_l2_face(bdry_imag_, rhs_vector_imag_, coarse_space_,
                                                            new Stuff::Grid::ApplyOn::NeumannIntersections< MacroGridViewType >(bdry_info_));
      walker.add(*bdry_functional_imag);

      //solve cell problems
      AllEllipticSolutionsStorageType elliptic_cell_solutions(dimDomain);
      for (auto& it : elliptic_cell_solutions) {
        std::vector<DiscreteFunction< typename EllipticCellProblem::CellSpaceType, RealVectorType > >
                  it1(1, DiscreteFunction< typename EllipticCellProblem::CellSpaceType, RealVectorType >(elliptic_cell_.cell_space()));
        it = DSC::make_unique< typename EllipticCellProblem::CellDiscreteFunctionType >(it1);
      }
      std::cout<< "computing elliptic cell problems"<< std::endl;
      elliptic_cell_.compute_cell_solutions(elliptic_cell_solutions);
      AllInclusionSolutionsStorageType inclusion_cell_solutions(1);
      for (auto& it : inclusion_cell_solutions) {
        std::vector<DiscreteFunction< typename InclusionCellProblem::SpaceType, RealVectorType > >
                  it1(2, DiscreteFunction< typename InclusionCellProblem::SpaceType, RealVectorType >(inclusion_cell_.cell_space()));
        it = DSC::make_unique< typename InclusionCellProblem::CellDiscreteFunctionType >(it1);
      }
      std::cout<< "computing inclusion cell problems"<< std::endl;
      inclusion_cell_.compute_cell_solutions(inclusion_cell_solutions);

      //lhs in scatterer
      typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMEllipticPeriodic< CellScalarFct, EllipticCellProblem > > HMMEllipticOperator;
      typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMHelmholtzInclusionPeriodic< CellScalarFct, InclusionCellProblem > > HMMInclusionOperator;
      HMMEllipticOperator hmmelliptic(elliptic_cell_, periodic_a_diel_, macro_a_diel_, elliptic_cell_solutions, filter_inclusion_);
      HMMInclusionOperator hmmincl_real(inclusion_cell_, periodic_a_incl_real_, periodic_a_incl_imag_, wavenumber_, true, macro_a_incl_real_, macro_a_incl_imag_, inclusion_cell_solutions);
      HMMInclusionOperator hmmincl_imag(inclusion_cell_, periodic_a_incl_real_, periodic_a_incl_imag_, wavenumber_, false, macro_a_incl_real_, macro_a_incl_imag_, inclusion_cell_solutions);
      LocalAssembler::Codim0Matrix< HMMEllipticOperator > hmm_ell_assembler(hmmelliptic);
      LocalAssembler::Codim0Matrix< HMMInclusionOperator > hmm_incl_real_assembler(hmmincl_real);
      LocalAssembler::Codim0Matrix< HMMInclusionOperator > hmm_incl_imag_assembler(hmmincl_imag);
      assert(filter_scatterer_);
      walker.add(hmm_ell_assembler, system_matrix_real_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_scatterer_));
      walker.add(hmm_incl_real_assembler, system_matrix_real_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_scatterer_));
      walker.add(hmm_incl_imag_assembler, system_matrix_imag_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_scatterer_));

      //lhs outside scatterer
      MacroConstFct one(1.0);
      //std::cout<< "watch out: diffusion_param outside scatterer is not 1.0!!!!"<<std::endl;
      assert(filter_outside_);
      typedef GDT::Operators::EllipticCG< MacroConstFct, RealMatrixType, SpaceType > EllipticOperatorType;
      EllipticOperatorType elliptic_operator_real(one, system_matrix_real_, coarse_space_);
      walker.add(elliptic_operator_real, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_outside_));
      //identity part
      MacroConstFct wavenumber_fct(-1.0*wavenumber_);
      MacroConstFct wavenumber_fct_squared(-1.0*wavenumber_ * wavenumber_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< MacroConstFct > > IdOperatorType;
      const IdOperatorType identity_operator_real(wavenumber_fct_squared);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler_real(identity_operator_real);
      walker.add(idMatrixAssembler_real, system_matrix_real_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_outside_));

      //boundary part for complex Robin-type condition
      typedef LocalOperator::Codim1BoundaryIntegral< LocalEvaluation::Product< MacroConstFct > > BdryOperatorType;
      const BdryOperatorType bdry_operator(wavenumber_fct);
      const LocalAssembler::Codim1BoundaryMatrix< BdryOperatorType > bdry_assembler(bdry_operator);
      walker.add(bdry_assembler, system_matrix_imag_, new Stuff::Grid::ApplyOn::NeumannIntersections< MacroGridViewType >(bdry_info_));

      //assemble
      std::cout<< "macro assembly" <<std::endl;
      walker.assemble();

      //assembly of total (complex) matrix and vector
      std::complex< double > im(0.0, 1.0);
      system_matrix_.backend() = system_matrix_imag_.backend().template cast< std::complex< double > >();
      system_matrix_.scal(im);
      system_matrix_.backend() += system_matrix_real_.backend().template cast< std::complex< double > >();
      rhs_vector_.backend() = rhs_vector_imag_.backend().template cast< std::complex< double > >();
      rhs_vector_.scal(im);
      rhs_vector_.backend() += rhs_vector_real_.backend().template cast< std::complex< double > > ();

      is_assembled_ = true;
    }
  }  //assemble

  bool assembled() const
  {
    return is_assembled_;
  }

  Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > effective_a() const
  {
    return elliptic_cell_.effective_matrix();
  }

  std::complex< RangeFieldType > effective_mu() const
  {
    return inclusion_cell_.effective_param();
  }

  const MatrixType& system_matrix() const
  {
    return system_matrix_;
  }

  const VectorType& rhs_vector() const
  {
    return rhs_vector_;
  }

  void solve(VectorType& solution,
             const Dune::Stuff::Common::Configuration& options = Dune::Stuff::LA::Solver< MatrixType >::options("bicgstab.diagonal")) const
  {
    if (!is_assembled_)
      assemble();
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, options);
  }

  /**
   * @brief solve_and_correct solves the HMM problem and directly computes the (discrete) correctors
   * @param macro_solution a vector (2 items, first for real, second for imaginary part) of DiscreteFunction (the space has to be given),
   *  where the macro_solution of the HMM will be saved to
   * @return a pair of PeriodicCorrector objects for the curl and identity corrector function
   */
  std::pair< Dune::GDT::PeriodicCorrector< DiscreteFunctionType, EllipticCellDiscreteFctType >, Dune::GDT::PeriodicCorrector< DiscreteFunctionType, InclusionCellDiscreteFctType > >
    solve_and_correct(std::vector< DiscreteFunctionType >& macro_solution,
                      const Dune::Stuff::Common::Configuration& options = Dune::Stuff::LA::Solver< MatrixType >::options("bicgstab.diagonal"))
  {
    if (!is_assembled_)
      assemble();
    VectorType solution(coarse_space_.mapper().size());
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, options);
    //get real and imaginary part and make discrete functions
    RealVectorType solution_real(coarse_space_.mapper().size());
    RealVectorType solution_imag(coarse_space_.mapper().size());
    solution_real.backend() = solution.backend().real();
    solution_imag.backend() = solution.backend().imag();
    assert(macro_solution.size() > 1);
    macro_solution[0].vector() = solution_real;
    macro_solution[1].vector() = solution_imag;
    //compute cell problems
    AllEllipticSolutionsStorageType elliptic_cell_solutions(dimDomain);
    for (auto& it : elliptic_cell_solutions) {
      std::vector< EllipticCellDiscreteFctType >it1(1, EllipticCellDiscreteFctType(elliptic_cell_.cell_space()));
      it = DSC::make_unique< typename EllipticCellProblem::CellDiscreteFunctionType >(it1);
    }
    std::cout<< "computing elliptic cell problems"<< std::endl;
    elliptic_cell_.compute_cell_solutions(elliptic_cell_solutions);
    AllInclusionSolutionsStorageType inclusion_cell_solutions(1);
    for (auto& it : inclusion_cell_solutions) {
      std::vector< InclusionCellDiscreteFctType>it1(2, InclusionCellDiscreteFctType(inclusion_cell_.cell_space()));
      it = DSC::make_unique< typename InclusionCellProblem::CellDiscreteFunctionType >(it1);
    }
    std::cout<< "computing inclusion cell problems"<< std::endl;
    inclusion_cell_.compute_cell_solutions(inclusion_cell_solutions);

    std::vector< std::vector< EllipticCellDiscreteFctType > > elliptic_cell_functions(dimDomain,
                                                                                      std::vector< EllipticCellDiscreteFctType >(1, EllipticCellDiscreteFctType(elliptic_cell_.cell_space())));
    std::vector< std::vector< InclusionCellDiscreteFctType > > incl_cell_functions(1,
                                                                                   std::vector< InclusionCellDiscreteFctType >(2, InclusionCellDiscreteFctType(inclusion_cell_.cell_space())));
    for (size_t ii = 0; ii < dimDomain; ++ii){
      elliptic_cell_functions[ii][0].vector() = elliptic_cell_solutions[ii]->operator[](0).vector();
    }
    incl_cell_functions[0][0].vector() = inclusion_cell_solutions[0]->operator[](0).vector();
    incl_cell_functions[0][1].vector() = inclusion_cell_solutions[0]->operator[](1).vector();
    //build correctors
    return std::make_pair(Dune::GDT::PeriodicCorrector< DiscreteFunctionType, EllipticCellDiscreteFctType >(macro_solution, elliptic_cell_functions, "elliptic"),
                          Dune::GDT::PeriodicCorrector< DiscreteFunctionType, InclusionCellDiscreteFctType >(macro_solution, incl_cell_functions, "id_incl"));
  } //solve and correct

  /** \brief computes the error between a reference solution (to the heterogeneous problem, on a fine grid) to the HMM approximation
   *
   * the HMM approximation is turned into a \ref DeltaCorrectorHelmholtz and the L2 or H1 seminorms can be requested
   * \tparam ReferenceFunctionImp Type for the discrete reference solution
   * \tparam CoarseFunctionImp Type for the macroscopic part of the HMM approximation
   * \tparam EllipticCellFunctionImp Type for the solutions to the cell problem for the gradient
   * \tparam InclusionCellFunctionImp Type for the solutions to the cell problem for the identity part in the inclusions
   */
  template< class ReferenceFunctionImp, class CoarseFunctionImp, class EllipticCellFunctionImp, class InclusionCellFunctionImp >
  RangeFieldType reference_error(std::vector< ReferenceFunctionImp >& reference_sol,
                                 Dune::GDT::PeriodicCorrector< CoarseFunctionImp, EllipticCellFunctionImp >& elliptic_corrector,
                                 Dune::GDT::PeriodicCorrector< CoarseFunctionImp, InclusionCellFunctionImp >& inclusion_corrector,
                                 double delta, std::string type)
  {
    //build DeltaCorrectorHelmholtz
    typedef Dune::GDT::DeltaCorrectorHelmholtz< CoarseFunctionImp, EllipticCellFunctionImp, InclusionCellFunctionImp > DeltaCorrectorType;
    DeltaCorrectorType corrector_real(elliptic_corrector.macro_function(), elliptic_corrector.cell_solutions(), inclusion_corrector.cell_solutions(),
                                      filter_scatterer_, filter_inclusion_, wavenumber_, delta, "real");
    DeltaCorrectorType corrector_imag(elliptic_corrector.macro_function(), elliptic_corrector.cell_solutions(), inclusion_corrector.cell_solutions(),
                                      filter_scatterer_, filter_inclusion_, wavenumber_, delta, "imag");
    //build errors
    typedef Dune::Stuff::Functions::Difference< ReferenceFunctionImp, DeltaCorrectorType > DifferenceFunctionType;
    DifferenceFunctionType error_real(reference_sol[0], corrector_real);
    DifferenceFunctionType error_imag(reference_sol[1], corrector_imag);
    //compute errors depending on type
    RangeFieldType result = 0;
    if(type == "l2") {
      Dune::GDT::Products::L2< typename ReferenceFunctionImp::SpaceType::GridViewType > l2_product(reference_sol[0].space().grid_view());
      result += l2_product.apply2(error_real, error_real);
      result += l2_product.apply2(error_imag, error_imag);
      return std::sqrt(result);
    }
    else if(type == "h1semi") {
      Dune::GDT::Products::H1Semi< typename ReferenceFunctionImp::SpaceType::GridViewType > h1_product(reference_sol[0].space().grid_view());
      result += h1_product.apply2(error_real, error_real);
      result += h1_product.apply2(error_imag, error_imag);
      return std::sqrt(result);
    }
    else
      DUNE_THROW(Dune::NotImplemented, "This type of norm is not implemented");
  } //reference error

private:
  const SpaceType                     coarse_space_;
  const BoundaryInfoType&             bdry_info_;
  const MacroScalarFct&               macro_a_diel_;
  const MacroScalarFct&               macro_a_incl_real_;
  const MacroScalarFct&               macro_a_incl_imag_;
  const MacroScalarFct&               bdry_real_;
  const MacroScalarFct&               bdry_imag_;
  const CellScalarFct&                periodic_a_diel_;
  const CellScalarFct&                periodic_a_incl_real_;
  const CellScalarFct&                periodic_a_incl_imag_;
  const CellScalarFct&                stabil_param_;
  const double                        wavenumber_;
  InclusionConstantFct                k_squared_neg_;
  const MacroFilterType               filter_scatterer_;
  const MacroFilterType               filter_outside_;
  const CellFilterType                filter_inclusion_;
  const EllipticCellProblem           elliptic_cell_;
  const InclusionCellProblem          inclusion_cell_;
  mutable bool                        is_assembled_;
  mutable RealMatrixType              system_matrix_real_;
  mutable RealMatrixType              system_matrix_imag_;
  mutable MatrixType                  system_matrix_;
  mutable RealVectorType              rhs_vector_real_;
  mutable RealVectorType              rhs_vector_imag_;
  mutable VectorType                  rhs_vector_;
}; //class HMMHelmholtzDiscretization


#endif // DUNE_GDT_TEST_HMM_HELMHOLTZ_HH
