// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_TEST_HMM_DISCRETIZATION_HH
#define DUNE_GDT_TEST_HMM_DISCRETIZATION_HH

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

#include <dune/gdt/spaces/nedelec/pdelab.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localevaluation/product.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/localoperator/hmmcodim0.hh>
#include <dune/gdt/operators/curlcurl.hh>
#include <dune/gdt/operators/cellreconstruction.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

//for error computation
#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/hcurl.hh>
#include <dune/gdt/discretefunction/corrector.hh>

//for HelmholtzDecomp
#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/operators/elliptic-cg.hh>


template< class MacroGridViewType, class CellGridType, int polynomialOrder >
class CurlHMMDiscretization {
public:
  typedef typename MacroGridViewType::ctype                                   MacroDomainFieldType;
  typedef typename MacroGridViewType::template Codim<0>::Entity               MacroEntityType;
  typedef double                                                              RangeFieldType;
  static const size_t       dimDomain = MacroGridViewType::dimension;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename MacroGridViewType::Intersection >                            BoundaryInfoType;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, dimRange > MacroVectorfct;
  typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >                 MacroConstFct;
  typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >                 MacroScalarFct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 RealMatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 RealVectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< MacroGridViewType, polOrder, double, dimRange > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, RealVectorType >                                 DiscreteFunctionType;

  typedef Dune::GDT::Operators::CurlCellReconstruction< SpaceType, CellGridType >                       CurlCellReconstruction;
  typedef Dune::GDT::Operators::IdEllipticCellReconstruction< SpaceType, CellGridType >                 EllipticCellReconstruction;
  typedef typename CurlCellReconstruction::PeriodicEntityType                                           CellEntityType;
  typedef typename CurlCellReconstruction::DomainFieldType                                              CellDomainFieldType;
  typedef typename CurlCellReconstruction::ScalarFct                                                    CellScalarFct;
  typedef Dune::Stuff::Functions::Constant< CellEntityType, CellDomainFieldType, dimDomain, double, 1 > CellConstFct;

  typedef Dune::GDT::DiscreteFunction< typename CurlCellReconstruction::CellSpaceType, RealVectorType >          CurlCellDiscreteFctType;
  typedef Dune::GDT::DiscreteFunction< typename EllipticCellReconstruction::CellSpaceType, RealVectorType >      EllCellDiscreteFctType;

  typedef typename CurlCellReconstruction::CellSolutionStorageType      AllCurlSolutionsStorageType;
  typedef typename EllipticCellReconstruction::CellSolutionStorageType  AllIdSolutionsStorageType;

  CurlHMMDiscretization(const MacroGridViewType& macrogridview,
                        CellGridType& cellgrid,
                        const BoundaryInfoType& info,
                        const CellScalarFct& mu,
                        const CellScalarFct& kappa_real,
                        const CellScalarFct& kappa_imag,
                        const MacroVectorfct& source_real,
                        const MacroVectorfct& source_imag,
                        const CellScalarFct& divparam,// = CellConstFct(1.0),
                        const CellScalarFct& stabil = CellConstFct(0.0001),
                        const MacroScalarFct& mu_macro = MacroConstFct(1.0),
                        const MacroScalarFct& kappa_real_macro = MacroConstFct(1.0),
                        const MacroScalarFct& kappa_imag_macro = MacroConstFct(1.0),
                        const bool is_periodic = true)
    : coarse_space_(macrogridview)
    , bdry_info_(info)
    , macro_mu_(mu_macro)
    , macro_kappa_real_(kappa_real_macro)
    , macro_kappa_imag_(kappa_imag_macro)
    , source_real_(source_real)
    , source_imag_(source_imag)
    , periodic_mu_(mu)
    , divparam_(divparam)
    , periodic_kappa_real_(kappa_real)
    , periodic_kappa_imag_(kappa_imag)
    , stabil_param_(stabil)
    , curl_cell_(coarse_space_, cellgrid, periodic_mu_, divparam_, stabil_param_)
    , ell_cell_(coarse_space_, cellgrid, periodic_kappa_real_, periodic_kappa_imag_, stabil_param_)
    , is_periodic_(is_periodic)
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

  const typename EllipticCellReconstruction::CellSpaceType& ell_cell_space() const
  {
    return ell_cell_.cell_space();
  }

  const typename CurlCellReconstruction::CellSpaceType& curl_cell_space() const
  {
    return curl_cell_.cell_space();
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
      auto source_functional_real = Functionals::make_l2_volume(source_real_, rhs_vector_real_, coarse_space_);
      walker.add(*source_functional_real);
      auto source_functional_imag = Functionals::make_l2_volume(source_imag_, rhs_vector_imag_, coarse_space_);
      walker.add(*source_functional_imag);

      Spaces::DirichletConstraints< typename MacroGridViewType::Intersection >
              dirichlet_constraints(bdry_info_, coarse_space_.mapper().size());

      if(!is_periodic_) {
        //lhs
        typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMCurlcurl< CellScalarFct, CurlCellReconstruction > > HMMCurlOperator;
        typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMIdentity< CellScalarFct, EllipticCellReconstruction > > HMMIdOperator;
        HMMCurlOperator hmmcurl(curl_cell_, periodic_mu_, divparam_, macro_mu_);
        HMMIdOperator hmmid_real(ell_cell_, periodic_kappa_real_, periodic_kappa_imag_, true, macro_kappa_real_, macro_kappa_imag_);
        HMMIdOperator hmmid_imag(ell_cell_, periodic_kappa_real_, periodic_kappa_imag_, false, macro_kappa_real_, macro_kappa_imag_);
        LocalAssembler::Codim0Matrix< HMMCurlOperator > hmm_curl_assembler(hmmcurl);
        LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_real_assembler(hmmid_real);
        LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_imag_assembler(hmmid_imag);
        walker.add(hmm_curl_assembler, system_matrix_real_);
        walker.add(hmm_id_real_assembler, system_matrix_real_);
        walker.add(hmm_id_imag_assembler, system_matrix_imag_);
      }
      else {
        //solve cell problems
        AllCurlSolutionsStorageType curl_cell_solutions(dimDomain);
          for (auto& it : curl_cell_solutions) {
            std::vector<DiscreteFunction< typename CurlCellReconstruction::CellSpaceType, RealVectorType > >
                    it1(1, DiscreteFunction< typename CurlCellReconstruction::CellSpaceType, RealVectorType >(curl_cell_.cell_space()));
            it = DSC::make_unique< typename CurlCellReconstruction::CellDiscreteFunctionType >(it1);
          }
        std::cout<< "computing curl cell problems"<< std::endl;
        curl_cell_.compute_cell_solutions(curl_cell_solutions);
        AllIdSolutionsStorageType ell_cell_solutions(dimDomain);
          for (auto& it : ell_cell_solutions) {
            std::vector<DiscreteFunction< typename EllipticCellReconstruction::CellSpaceType, RealVectorType > >
                    it1(2, DiscreteFunction< typename EllipticCellReconstruction::CellSpaceType, RealVectorType >(ell_cell_.cell_space()));
            it = DSC::make_unique< typename EllipticCellReconstruction::CellDiscreteFunctionType >(it1);
          }
        std::cout<< "computing identity cell problems"<< std::endl;
        ell_cell_.compute_cell_solutions(ell_cell_solutions);

        //lhs
        typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMCurlcurlPeriodic< CellScalarFct, CurlCellReconstruction > > HMMCurlOperator;
        typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMIdentityPeriodic< CellScalarFct, EllipticCellReconstruction > > HMMIdOperator;
        HMMCurlOperator hmmcurl(curl_cell_, periodic_mu_, divparam_, macro_mu_, curl_cell_solutions);
        HMMIdOperator hmmid_real(ell_cell_, periodic_kappa_real_, periodic_kappa_imag_, true, macro_kappa_real_, macro_kappa_imag_, ell_cell_solutions);
        HMMIdOperator hmmid_imag(ell_cell_, periodic_kappa_real_, periodic_kappa_imag_, false, macro_kappa_real_, macro_kappa_imag_, ell_cell_solutions);
        LocalAssembler::Codim0Matrix< HMMCurlOperator > hmm_curl_assembler(hmmcurl);
        LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_real_assembler(hmmid_real);
        LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_imag_assembler(hmmid_imag);
        walker.add(hmm_curl_assembler, system_matrix_real_);
        walker.add(hmm_id_real_assembler, system_matrix_real_);
        walker.add(hmm_id_imag_assembler, system_matrix_imag_);
      }

      //assemble Dirichlet constraints
      walker.add(dirichlet_constraints);
      std::cout<< "macro assembly" <<std::endl;
      walker.assemble();

      //apply the homogenous (!) Dirichlet constraints on the macro grid
      dirichlet_constraints.apply(system_matrix_real_, rhs_vector_real_);
      dirichlet_constraints.apply(system_matrix_imag_, rhs_vector_imag_);

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

  Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > effective_mu() const
  {
    return curl_cell_.effective_matrix();
  }

  std::vector< Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > > effective_kappa() const
  {
    return ell_cell_.effective_matrix();
  }

  const MatrixType& system_matrix() const
  {
    return system_matrix_;
  }

  const VectorType& rhs_vector() const
  {
    return rhs_vector_;
  }

  void solve(VectorType& solution) const
  {
    if (!is_assembled_)
      assemble();
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "bicgstab.diagonal");
  }

  /**
   * @brief solve_and_correct solves the HMM problem and directly computes the (discrete) correctors
   * @param macro_solution a vector (2 items, first for real, second for imaginary part) of DiscreteFunction (the dspace has to be given),
   *  where the macro_solution of the HMM will be saved to
   * @return a pair of PeriodicCorrector objects for the curl and identity corrector function
   */
  std::pair< Dune::GDT::PeriodicCorrector< DiscreteFunctionType, CurlCellDiscreteFctType >, Dune::GDT::PeriodicCorrector< DiscreteFunctionType, EllCellDiscreteFctType > >
    solve_and_correct(std::vector< DiscreteFunctionType >& macro_solution)
  {
    if(!is_periodic_)
      DUNE_THROW(Dune::NotImplemented, "Computation of correctors for non-periodic HMM not implemented yet");
    if (!is_assembled_)
      assemble();
    VectorType solution(coarse_space_.mapper().size());
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "bicgstab.diagonal");
    //get real and imaginary part and make discrete functions
    RealVectorType solution_real(coarse_space_.mapper().size());
    RealVectorType solution_imag(coarse_space_.mapper().size());
    solution_real.backend() = solution.backend().real();
    solution_imag.backend() = solution.backend().imag();
    assert(macro_solution.size() > 1);
    macro_solution[0].vector() = solution_real;
    macro_solution[1].vector() = solution_imag;
    //compute cell problems
    AllCurlSolutionsStorageType curl_cell_solutions(dimDomain);
    for (auto& it : curl_cell_solutions) {
      std::vector< CurlCellDiscreteFctType >it1(1, CurlCellDiscreteFctType(curl_cell_.cell_space()));
      it = DSC::make_unique< typename CurlCellReconstruction::CellDiscreteFunctionType >(it1);
    }
    std::cout<< "computing curl cell problems"<< std::endl;
    curl_cell_.compute_cell_solutions(curl_cell_solutions);
    AllIdSolutionsStorageType ell_cell_solutions(dimDomain);
    for (auto& it : ell_cell_solutions) {
      std::vector< EllCellDiscreteFctType>it1(2, EllCellDiscreteFctType(ell_cell_.cell_space()));
      it = DSC::make_unique< typename EllipticCellReconstruction::CellDiscreteFunctionType >(it1);
    }
    std::cout<< "computing identity cell problems"<< std::endl;
    ell_cell_.compute_cell_solutions(ell_cell_solutions);

    std::vector< std::vector< CurlCellDiscreteFctType > > curl_cell_functions(dimDomain, std::vector< CurlCellDiscreteFctType >(1, CurlCellDiscreteFctType(curl_cell_.cell_space())));
    std::vector< std::vector< EllCellDiscreteFctType > > ell_cell_functions(dimDomain, std::vector< EllCellDiscreteFctType >(2, EllCellDiscreteFctType(ell_cell_.cell_space())));
    for (size_t ii = 0; ii < dimDomain; ++ii){
      curl_cell_functions[ii][0].vector() = curl_cell_solutions[ii]->operator[](0).vector();
      ell_cell_functions[ii][0].vector() = ell_cell_solutions[ii]->operator[](0).vector();
      ell_cell_functions[ii][1].vector() = ell_cell_solutions[ii]->operator[](1).vector();
    }
    //build correctors
    return std::make_pair(Dune::GDT::PeriodicCorrector< DiscreteFunctionType, CurlCellDiscreteFctType >(macro_solution, curl_cell_functions, "curl"),
                          Dune::GDT::PeriodicCorrector< DiscreteFunctionType, EllCellDiscreteFctType >(macro_solution, ell_cell_functions, "id"));
  } //solve and correct

  /** \brief computes the error between an analytical and the discrete corrector
   *
   * the analytical corrector is given as \sum_i expected_macro_part[i] * expected_cell_solutions[i]
   * supported types are the error for the curl corrector ("curl", "curl_real" or "curl_imag") or the id corrector ("id", "id_real" or "id_imag")
   * \tparam ExpectedMacroFctType Type for the macroscopic part of the analytical corrector
   * \tparam ExpectedMicroFctType Type for the cell solutions of the analytical corrector
   * \tparam DiscreteMacroFctType Type for the macrosocpic part of the discrete (HMM) corrector
   * \tparam DiscreteMicroFctType Type for the discrete cell solutions of the HMm approximation
   *
   * \note If you want to compute the error for the curl corrector, the expected_macro_oart has to be the curl of your expected analytical solution
   */
  template< class ExpectedMacroFctType, class ExpectedMicroFctType, class DiscreteMacroFctType, class DiscreteMicroFctType >
  RangeFieldType corrector_error(std::vector< ExpectedMacroFctType >& expected_macro_part,
                                 std::vector< std::vector< ExpectedMicroFctType > >& expected_cell_solutions,
                                 Dune::GDT::PeriodicCorrector< DiscreteMacroFctType, DiscreteMicroFctType >& discrete_corrector,
                                 std::string type)
  {
    RangeFieldType result = 0;
    RangeFieldType cube_result = 0;
    typename ExpectedMicroFctType::JacobianRangeType micro_real(0);
    typename ExpectedMicroFctType::JacobianRangeType micro_imag(0);
    for (auto& macro_entity : DSC::entityRange(coarse_space_.grid_view())) {
      auto expected_macro_local_real = expected_macro_part[0].local_function(macro_entity);
      auto expected_macro_local_imag = expected_macro_part[1].local_function(macro_entity);
      auto discrete_correc_macro_local = discrete_corrector.local_function(macro_entity);
      size_t integrand_order = boost::numeric_cast< size_t >(2* std::max(ssize_t(expected_macro_local_real->order()), ssize_t(discrete_correc_macro_local->order())));
      typedef Dune::QuadratureRules< MacroDomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< MacroDomainFieldType, dimDomain > VolumeQuadratureType;
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(macro_entity.type(), boost::numeric_cast<int>(integrand_order));
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt) {
        const Dune::FieldVector< MacroDomainFieldType, dimDomain > xx = quadPointIt->position();
        //integration factors
        const double integration_factor = macro_entity.geometry().integrationElement(xx);
        const double quadrature_weight = quadPointIt->weight();
        auto expected_macro_value_real = expected_macro_local_real->evaluate(xx);
        auto expected_macro_value_imag = expected_macro_local_imag->evaluate(xx);
        std::vector< DiscreteMicroFctType > discrete_correc_macro(2, DiscreteMicroFctType(discrete_corrector.cell_space()));
        discrete_correc_macro_local->evaluate(xx, discrete_correc_macro);
        cube_result *= 0;
        //loop over micro entities
        for (auto& micro_entity : DSC::entityRange(discrete_corrector.cell_space().grid_view())) {
          auto local_discrete_correc_real = discrete_correc_macro[0].local_function(micro_entity);
          auto local_discrete_correc_imag = discrete_correc_macro[1].local_function(micro_entity);
          size_t integrand_order_micro = boost::numeric_cast< size_t >
                                        (2* std::max(ssize_t(expected_cell_solutions[0][0].local_function(micro_entity)->order()), ssize_t(local_discrete_correc_real->order())));
          typedef Dune::QuadratureRules< CellDomainFieldType, dimDomain > VolumeQuadratureRulesMicro;
          typedef Dune::QuadratureRule< CellDomainFieldType, dimDomain > VolumeQuadratureTypeMicro;
          const VolumeQuadratureTypeMicro& volumeQuadrature_micro = VolumeQuadratureRulesMicro::rule(micro_entity.type(), boost::numeric_cast<int>(integrand_order_micro));
          //loop over all microscopic quadrature points
          const auto quadPointEndIt_micro = volumeQuadrature_micro.end();
          for (auto quadPointIt_micro = volumeQuadrature_micro.begin(); quadPointIt_micro != quadPointEndIt_micro; ++quadPointIt_micro) {
            const auto yy = quadPointIt_micro->position();
            //integration factors
            const double integration_factor_micro = micro_entity.geometry().integrationElement(yy);
            const double quadrature_weight_micro = quadPointIt_micro->weight();
            //evaluate
            micro_real *= 0;
            micro_imag *= 0;
            //evaluate
            for (size_t jj = 0; jj < dimDomain; ++jj) {
              auto jacob_real = expected_cell_solutions[jj][0].local_function(micro_entity)->jacobian(yy);
              auto jacob_real1 = jacob_real;
              jacob_real *= expected_macro_value_real[jj];
              jacob_real1 *= expected_macro_value_imag[jj];
              auto jacob_imag = expected_cell_solutions[jj][1].local_function(micro_entity)->jacobian(yy);
              auto jacob_imag1 = jacob_imag;
              jacob_imag *= expected_macro_value_imag[jj];
              jacob_imag1 *= expected_macro_value_real[jj];
              micro_real += jacob_real;
              micro_real -= jacob_imag;
              micro_imag += jacob_real1;
              micro_imag += jacob_imag1;
            }
            micro_real -= local_discrete_correc_real->jacobian(yy);
            micro_imag -= local_discrete_correc_imag->jacobian(yy);
            //compute local contribution to the norm
            if(type == "id")
              cube_result += quadrature_weight_micro * integration_factor_micro * (micro_real[0].two_norm2() + micro_imag[0].two_norm2());
            else if(type == "id_real")
              cube_result += quadrature_weight_micro * integration_factor_micro * micro_real[0].two_norm2();
            else if(type == "id_imag")
              cube_result += quadrature_weight_micro * integration_factor_micro * micro_imag[0].two_norm2();
            else if(type == "curl")
              cube_result += quadrature_weight_micro * integration_factor_micro * (micro_real.frobenius_norm2() + micro_imag.frobenius_norm2());
            else if(type == "curl_real")
              cube_result += quadrature_weight_micro * integration_factor_micro * micro_real.frobenius_norm2();
            else if(type == "curl_imag")
              cube_result += quadrature_weight_micro * integration_factor_micro * micro_imag.frobenius_norm2();
            else
              DUNE_THROW(Dune::NotImplemented, "This type of norm is not implemented");
            }//loop over micro quadrature points
          }//loop over micro entities
          result += quadrature_weight * integration_factor * cube_result;
        } //loop over macro quadrature points
      }//loop over macro entities
      return std::sqrt(result);
  } //corrector error

  /** \brief computes the error between a reference solution (to the heterogeneous problem, on a fine grid) to the HMM approximation
   *
   * the HMM approximation is turned into a \ref DeltaCorrectorCurl and the L2 or Hcurl seminorms can be requested
   * \tparam ReferenceFunctionImp Type for the discrete reference solution
   * \tparam CoarseFunctionImp Type for the macroscopic part of the HMM approximation
   * \tparam CurlCellFunctionImp Type for the solutions to the cell problem for the curl
   * \tparam IdCellFunctionImp Type for the solutions to the cell problem for the identity part
   */
  template< class ReferenceFunctionImp, class CoarseFunctionImp, class CurlCellFunctionImp, class IdCellFunctionImp >
  RangeFieldType reference_error(std::vector< ReferenceFunctionImp >& reference_sol,
                                 Dune::GDT::PeriodicCorrector< CoarseFunctionImp, CurlCellFunctionImp >& curl_corrector,
                                 Dune::GDT::PeriodicCorrector< CoarseFunctionImp, IdCellFunctionImp >& id_corrector,
                                 double delta, std::string type)
  {
    //build DeltaCorrectorCurl
    typedef Dune::GDT::DeltaCorrectorCurl< CoarseFunctionImp, CurlCellFunctionImp, IdCellFunctionImp > DeltaCorrectorType;
    DeltaCorrectorType corrector_real(curl_corrector.macro_function(), curl_corrector.cell_solutions(), id_corrector.cell_solutions(), delta, "real");
    DeltaCorrectorType corrector_imag(curl_corrector.macro_function(), curl_corrector.cell_solutions(), id_corrector.cell_solutions(), delta, "imag");
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
    else if(type == "hcurlsemi") {
      Dune::GDT::Products::HcurlSemi< typename ReferenceFunctionImp::SpaceType::GridViewType > hcurl_product(reference_sol[0].space().grid_view());
      result += hcurl_product.apply2(error_real, error_real);
      result += hcurl_product.apply2(error_imag, error_imag);
      return std::sqrt(result);
    }
    else
      DUNE_THROW(Dune::NotImplemented, "This type of norm is not implemented");
  } //reference error


private:
  const SpaceType                     coarse_space_;
  const BoundaryInfoType&             bdry_info_;
  const MacroScalarFct&               macro_mu_;
  const MacroScalarFct&               macro_kappa_real_;
  const MacroScalarFct&               macro_kappa_imag_;
  const MacroVectorfct&               source_real_;
  const MacroVectorfct&               source_imag_;
  const CellScalarFct&                periodic_mu_;
  const CellScalarFct&                divparam_;
  const CellScalarFct&                periodic_kappa_real_;
  const CellScalarFct&                periodic_kappa_imag_;
  const CellScalarFct&                stabil_param_;
  const CurlCellReconstruction        curl_cell_;
  const EllipticCellReconstruction    ell_cell_;
  const bool                          is_periodic_;
  mutable bool                        is_assembled_;
  mutable RealMatrixType              system_matrix_real_;
  mutable RealMatrixType              system_matrix_imag_;
  mutable MatrixType                  system_matrix_;
  mutable RealVectorType              rhs_vector_real_;
  mutable RealVectorType              rhs_vector_imag_;
  mutable VectorType                  rhs_vector_;
};


template< class GridViewImp, int polynomialOrder >
class HelmholtzDecomp {
public:
  typedef typename GridViewImp::ctype DomainFieldType;
  static const size_t dimDomain = GridViewImp::dimension;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewImp::Intersection > BdryInfoType;
  typedef Dune::Stuff::Functions::Constant< typename GridViewImp::template Codim<0>::Entity, DomainFieldType, dimDomain, double, 1 > ConstantFct;
  typedef Dune::Stuff::Functions::Constant< typename GridViewImp::template Codim<0>::Entity, DomainFieldType, dimDomain, double, dimDomain > ConstantVectorFct;
  typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewImp::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimDomain > VectorFct;
  typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewImp::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 >        ScalarFct;


  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 VectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, polOrder, double, 1 > SpaceType;
  typedef Dune::GDT::ConstDiscreteFunction< SpaceType, VectorType > DiscreteFctType;


  HelmholtzDecomp(const GridViewImp& gp,
                  const BdryInfoType& info,
                  const VectorFct& sourcereal,
                  const VectorFct& sourceimag = ConstantVectorFct(0.0),
                  const ScalarFct& param = ConstantFct(1.0))
    : space_(gp)
    , boundary_info_(info)
    , source_real_(sourcereal)
    , source_imag_(sourceimag)
    , param_(param)
    , is_assembled_(false)
    , system_matrix_(0,0)
    , system_matrix_complex_(0,0)
    , rhs_vector_real_(0)
    , rhs_vector_imag_(0)
    , rhs_vector_total_(0)
  {}

  const SpaceType& space() const
  {
    return space_;
  }

  VectorType create_vector() const
  {
    return VectorType(space_.mapper().size());
  }

  VectorTypeComplex create_complex_vector() const
  {
    return VectorTypeComplex(space_.mapper().size());
  }

  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if (!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_complex_ = MatrixTypeComplex(space_.mapper().size(), space_.mapper().size());
      rhs_vector_real_ = VectorType(space_.mapper().size());
      rhs_vector_imag_ = VectorType(space_.mapper().size());
      rhs_vector_total_ = VectorTypeComplex(space_.mapper().size());
      SystemAssembler< SpaceType > grid_walker(space_);

      //lhs
      typedef Operators::EllipticCG< ScalarFct, MatrixType, SpaceType > EllipticOpType;
      EllipticOpType elliptic_operator(param_, system_matrix_, space_);
      grid_walker.add(elliptic_operator);

      //rhs
      typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2grad< VectorFct > > L2gradOp;
      const L2gradOp l2gradreal(source_real_);
      const L2gradOp l2gradimag(source_imag_);
      typedef LocalAssembler::Codim0Vector< L2gradOp > VectorAssType;
      VectorAssType vectorassembler1(l2gradreal);
      VectorAssType vectorassembler2(l2gradimag);
      grid_walker.add(vectorassembler1, rhs_vector_real_);
      grid_walker.add(vectorassembler2, rhs_vector_imag_);

      //dirichlet constraints
      Spaces::DirichletConstraints< typename GridViewImp::Intersection >
              dirichlet_constraints(boundary_info_, space_.mapper().size());
      grid_walker.add(dirichlet_constraints);
      grid_walker.assemble();

      //build complex matrix and rhs
      std::complex< double > im(0.0, 1.0);
      system_matrix_complex_.backend() = system_matrix_.backend().template cast<std::complex< double > >();
      rhs_vector_total_.backend() = rhs_vector_imag_.backend().template cast<std::complex< double > >();
      rhs_vector_total_.scal(im);
      rhs_vector_total_.backend() += rhs_vector_real_.backend().template cast< std::complex< double > >();

      dirichlet_constraints.apply(system_matrix_complex_, rhs_vector_total_);
      is_assembled_ = true;
    }
  } //assemble()

  bool assembled() const
  {
    return is_assembled_;
  }


  const MatrixTypeComplex& system_matrix() const
  {
    return system_matrix_complex_;
  }


  const VectorTypeComplex& rhs_vector() const
  {
    return rhs_vector_total_;
  }

  void solve(VectorTypeComplex& solution) const
  {
    if(!is_assembled_)
      assemble();

    // instantiate solver and options
    typedef Dune::Stuff::LA::Solver< MatrixTypeComplex > SolverType;
    SolverType solver(system_matrix_complex_);
    //solve
    solver.apply(rhs_vector_total_, solution, "bicgstab.diagonal");
  } //solve()


private:
  const SpaceType           space_;
  const BdryInfoType&       boundary_info_;
  const VectorFct&          source_real_;
  const VectorFct&          source_imag_;
  const ScalarFct&          param_;
  mutable bool              is_assembled_;
  mutable MatrixType        system_matrix_;
  mutable MatrixTypeComplex system_matrix_complex_;
  mutable VectorType        rhs_vector_real_;
  mutable VectorType        rhs_vector_imag_;
  mutable VectorTypeComplex rhs_vector_total_;
};

#endif // DUNE_GDT_TEST_HMM_DISCRETIZATION_HH
