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
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/nedelec/pdelab.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/operators/curlcurl.hh>
#include <dune/gdt/operators/cellreconstruction.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

//for truly HMM
#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localoperator/hmmcodim0.hh>

//for HelmholtzDecomp
#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/operators/elliptic-cg.hh>

template< class MacroGridViewType, class CellGridPartType, int polynomialOrder, class MacroParamImp = void >
class HMMDiscretization
{
public:
  typedef typename MacroGridViewType::ctype                                   MacroDomainFieldType;
  typedef typename MacroGridViewType::template Codim<0>::Entity               MacroEntityType;
  typedef typename CellGridPartType::ctype                                    CellDomainFieldType;
  typedef typename CellGridPartType::GridViewType::template Codim<0>::Entity  CellEntityType;
  static const size_t       dimDomain = MacroGridViewType::dimension;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename MacroGridViewType::Intersection >                            BoundaryInfoType;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, dimRange > MacroVectorfct;
  typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >                 MacroConstFct;
  typedef Dune::Stuff::LocalizableFunctionInterface< CellEntityType, CellDomainFieldType, dimDomain, double, 1 >          CellScalarFct;
  typedef Dune::Stuff::Functions::Constant< CellEntityType, CellDomainFieldType, dimDomain, double, 1 >                   CellConstFct;

private:
  template< class MacroParamType, bool anything = true >
  struct Helper {
    typedef MacroParamType Type;
  };

  template< bool anything >
  struct Helper< void, anything > {
    typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 > Type;
  };

public:
  typedef typename Helper< MacroParamImp, true >::Type MacroScalarFct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType ComplexMatrixType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType ComplexVectorType;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< MacroGridViewType, polOrder, double, dimRange > SpaceType;

  typedef Dune::GDT::ConstDiscreteFunction< SpaceType, VectorType > ConstDiscreteFunctionType;


  HMMDiscretization(const MacroGridViewType& macrogridview,
                    const CellGridPartType& cellgridpart,
                    const BoundaryInfoType& info,
                    const CellScalarFct& mu,
                    const CellScalarFct& kappa_real,
                    const CellScalarFct& kappa_imag,
                    const MacroVectorfct& source_real,
                    const MacroVectorfct& source_imag,
                    const CellScalarFct& divparam,// = CellConstFct(1.0),
                    const MacroScalarFct& mu_macro = MacroConstFct(1.0),
                    const MacroScalarFct& kappa_real_macro = MacroConstFct(1.0),
                    const MacroScalarFct& kappa_imag_macro = MacroConstFct(1.0))
    : space_(macrogridview)
    , cellgridpart_(cellgridpart)
    , bdry_info_(info)
    , mu_(mu)
    , mu_macro_(mu_macro)
    , kappa_real_(kappa_real)
    , kappa_imag_(kappa_imag)
    , kappa_real_macro_(kappa_real_macro)
    , kappa_imag_macro_(kappa_imag_macro)
    , source_real_(source_real)
    , source_imag_(source_imag)
    , divparam_(divparam)
    , is_assembled_(false)
    , system_matrix_real_(0,0)
    , system_matrix_imag_(0,0)
    , system_matrix_total_(0,0)
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

  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if (!is_assembled_) {
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_real_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_imag_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_total_ = ComplexMatrixType(space_.mapper().size(), space_.mapper().size());
      rhs_vector_real_ = VectorType(space_.mapper().size());
      rhs_vector_imag_ = VectorType(space_.mapper().size());
      rhs_vector_total_ = ComplexVectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //rhs
      auto source_functional_real = Functionals::make_l2_volume(source_real_, rhs_vector_real_, space_);
      walker.add(*source_functional_real);
      auto source_functional_imag = Functionals::make_l2_volume(source_imag_, rhs_vector_imag_, space_);
      walker.add(*source_functional_imag);

      //lhs with truly HMM
      /*typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMCurlcurl< ScalarFct, CellGridViewType, polOrder > > HMMcurlOp;
      const HMMcurlOp hmmcurl(mu_, divparam_, cellgridview_);
      const LocalAssembler::Codim0Matrix< HMMcurlOp > curlassembler(hmmcurl);
      walker.add(curlassembler, system_matrix_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMIdentity< ScalarFct, CellGridViewType, polOrder > > HMMidOp;
      const HMMidOp hmmid(kappa_, cellgridview_);
      const LocalAssembler::Codim0Matrix< HMMidOp > idassembler(hmmid);
      walker.add(idassembler, system_matrix_); */

      //lhs with effective matrices
      //solve cell problems/compute effective matrices
      GDT::Operators::FemCurlCell< CellGridPartType, polOrder > curlcell(cellgridpart_, mu_, divparam_);
      GDT::Operators::FemEllipticCell< CellGridPartType, polOrder > ellipticcell(cellgridpart_, kappa_real_, kappa_imag_);
      curlcell.assemble();
      ellipticcell.assemble();
      auto effective_mu = curlcell.effective_matrix();
      auto effective_kappa = ellipticcell.effective_matrix();
      auto effective_kappa_real = effective_kappa[0];
      auto effective_kappa_imag = effective_kappa[1];
      //the effective matrices have to be cast into constant macro functions
      typedef Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, dimDomain, dimDomain > MatrixFct;
      MatrixFct eff_mu_fct(effective_mu);
      MatrixFct eff_kappa_real_fct(effective_kappa_real);
      MatrixFct eff_kappa_imag_fct(effective_kappa_imag);
      typedef Stuff::Functions::Product< MacroScalarFct, MatrixFct > ProductFct;
      ProductFct kappa_real_fct(kappa_real_macro_, eff_kappa_real_fct, "stuff.functions.product");
      ProductFct kappa_imag_fct(kappa_imag_macro_, eff_kappa_imag_fct, "stuff.functions.product");

      //assemble macro lhs
      typedef LocalOperator::Codim0Integral< LocalEvaluation::CurlCurl< MacroScalarFct, MatrixFct > > CurlOpType;
      const CurlOpType curlop(mu_macro_, eff_mu_fct);
      const LocalAssembler::Codim0Matrix< CurlOpType > curlMatrixAssembler(curlop);
      walker.add(curlMatrixAssembler, system_matrix_real_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ProductFct > > IdOperatorType;
      const IdOperatorType idop1(kappa_real_fct);
      const IdOperatorType idop2(kappa_imag_fct);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler1(idop1);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler2(idop2);
      walker.add(idMatrixAssembler1, system_matrix_real_);
      walker.add(idMatrixAssembler2, system_matrix_imag_);

      //apply the homogenous (!) Dirichlet constraints on the macro grid
      Spaces::DirichletConstraints< typename MacroGridViewType::Intersection >
              dirichlet_constraints(bdry_info_, space_.mapper().size());
      walker.add(dirichlet_constraints);
      walker.assemble();
      dirichlet_constraints.apply(system_matrix_real_, rhs_vector_real_);
      dirichlet_constraints.apply(system_matrix_imag_, rhs_vector_imag_);

      //assembly of total (complex) matrix and vector
      std::complex< double > im(0.0, 1.0);
      system_matrix_total_.backend() = system_matrix_imag_.backend().template cast< std::complex< double > >();
      system_matrix_total_.scal(im);
      system_matrix_total_.backend() += system_matrix_real_.backend().template cast< std::complex< double > >();
      rhs_vector_total_.backend() = rhs_vector_imag_.backend().template cast< std::complex< double > >();
      rhs_vector_total_.scal(im);
      rhs_vector_total_.backend() += rhs_vector_real_.backend().template cast< std::complex< double > > ();

      is_assembled_ = true;
    }
  } //assemble

  bool assembled() const
  {
    return is_assembled_;
  }


  const ComplexMatrixType& system_matrix() const
  {
    return system_matrix_total_;
  }


  const ComplexVectorType& rhs_vector() const
  {
    return rhs_vector_total_;
  }

  void solve(ComplexVectorType& solution) const
  {
    if (!is_assembled_)
      assemble();

    Dune::Stuff::LA::Solver< ComplexMatrixType > solver(system_matrix_total_);
    solver.apply(rhs_vector_total_, solution, "bicgstab.diagonal");
  } //solve

private:
  const SpaceType space_;
  const CellGridPartType& cellgridpart_;
  const BoundaryInfoType& bdry_info_;
  const CellScalarFct& mu_;
  const MacroScalarFct& mu_macro_;
  const CellScalarFct& kappa_real_;
  const CellScalarFct& kappa_imag_;
  const MacroScalarFct& kappa_real_macro_;
  const MacroScalarFct& kappa_imag_macro_;
  const MacroVectorfct& source_real_;
  const MacroVectorfct& source_imag_;
  const CellScalarFct& divparam_;
  mutable bool is_assembled_;
  mutable MatrixType system_matrix_real_;
  mutable MatrixType system_matrix_imag_;
  mutable ComplexMatrixType system_matrix_total_;
  mutable VectorType rhs_vector_real_;
  mutable VectorType rhs_vector_imag_;
  mutable ComplexVectorType rhs_vector_total_;
}; //class HMMDiscretization


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
  typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 > MacroScalarFct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType RealMatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType RealVectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< MacroGridViewType, polOrder, double, dimRange > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, RealVectorType > DiscreteFunctionType;

  typedef Dune::GDT::Operators::CurlCellReconstruction< SpaceType, CellGridType > CurlCellReconstruction;
  typedef Dune::GDT::Operators::IdEllipticCellReconstruction< SpaceType, CellGridType > EllipticCellReconstruction;
  typedef typename CurlCellReconstruction::PeriodicEntityType CellEntityType;
  typedef typename CurlCellReconstruction::DomainFieldType CellDomainFieldType;
  typedef std::map< std::pair< size_t, size_t >, typename CurlCellReconstruction::CellSolutionStorageType > AllCurlSolutionsStorageType;
  typedef std::map< std::pair< size_t, size_t >, typename EllipticCellReconstruction::CellSolutionStorageType > AllIdSolutionsStorageType;
  typedef typename CurlCellReconstruction::ScalarFct CellScalarFct;
  typedef Dune::Stuff::Functions::Constant< CellEntityType, CellDomainFieldType, dimDomain, double, 1 > CellConstFct;

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
                        const MacroScalarFct& kappa_imag_macro = MacroConstFct(1.0))
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

      //solve and store cell problems
      AllCurlSolutionsStorageType curl_correctors;
      AllIdSolutionsStorageType id_correctors;
      std::cout<< "computing curl correctors" <<std::endl;
      curl_cell_.solve_for_all_quad_points(0, curl_correctors);
      std::cout<< "computing id correctors" <<std::endl;
      ell_cell_.solve_for_all_quad_points(2, id_correctors);    //automatically get order here!

      //lhs
      typedef LocalOperator::HMMCodim0Integral< LocalEvaluation::HMMCurlcurl< CellScalarFct, CurlCellReconstruction >, MacroGridViewType > HMMCurlOperator;
      typedef LocalOperator::HMMCodim0Integral< LocalEvaluation::HMMIdentity< CellScalarFct, EllipticCellReconstruction >, MacroGridViewType > HMMIdOperator;
      HMMCurlOperator hmmcurl(coarse_space_.grid_view(), curl_correctors, periodic_mu_, divparam_, macro_mu_);
      HMMIdOperator hmmid_real(coarse_space_.grid_view(), id_correctors, periodic_kappa_real_, periodic_kappa_imag_, true, macro_kappa_real_, macro_kappa_imag_);
      HMMIdOperator hmmid_imag(coarse_space_.grid_view(), id_correctors, periodic_kappa_real_, periodic_kappa_imag_, false, macro_kappa_real_, macro_kappa_imag_);
      LocalAssembler::Codim0Matrix< HMMCurlOperator > hmm_curl_assembler(hmmcurl);
      LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_real_assembler(hmmid_real);
      LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_imag_assembler(hmmid_imag);
      walker.add(hmm_curl_assembler, system_matrix_real_);
      walker.add(hmm_id_real_assembler, system_matrix_real_);
      walker.add(hmm_id_imag_assembler, system_matrix_imag_);

      //rhs
      auto source_functional_real = Functionals::make_l2_volume(source_real_, rhs_vector_real_, coarse_space_);
      walker.add(*source_functional_real);
      auto source_functional_imag = Functionals::make_l2_volume(source_imag_, rhs_vector_imag_, coarse_space_);
      walker.add(*source_functional_imag);

      //apply the homogenous (!) Dirichlet constraints on the macro grid
      Spaces::DirichletConstraints< typename MacroGridViewType::Intersection >
              dirichlet_constraints(bdry_info_, coarse_space_.mapper().size());
      walker.add(dirichlet_constraints);
      std::cout<< "macro assembly" <<std::endl;
      walker.assemble();
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

private:
  const SpaceType coarse_space_;
  const BoundaryInfoType& bdry_info_;
  const MacroScalarFct& macro_mu_;
  const MacroScalarFct& macro_kappa_real_;
  const MacroScalarFct& macro_kappa_imag_;
  const MacroVectorfct& source_real_;
  const MacroVectorfct& source_imag_;
  const CellScalarFct& periodic_mu_;
  const CellScalarFct& divparam_;
  const CellScalarFct& periodic_kappa_real_;
  const CellScalarFct& periodic_kappa_imag_;
  const CellScalarFct& stabil_param_;
  const CurlCellReconstruction curl_cell_;
  const EllipticCellReconstruction ell_cell_;
  mutable bool is_assembled_;
  mutable RealMatrixType system_matrix_real_;
  mutable RealMatrixType system_matrix_imag_;
  mutable MatrixType system_matrix_;
  mutable RealVectorType rhs_vector_real_;
  mutable RealVectorType rhs_vector_imag_;
  mutable VectorType rhs_vector_;
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
