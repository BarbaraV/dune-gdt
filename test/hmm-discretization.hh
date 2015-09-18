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
//#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/operators/curlcurl.hh>
#include <dune/gdt/operators/cellreconstruction.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

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
};

#endif // DUNE_GDT_TEST_HMM_DISCRETIZATION_HH
