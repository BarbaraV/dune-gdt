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
#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/operators/curlcurl.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

template< class MacroGridViewType, class CellGridViewType, int polynomialOrder >
class HMMDiscretization
{
public:
  typedef typename MacroGridViewType::ctype                     MacroDomainFieldType;
  typedef typename MacroGridViewType::template Codim<0>::Entity MacroEntityType;
  static const size_t                                           dimDomain = MacroGridViewType::dimension;
  static const size_t                                           dimRange = dimDomain;
  static const unsigned int                                     polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename MacroGridViewType::Intersection >                            BoundaryInfoType;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, dimRange > Vectorfct;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >        ScalarFct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< MacroGridViewType, polOrder, double, dimRange > SpaceType;

  typedef Dune::GDT::ConstDiscreteFunction < SpaceType, VectorType > ConstDiscreteFunctionType;


  HMMDiscretization(const MacroGridViewType& macrogridview,
                    const CellGridViewType& cellgridview,
                    const BoundaryInfoType& info,
                    const ScalarFct& mu,
                    const ScalarFct& kappa,
                    const Vectorfct& source,
                    const ScalarFct& divparam)
    : space_(macrogridview)
    , cellgridview_(cellgridview)
    , bdry_info_(info)
    , mu_(mu)
    , kappa_(kappa)
    , source_(source)
    , divparam_(divparam)
    , is_assembled_(false)
    , system_matrix_(0,0)
    , rhs_vector_(0)
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
      system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      rhs_vector_ = VectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //rhs
      auto source_functional = Functionals::make_l2_volume(source_, rhs_vector_, space_);
      walker.add(*source_functional);

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
      GDT::Operators::Cell< CellGridViewType, polOrder, GDT::Operators::ChooseCellProblem::CurlcurlDivreg > curlcell(cellgridview_, mu_, divparam_);
      GDT::Operators::Cell< CellGridViewType, polOrder, GDT::Operators::ChooseCellProblem::Elliptic > ellipticcell(cellgridview_, kappa_);
      curlcell.assemble();
      ellipticcell.assemble();
      auto effective_mu = curlcell.effective_matrix();
      auto effective_kappa = ellipticcell.effective_matrix();
      //the effective matrices have to be cassted into constant macro functions
      typedef Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, dimDomain, dimDomain > MatrixFct;
      MatrixFct eff_mu_fct(effective_mu);
      MatrixFct eff_kappa_fct(effective_kappa);

      //assemble macro lhs
      typedef GDT::Operators::CurlCurl< MatrixFct, MatrixType, SpaceType > CurlOpType;
      const CurlOpType curlop(eff_mu_fct, system_matrix_, space_);
      walker.add(curlop);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< MatrixFct > > IdOperatorType;
      const IdOperatorType idop(eff_kappa_fct);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler(idop);
      walker.add(idMatrixAssembler, system_matrix_);

      //apply the homogenous (!) Dirichlet constraints on the macro grid
      Spaces::DirichletConstraints< typename MacroGridViewType::Intersection >
              dirichlet_constraints(bdry_info_, space_.mapper().size());
      walker.add(dirichlet_constraints);
      walker.assemble();
      dirichlet_constraints.apply(system_matrix_, rhs_vector_);

      is_assembled_ = true;
    }
  } //assemble

  bool assembled() const
  {
    return is_assembled_;
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
  } //solve

private:
  const SpaceType space_;
  const CellGridViewType& cellgridview_;
  const BoundaryInfoType& bdry_info_;
  const ScalarFct& mu_;
  const ScalarFct& kappa_;
  const Vectorfct& source_;
  const ScalarFct& divparam_;
  mutable bool is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
};

#endif // DUNE_GDT_TEST_HMM_DISCRETIZATION_HH
