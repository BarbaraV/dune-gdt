// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH
#define DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH

#if !HAVE_DUNE_PDELAB
# error "This one requires dune-pdelab!"
#endif

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/localevaluation/elliptic.hh>
#include <dune/gdt/localevaluation/curlcurl.hh>
#include <dune/gdt/localevaluation/divdiv.hh>           // koennte auch noch divdiv-operator schreiben um tw. Konstruktionen zu verkuerzen
#include <dune/gdt/localevaluation/product-l2deriv.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/discretefunction/default.hh>

namespace Dune {
namespace GDT {
namespace Operators {


enum class ChooseCellProblem {Elliptic, CurlcurlDivreg};

//forward
template< class GridViewImp, int polynomialOrder, ChooseCellProblem >
class Cell;

namespace internal {


template< class GridViewImp, int polynomialOrder >
class CellTraits
{
public:
  typedef GridViewImp                                          GridViewType;
  typedef typename GridViewType::template Codim< 0 >::Entity   EntityType;
  typedef typename GridViewType::ctype                         DomainFieldType;
  static const size_t                                          dimDomain = GridViewType::dimension;

  static const unsigned int polOrder = polynomialOrder;
  typedef double            RangeFieldType;

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;

};


} //namespace internal


template< class GridViewImp, int polynomialOrder >
class Cell< GridViewImp, polynomialOrder, ChooseCellProblem::Elliptic >
{
public:
  typedef internal::CellTraits< GridViewImp, polynomialOrder> Traits;
  typedef typename Traits::EntityType EntityType;
  typedef typename Traits::DomainFieldType DomainFieldType;
  typedef typename Traits::RangeFieldType RangeFieldType;
  typedef typename Traits::MatrixType MatrixType;
  typedef typename Traits::VectorType VectorType;

  static const size_t       dimDomain = Traits::dimDomain;
  static const unsigned int polOrder = Traits::polOrder;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, RangeFieldType, 1 > SpaceType;

  Cell(const GridViewImp& gridview, const ScalarFct& kappa)
    : space_(gridview)
    , kappa_(kappa)
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
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      rhs_vector_ = VectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //lhs
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > > EllipticOp;
      EllipticOp ellipticop(kappa_);
      LocalAssembler::Codim0Matrix< EllipticOp > matrixassembler(ellipticop);
      walker.add(matrixassembler, system_matrix_);

      walker.assemble();
      is_assembled_ = true;
    }
  } //assemble

  bool is_assembled() const
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

  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, DiscreteFunction< SpaceType, VectorType > reconstruction) const
  {
    if(!is_assembled_)
      assemble();
    // set up rhs
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, typename RhsVectorType::field_type, dimDomain > ConstFct;
    externfctvalue *= -1.0;
    ConstFct constrhs(externfctvalue);
    typedef Stuff::Functions::Product< ScalarFct, ConstFct > RhsFuncType;
    RhsFuncType rhsfunc(kappa_, constrhs);
    typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2grad< RhsFuncType > > L2gradOp;
    L2gradOp l2gradop(rhsfunc);
    LocalAssembler::Codim0Vector< L2gradOp > vectorassembler(l2gradop);

    //assemble rhs
    SystemAssembler< SpaceType > walker(space_);
    walker.add(vectorassembler, rhs_vector_);
    walker.assemble();

    //solve
    auto solution = create_vector();
    Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "lu.sparse");

    //make discrete function
    reconstruction.vector() = solution;
  }

  const typename ScalarFct::RangeFieldType averageparameter() const
  {
    // integrate kappa_ over UnitCube
    return 1;
  }

private:
  const SpaceType    space_;
  const ScalarFct&   kappa_;
  mutable bool       is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
}; //class Cell<... ChoosecellProblem::Elliptic >


template< class GridViewImp, int polynomialOrder >
class Cell< GridViewImp, polynomialOrder, ChooseCellProblem::CurlcurlDivreg >
{
public:
  typedef internal::CellTraits< GridViewImp, polynomialOrder> Traits;
  typedef typename Traits::EntityType EntityType;
  typedef typename Traits::DomainFieldType DomainFieldType;
  typedef typename Traits::RangeFieldType RangeFieldType;
  typedef typename Traits::MatrixType MatrixType;
  typedef typename Traits::VectorType VectorType;

  static const size_t       dimDomain = Traits::dimDomain;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = Traits::polOrder;

  static_assert(dimDomain == 3, "This cell problem is only defined in 3d");

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, RangeFieldType, dimRange, 1 > SpaceType;

  Cell(const GridViewImp& gridview, const ScalarFct& mu,
       const ScalarFct& divparam = Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1>(1.0))
    : space_(gridview)
    , mu_(mu)
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
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      rhs_vector_ = VectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //lhs
      typedef LocalOperator::Codim0Integral< LocalEvaluation::CurlCurl< ScalarFct > > CurlOp;
      CurlOp curlop(mu_);
      LocalAssembler::Codim0Matrix< CurlOp > matrixassembler1(curlop);
      walker.add(matrixassembler1, system_matrix_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Divdiv< ScalarFct > > DivOp;
      DivOp divop(divparam_);
      LocalAssembler::Codim0Matrix< DivOp > matrixassembler2(divop);
      walker.add(matrixassembler2, system_matrix_);

      walker.assemble();
      is_assembled_ = true;
    }
  } //assemble

  bool is_assembled() const
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

  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, DiscreteFunction< SpaceType, VectorType > reconstruction) const
  {
    if(!is_assembled_)
      assemble();
    // set up rhs
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, typename RhsVectorType::field_type, dimDomain > ConstFct;
    externfctvalue *= -1.0;
    ConstFct constrhs(externfctvalue);
    typedef Stuff::Functions::Product< ScalarFct, ConstFct > RhsFuncType;
    RhsFuncType rhsfunc(mu_, constrhs);
    typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2curl< RhsFuncType > > L2curlOp;
    L2curlOp l2curlop(rhsfunc);
    LocalAssembler::Codim0Vector< L2curlOp > vectorassembler(l2curlop);

    //assemble rhs
    SystemAssembler< SpaceType > walker(space_);
    walker.add(vectorassembler, rhs_vector_);
    walker.assemble();

    //solve
    auto solution = create_vector();
    Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "lu.sparse");

    //make discrete function
    reconstruction.vector() = solution;
  }

  const typename ScalarFct::RangeFieldType averageparameter() const
  {
    //intergate mu_ over UnitCube
    return 1;
  }

private:
  const SpaceType    space_;
  const ScalarFct&   mu_;
  const ScalarFct&   divparam_;
  mutable bool       is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
}; //class Cell<... ChoosecellProblem::CurlcurlDivreg >


} //namespace Operators
} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH
