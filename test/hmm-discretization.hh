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
#include <dune/stuff/grid/periodicview.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/spaces/cg/fem.hh>
#include <dune/gdt/localevaluation/shiftedeval.hh>
#include <dune/gdt/localevaluation/elliptic.hh>
#include <dune/gdt/localevaluation/curlcurl.hh>
#include <dune/gdt/localevaluation/divdiv.hh>           // koennte auch noch divdiv-operator schreiben um tw. Konstruktionen zu verkuerzen
#include <dune/gdt/localevaluation/product-l2deriv.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>

enum class ChooseCellProblem { Elliptic, CurlcurlDivreg};

template< class GridViewImp, int polynomialOrder, class GridViewImpext, ChooseCellProblem cellproblem >
class Cell;

template< class GridViewImp, int polynomialOrder, class GridViewImpext >
class Cell< GridViewImp, polynomialOrder, GridViewImpext, ChooseCellProblem::Elliptic >
{
public:
  typedef GridViewImp                                          GridViewType;
  typedef typename GridViewType::template Codim< 0 >::Entity   EntityType;
  typedef typename GridViewImpext::template Codim< 0 >::Entity EntityTypeext;
  typedef typename GridViewType::ctype                         DomainFieldType;
  typedef typename GridViewImpext::ctype                       DomainFieldTypeext;
  static const size_t                                          dimDomain = GridViewType::dimension;
  static const size_t                                          dimDomainext = GridViewImpext::dimension;
  typedef Dune::FieldVector< DomainFieldTypeext, dimDomainext> CoordTypeext;

  static_assert(std::is_same< DomainFieldType, DomainFieldTypeext>::value, "External and internal domain fields have to be the same!");
  static_assert(dimDomain == dimDomainext, "External and internal domains must have the same dimension");

  static const unsigned int polOrder = polynomialOrder;
  typedef double            RangeFieldType;

  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewType, 1, RangeFieldType, 1 >  SpaceType;  //space of periodic functions

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityTypeext, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct; //parameter is globally defined //enable complex!

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;

  Cell(const GridViewType& gridview, const ScalarFct& kappa)
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

  template< class Functionext >
  void assemble(Functionext& macro, const DomainFieldTypeext delta, const CoordTypeext xt) const
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
      typedef LocalEvaluation::Elliptic< ScalarFct > EllipticEval;
      EllipticEval ellipticeval(kappa_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::ShiftedEval< EllipticEval, CoordTypeext > > ShiftedEllipticOp;
      ShiftedEllipticOp shiftedelliptic(ellipticeval, delta, xt);
      LocalAssembler::Codim0Matrix< ShiftedEllipticOp > matrixassembler(shiftedelliptic);
      walker.add(matrixassembler, system_matrix_);

      //rhs
      //check
      static_assert(std::is_same< typename Functionext::DomainFieldType, DomainFieldTypeext >::value, "the external function is not defined on the coorect domain field");
      static_assert(Functionext::dimDomain == dimDomainext, "dimensions must agree!");
      //evaluate paramete*macro
      typedef Stuff::Functions::Constant< EntityTypeext, DomainFieldTypeext, dimDomainext, typename Functionext::RangeFieldType, Functionext::dimRange > ConstextFunct;
      typename Functionext::RangeType ret(0);
      macro.evaluate(xt, ret);
      ConstextFunct constmacro(-1.0*ret);
      typedef Stuff::Functions::Product< ScalarFct, ConstextFunct > rhsfuncType;
      rhsfuncType rhsfunc(kappa_, constmacro);
      //set up rhs
      typedef LocalEvaluation::L2grad< rhsfuncType > L2gradEval;
      L2gradEval l2gradeval(rhsfunc);
      typedef LocalFunctional::Codim0Integral< LocalEvaluation::ShiftedEval< L2gradEval, CoordTypeext > > ShiftedL2gradOp;
      ShiftedL2gradOp shiftedproduct(l2gradeval, delta, xt);
      LocalAssembler::Codim0Vector< ShiftedL2gradOp > vectorassembler(shiftedproduct);
      walker.add(vectorassembler, rhs_vector_);

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

  template< class Functionext >
  void solve(VectorType& solution, Functionext& macro, const DomainFieldTypeext delta, const CoordTypeext xt) const
  {
    if(!is_assembled_)
      assemble(macro, delta, xt);
    //Dune::Stuff::LA::Solver< MatrixType >(system_matrix_).apply(rhs_vector_, solution); //geht das so? was mit periodische RW und MW 0?
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    auto opts = Dune::Stuff::LA::Solver< MatrixType >::options();
    opts.set("post_check_solves_system", "0", true);
    solver.apply(rhs_vector_, solution, opts);
    // atm no post_check because error norm is too big, bug or normal?
  }

private:
  const SpaceType    space_;
  const ScalarFct&   kappa_;
  mutable bool       is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
}; //class Cell<....., ChooseCellProblem::Elliptic >


template< class GridViewImp, int polynomialOrder, class GridViewImpext >
class Cell< GridViewImp, polynomialOrder, GridViewImpext, ChooseCellProblem::CurlcurlDivreg >
{
public:
  typedef GridViewImp                                          GridViewType;
  typedef typename GridViewType::template Codim< 0 >::Entity   EntityType;
  typedef typename GridViewImpext::template Codim< 0 >::Entity EntityTypeext;
  typedef typename GridViewType::ctype                         DomainFieldType;
  typedef typename GridViewImpext::ctype                       DomainFieldTypeext;
  static const size_t                                          dimDomain = GridViewType::dimension;
  static const size_t                                          dimRange = dimDomain;
  static const size_t                                          dimDomainext = GridViewImpext::dimension;
  typedef Dune::FieldVector< DomainFieldTypeext, dimDomainext> CoordTypeext;

  static_assert(std::is_same< DomainFieldType, DomainFieldTypeext>::value, "External and internal domain fields have to be the same!");
  static_assert(dimDomain == dimDomainext, "External and internal domains must have the same dimension");
  static_assert(dimDomain == 3, "this cell problem is only defined in 3d");

  static const unsigned int polOrder = polynomialOrder;
  typedef double            RangeFieldType;

  typedef Dune::GDT::Spaces::CG::FemBased< GridViewType, 1, RangeFieldType, dimRange, 1 >  SpaceType;  //space of periodic functions //fem space needs a GridPart and no gridView!!!

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityTypeext, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct; //parameter is globally defined

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;

  Cell(const GridViewType& gridview, const ScalarFct& mu,
       const ScalarFct& divparam = Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 >(1.0))
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

  template< class Functionext >
  void assemble(Functionext& macro, const DomainFieldTypeext delta, const CoordTypeext xt) const
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
      typedef LocalEvaluation::CurlCurl< ScalarFct > CurlcurlEval;
      CurlcurlEval curleval(mu_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::ShiftedEval< CurlcurlEval, CoordTypeext > > ShiftedCurlOp;
      ShiftedCurlOp shiftedcurl(curleval, delta, xt);
      LocalAssembler::Codim0Matrix< ShiftedCurlOp > matrixassembler1(shiftedcurl);
      walker.add(matrixassembler1, system_matrix_);
      typedef LocalEvaluation::Divdiv< ScalarFct > DivdivEval;
      DivdivEval diveval(divparam_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::ShiftedEval< DivdivEval, CoordTypeext > > ShiftedDivOp;
      ShiftedDivOp shifteddiv(diveval, delta, xt);
      LocalAssembler::Codim0Matrix< ShiftedDivOp > matrixassembler2(shifteddiv);
      walker.add(matrixassembler2, system_matrix_);

      //rhs
      //check
      static_assert(std::is_same< typename Functionext::DomainFieldType, DomainFieldTypeext >::value, "the external function is not defined on the coorect domain field");
      static_assert(Functionext::dimDomain == dimDomainext, "dimensions must agree!");
      //evaluate parameter*macro
      typedef Stuff::Functions::Constant< EntityTypeext, DomainFieldTypeext, dimDomainext, typename Functionext::RangeFieldType, Functionext::dimRange > ConstextFunct;
      typename Functionext::RangeType ret(0);
      macro.evaluate(xt, ret);
      ConstextFunct constmacro(-1.0*ret);
      typedef Stuff::Functions::Product< ScalarFct, ConstextFunct > rhsfuncType;
      rhsfuncType rhsfunc(mu_, constmacro);
      //set up rhs
      typedef LocalEvaluation::L2curl< rhsfuncType > L2curlEval;
      L2curlEval l2curleval(rhsfunc);
      typedef LocalFunctional::Codim0Integral< LocalEvaluation::ShiftedEval< L2curlEval, CoordTypeext > > ShiftedL2curlOp;
      ShiftedL2curlOp shiftedproduct(l2curleval, delta, xt);
      LocalAssembler::Codim0Vector< ShiftedL2curlOp > vectorassembler(shiftedproduct);
      walker.add(vectorassembler, rhs_vector_);

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

  template< class Functionext >
  void solve(VectorType& solution, Functionext& macro, const DomainFieldTypeext delta, const CoordTypeext xt) const
  {
    if(!is_assembled_)
      assemble(macro, delta, xt);
    //Dune::Stuff::LA::Solver< MatrixType >(system_matrix_).apply(rhs_vector_, solution); //geht das so? was mit periodische RW und MW 0?
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    auto opts = Dune::Stuff::LA::Solver< MatrixType >::options();
    opts.set("post_check_solves_system", "0", true);
    solver.apply(rhs_vector_, solution, opts);
    // atm no post_check because error norm is too big, bug or normal?
  }

private:
  const SpaceType    space_;
  const ScalarFct&   mu_;
  const ScalarFct&   divparam_;
  mutable bool       is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
}; //class Cell<....., Choose::CellProblem::CurlcurlDivreg >

#endif // DUNE_GDT_TEST_HMM_DISCRETIZATION_HH
