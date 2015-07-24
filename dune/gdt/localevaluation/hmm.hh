// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_EVALUATION_HMM_HH
#define DUNE_GDT_EVALUATION_HMM_HH

#include <tuple>

#include <boost/numeric/conversion/cast.hpp>

#include <dune/common/dynmatrix.hh>
#include <dune/common/typetraits.hh>

#include <dune/stuff/functions/interfaces.hh>

#include <dune/gdt/operators/cellreconstruction.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalEvaluation {


//forward
template< class FunctionImp, class GridViewImp, int polOrder >
class HMMCurlcurl;

template< class FunctionImp, class GridViewImp, int polOrder >
class HMMIdentity;

namespace internal {


template< class FunctionType, class GridViewImp, int polOrder >
class HMMCurlcurlTraits
{
  static_assert(Stuff::is_localizable_function< FunctionType >::value,
                "FunctionType has to be a localizable function!");
public:
  typedef HMMCurlcurl< FunctionType, GridViewImp, polOrder >                        derived_type;
  typedef typename FunctionType::EntityType                                         EntityType;
  typedef typename FunctionType::DomainFieldType                                    DomainFieldType;
  typedef std::tuple< std::shared_ptr< typename FunctionType::LocalfunctionType > > LocalfunctionTupleType;
  static const size_t                                                               dimDomain = FunctionType::dimDomain;

  typedef Operators::Cell< GridViewImp, polOrder, Operators::ChooseCellProblem::CurlcurlDivreg > CellProblemType;
}; //class HMMCurlcurlTraits

template< class FunctionType, class GridViewImp, int polOrder >
class HMMIdentityTraits
{
  static_assert(Stuff::is_localizable_function< FunctionType >::value,
                "FunctionType has to be a localizable function!");
public:
  typedef HMMIdentity< FunctionType, GridViewImp, polOrder >                        derived_type;
  typedef typename FunctionType::EntityType                                         EntityType;
  typedef typename FunctionType::DomainFieldType                                    DomainFieldType;
  typedef std::tuple< std::shared_ptr< typename FunctionType::LocalfunctionType > > LocalfunctionTupleType;
  static const size_t                                                               dimDomain = FunctionType::dimDomain;

  typedef Operators::Cell< GridViewImp, polOrder, Operators::ChooseCellProblem::Elliptic > CellProblemType;
}; //class HMMCurlcurlTraits


}//namespace internal


template< class FunctionImp, class GridViewImp, int polynomialOrder >
class HMMCurlcurl
  : public LocalEvaluation::Codim0Interface< internal::HMMCurlcurlTraits< FunctionImp, GridViewImp, polynomialOrder >, 2 >
{
public:
  typedef internal::HMMCurlcurlTraits< FunctionImp, GridViewImp, polynomialOrder > Traits;
  typedef FunctionImp                             FunctionType;
  typedef GridViewImp                             GridViewType;
  typedef typename Traits::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename Traits::EntityType             EntityType;
  typedef typename Traits::DomainFieldType        DomainFieldType;
  typedef typename Traits::CellProblemType        CellProblemType;
  static const size_t                             dimDomain = Traits::dimDomain;
  static const unsigned int                       polOrder = polynomialOrder;

  explicit HMMCurlcurl(const FunctionType& param,
                       const FunctionType& divparam,
                       const GridViewImp& cellgrid)
    : param_(param)
    , divparam_(divparam)
    , cellgridview_(cellgrid)
  {}

  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(param_.local_function(entity));
  }

  /**
   * \brief extracts the local functions and calls the correct order() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const LocalfunctionTupleType& localFuncs,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase) const
  {
    return order(*std::get< 0 >(localFuncs), testBase, ansatzBase);
  }

  /**
   * \brief extracts the local functions and calls the correct evaluate() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  void evaluate(const LocalfunctionTupleType& localFuncs,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    evaluate(*std::get< 0 >(localFuncs), testBase, ansatzBase, localPoint, ret);
  }

  /// \}
  /// \name Actual implmentation of order
  /// \{

  /**
    * \return localFunction.order()+(testBase.order()-1)+(ansatzBase.order()-1)
    */
  template< class R, size_t rL, size_t rCl, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rL, rCl >& localFunction,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase)
  const
  {
    return localFunction.order()
     + boost::numeric_cast< size_t >(std::max(ssize_t(testBase.order()) -1, ssize_t(0)))
     + boost::numeric_cast< size_t >(std::max(ssize_t(ansatzBase.order()) - 1, ssize_t(0)));
  } // ...order(....)


  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
    * \brief Computes a curlcurl evaluation for scalar local function and vector valued ansatz and test spaces
    * \tparam R RangeFieldType
    * \note at the moment, there is no global-valued (i.e. defined on macroscopic grid) local function to be evaluated, this is only ok for truly periodic cases
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& /*localFunction*/,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::JacobianRangeType JacobianRangeType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeType         RangeType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeFieldType    RangeFieldType;
    //evaluate testGradient
    const size_t rows = testBase.size();
    std::vector< JacobianRangeType > tGrad(rows, JacobianRangeType(0));
    testBase.jacobian(localPoint, tGrad);
    std::vector< RangeType > testcurl(rows, RangeType(0));
    //evaluate ansatz gradient
    const size_t cols = ansatzBase.size();
    std::vector< JacobianRangeType > aGrad(cols, JacobianRangeType(0));
    ansatzBase.jacobian(localPoint, aGrad);
    std::vector< RangeType > ansatzcurl(cols, RangeType(0));
    assert(ret.rows()>= rows);
    assert(ret.cols()>= cols);
    //set up and assemble cell problem
    CellProblemType cell(cellgridview_, param_, divparam_);
    cell.assemble();
    const auto& cellmatrix = cell.system_matrix();
    const auto averageparam = cell.averageparameter();
    //perpare temporary storage
    typedef DiscreteFunction< typename CellProblemType::SpaceType, typename CellProblemType::VectorType > DiscrFunction;
    typename CellProblemType::VectorType tmp_vector(cell.space().mapper().size());
    std::vector< DiscrFunction > testreconstr(rows, DiscrFunction(cell.space(), tmp_vector));
    std::vector< DiscrFunction > ansatzreconstr(cols, DiscrFunction(cell.space(), tmp_vector));
    std::vector< typename CellProblemType::VectorType > testrhs;
    std::vector< typename CellProblemType::VectorType > ansatzrhs;
    // perpare ansatz curls and their reconstruction
    for (size_t jj = 0; jj<cols; ++jj) {
      ansatzcurl[jj][0] = aGrad[jj][2][1]-aGrad[jj][1][2];
      ansatzcurl[jj][1] = aGrad[jj][0][2]-aGrad[jj][2][0];
      ansatzcurl[jj][2] = aGrad[jj][1][0]-aGrad[jj][0][1];
      cell.reconstruct(ansatzcurl[jj], ansatzreconstr[jj]);
      ansatzrhs.emplace_back(cell.rhs_vector());
      ansatzrhs[jj].scal(-1.0); //needed here because rhs = -1*param*ansatzcurl*curl cellfct
    }
    for (size_t ii = 0; ii<rows; ++ii) {
      //prepare test curls and their reconstruction
      testcurl[ii][0] = tGrad[ii][2][1]-tGrad[ii][1][2];
      testcurl[ii][1] = tGrad[ii][0][2]-tGrad[ii][2][0];
      testcurl[ii][2] = tGrad[ii][1][0]-tGrad[ii][0][1];
      cell.reconstruct(testcurl[ii], testreconstr[ii]);
      testrhs.emplace_back(cell.rhs_vector());
      testrhs[ii].scal(-1.0);
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj<cols; ++jj) {
       retRow[jj] += averageparam*(ansatzcurl[jj]*testcurl[ii]);
       retRow[jj] += testrhs[jj]*ansatzreconstr[ii].vector();
       retRow[jj] += testreconstr[jj].vector() * ansatzrhs[ii]; //for complex::conjugate!
       cellmatrix.mv(testreconstr[ii].vector(), tmp_vector); //check indices and order of multiplication here
       retRow[jj] += ansatzreconstr[jj].vector()*tmp_vector;  //the divdiv part is included in the system matrix
      } //lop over cols
    } //loop over rows
  } // ... evaluate (...)

private:
  const FunctionType& param_;
  const FunctionType& divparam_;
  const GridViewImp&  cellgridview_;
}; //class HMMCurlcurl

template< class FunctionImp, class GridViewImp, int polynomialOrder >
class HMMIdentity
  : public LocalEvaluation::Codim0Interface< internal::HMMIdentityTraits< FunctionImp, GridViewImp, polynomialOrder >, 2 >
{
public:
  typedef internal::HMMIdentityTraits< FunctionImp, GridViewImp, polynomialOrder > Traits;
  typedef FunctionImp                             FunctionType;
  typedef GridViewImp                             GridViewType;
  typedef typename Traits::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename Traits::EntityType             EntityType;
  typedef typename Traits::DomainFieldType        DomainFieldType;
  typedef typename Traits::CellProblemType        CellProblemType;
  static const size_t                             dimDomain = Traits::dimDomain;
  static const unsigned int                       polOrder = polynomialOrder;

  explicit HMMIdentity(const FunctionType& param,
                       const GridViewImp& cellgrid)
    : param_(param)
    , cellgridview_(cellgrid)
  {}

  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(param_.local_function(entity));
  }

  /**
   * \brief extracts the local functions and calls the correct order() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const LocalfunctionTupleType& localFuncs,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase) const
  {
    return order(*std::get< 0 >(localFuncs), testBase, ansatzBase);
  }

  /**
   * \brief extracts the local functions and calls the correct evaluate() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  void evaluate(const LocalfunctionTupleType& localFuncs,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    evaluate(*std::get< 0 >(localFuncs), testBase, ansatzBase, localPoint, ret);
  }

  /// \}
  /// \name Actual implmentation of order
  /// \{

  /**
    * \return localFunction.order()+(testBase.order()-1)+(ansatzBase.order()-1)
    */
  template< class R, size_t rL, size_t rCl, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rL, rCl >& localFunction,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase)
  const
  {
    return localFunction.order()
     + boost::numeric_cast< size_t >(std::max(ssize_t(testBase.order()) -1, ssize_t(0)))
     + boost::numeric_cast< size_t >(std::max(ssize_t(ansatzBase.order()) - 1, ssize_t(0)));
  } // ...order(....)


  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
    * \brief Computes a curlcurl evaluation for scalar local function and vector valued ansatz and test spaces
    * \tparam R RangeFieldType
    * \note at the moment, there is no global-valued (i.e. defined on macroscopic grid) local function to be evaluated
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& /*localFunction*/,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeType         RangeType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeFieldType    RangeFieldType;
    //evaluate test functions
    const size_t rows = testBase.size();
    auto tValue = testBase.evaluate(localPoint);
    //evaluate ansatz functions
    const size_t cols = ansatzBase.size();
    auto aValue = ansatzBase.evaluate(localPoint);
    assert(ret.rows()>= rows);
    assert(ret.cols()>= cols);
    //set up and assemble cell problem
    CellProblemType cell(cellgridview_, param_);
    cell.assemble();
    const auto& cellmatrix = cell.system_matrix();
    const auto averageparam = cell.averageparameter();
    //perpare temporary storage
    typename CellProblemType::VectorType tmp_vector(cell.space().mapper().size());
    typedef DiscreteFunction< typename CellProblemType::SpaceType , typename CellProblemType::VectorType >  DiscrFct;
    std::vector< DiscrFct > testreconstr(rows, DiscrFct(cell.space(), tmp_vector));
    std::vector< DiscrFct > ansatzreconstr(cols, DiscrFct(cell.space(), tmp_vector));
    std::vector< typename CellProblemType::VectorType > testrhs;
    std::vector< typename CellProblemType::VectorType > ansatzrhs;
    // perpare ansatz reconstruction
    for (size_t jj = 0; jj<cols; ++jj) {
      cell.reconstruct(aValue[jj], ansatzreconstr[jj]);
      ansatzrhs.emplace_back(cell.rhs_vector());
      ansatzrhs[jj].scal(-1.0);
    }
    for (size_t ii = 0; ii<rows; ++ii) {
      //prepare test reconstruction
      cell.reconstruct(tValue[ii], testreconstr[ii]);
      testrhs.emplace_back(cell.rhs_vector());
      testrhs[ii].scal(-1.0);
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj<cols; ++jj) {
        retRow[jj] += averageparam*(tValue[jj]*aValue[ii]);
        retRow[jj] += testrhs[jj]*ansatzreconstr[ii].vector();
        retRow[jj] += testreconstr[jj].vector() * ansatzrhs[ii];  //for complex:conjugate!
        cellmatrix.mv(testreconstr[ii].vector(), tmp_vector);
        retRow[jj] += ansatzreconstr[jj].vector()*tmp_vector;  //check for correct indices and order of multiplication!
      } //lop over cols
    } //loop over rows
  } // ... evaluate (...)

private:
  const FunctionType& param_;
  const GridViewImp&  cellgridview_;
}; //class HMMIdentity

}//namespace LocalEvaluation
}//namespace GDT
}//namespace Dune

#endif // DUNE_GDT_EVALUATION_HMM_HH
