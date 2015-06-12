// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_EVALUATION_DIVDIV_HH
#define DUNE_GDT_EVALUATION_DIVDIV_HH

#include <tuple>

#include <boost/numeric/conversion/cast.hpp>

#include <dune/common/dynmatrix.hh>
#include <dune/common/typetraits.hh>

#include <dune/stuff/functions/interfaces.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalEvaluation {

//forward
template< class ParamFunctionImp >
class Divdiv;

namespace internal {

/**
 * \brief Traits for the divdiv evaluation
 * \note only implemented for scalar parameter functions at the moment
*/
template< class ParamFunctionType >
class DivdivTraits
{
  static_assert(Stuff::is_localizable_function< ParamFunctionType >::value,
                "ParamFunctionType has to be a localizable function!");
public:
  typedef Divdiv< ParamFunctionType >                 derived_type;
  typedef typename ParamFunctionType::EntityType      EntityType;
  typedef typename ParamFunctionType::DomainFieldType DomainFieldType;
  typedef std::tuple< std::shared_ptr< typename ParamFunctionType::LocalfunctionType > > LocalfunctionTupleType;
  static const size_t dimDomain = ParamFunctionType::dimDomain;
}; //class DivdivTraits

}  //namespace internal


/**
  * \brief Computes a dividv evaluation: (paramfunction*div(ansatz function)* div(test function))
  * \note only implemented for scalar parameter functions at the moment
  */
template< class ParamFunctionImp >
class Divdiv
  : public LocalEvaluation::Codim0Interface< internal::DivdivTraits< ParamFunctionImp >, 2 >
{
public:
  typedef ParamFunctionImp                             ParamType;
  typedef internal::DivdivTraits< ParamFunctionImp >   Traits;
  typedef typename Traits::LocalfunctionTupleType      LocalfunctionTupleType;
  typedef typename Traits::EntityType                  EntityType;
  typedef typename Traits::DomainFieldType             DomainFieldType;
  static const size_t                                  dimDomain = Traits::dimDomain;

  explicit Divdiv(const ParamType& param)
    : param_(param)
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
    * \brief Computes a divdiv evaluation for scalar local function and vector valued ansatz and test spaces
    * \tparam R RangeFieldType
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunction,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    static_assert(r == dimDomain, "only possible for test and ansatz function from r^d to r^d");
    typedef typename Stuff::LocalfunctionSetInterface
         < EntityType, DomainFieldType, dimDomain, R, r, 1 >::JacobianRangeType JacobianRangeType;
    //evaluate local function
    const auto functionValue = localFunction.evaluate(localPoint);
    //evaluate test gradient
    const size_t rows = testBase.size();
    std::vector< JacobianRangeType > tGrad(rows, JacobianRangeType(0));
    testBase.jacobian(localPoint, tGrad);
    //evaluate ansatz gradient
    const size_t cols = ansatzBase.size();
    std::vector< JacobianRangeType > aGrad(cols, JacobianRangeType(0));
    ansatzBase.jacobian(localPoint, aGrad);
    //compute products
    assert(ret.rows()>= rows);
    assert(ret.cols() >= cols);
    for (size_t ii =0; ii<rows; ++ii) {
      auto& retRow = ret[ii];
      for(size_t jj = 0; jj < cols; ++jj) {
        for(size_t ll =0; ll < dimDomain; ++ll)
          retRow[jj] += tGrad[ii][ll][ll]*aGrad[jj][ll][ll];
        retRow[jj] *= functionValue;
      }
    }
  } // ... evaluate (...)

private:
  const ParamType& param_;
}; //class Divdiv



} //namespace LocalEvaluation
} //namespace GDT
} //namespace Dune


#endif // DUNE_GDT_EVALUATION_DIVDIV_HH
