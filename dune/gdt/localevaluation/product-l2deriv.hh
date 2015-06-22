// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_EVALUATION_PRODUCT_L2DERIV_HH
#define DUNE_GDT_EVALUATION_PRODUCT_L2DERIV_HH

#include <type_traits>

#include <dune/common/dynmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>

#include <dune/stuff/functions/interfaces.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalEvaluation {

//forward
template< class LocalizableFunctionImp >
class L2grad;

template< class LocalizableFunctionImp >
class L2curl;

namespace internal {


/**
 *  \brief Traits for the L2grad product.
 */
template< class LocalizableFunctionImp >
class L2gradTraits
{
  static_assert(Stuff::is_localizable_function< LocalizableFunctionImp >::value,
                "LocalizableFunctionImp has to be a localizable function.");
public:
  typedef LocalizableFunctionImp                              LocalizableFunctionType;
  typedef L2grad< LocalizableFunctionType >                   derived_type;
  typedef typename LocalizableFunctionType::EntityType        EntityType;
  typedef typename LocalizableFunctionType::DomainFieldType   DomainFieldType;
  typedef typename LocalizableFunctionType::LocalfunctionType LocalfunctionType;
  typedef std::tuple< std::shared_ptr< LocalfunctionType > >  LocalfunctionTupleType;
  static const size_t                                         dimDomain = LocalizableFunctionType::dimDomain;
}; // class L2gradTraits

/**
 *  \brief Traits for the L2curl product.
 */
template< class LocalizableFunctionImp >
class L2curlTraits
{
  static_assert(Stuff::is_localizable_function< LocalizableFunctionImp >::value,
                "LocalizableFunctionImp has to be a localizable function.");
public:
  typedef LocalizableFunctionImp                              LocalizableFunctionType;
  typedef L2curl< LocalizableFunctionType >                   derived_type;
  typedef typename LocalizableFunctionType::EntityType        EntityType;
  typedef typename LocalizableFunctionType::DomainFieldType   DomainFieldType;
  typedef typename LocalizableFunctionType::LocalfunctionType LocalfunctionType;
  typedef std::tuple< std::shared_ptr< LocalfunctionType > >  LocalfunctionTupleType;
  static const size_t                                         dimDomain = LocalizableFunctionType::dimDomain;
}; // class L2curlTraits


} //namespace internal


/**
 *  \brief  Computes a product evaluation between a vector valued local l2 function and the gradientss of a test space.
 */
template< class LocalizableFunctionImp >
class L2grad
  : public LocalEvaluation::Codim0Interface< internal::L2gradTraits< LocalizableFunctionImp >, 1 >
{
public:
  typedef internal::L2gradTraits< LocalizableFunctionImp > Traits;
  typedef typename Traits::LocalizableFunctionType          LocalizableFunctionType;
  typedef typename Traits::LocalfunctionTupleType           LocalfunctionTupleType;
  typedef typename Traits::EntityType                       EntityType;
  typedef typename Traits::DomainFieldType                  DomainFieldType;
  static const size_t                                       dimDomain = Traits::dimDomain;

  L2grad(const LocalizableFunctionType& inducingFunction)
    : inducingFunction_(inducingFunction)
  {}

  /// \name Required by all variants of LocalEvaluation::Codim0Interface
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(inducingFunction_.local_function(entity));
  }

  /// \}
  /// \name Required by LocalEvaluation::Codim0Interface< ..., 1 >
  /// \{

  /**
   * \brief extracts the local function and calls the correct order() method
   */
  template< class R, size_t rT, size_t rCT >
  size_t order(const LocalfunctionTupleType& localFuncs,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase) const
  {
    return order(*std::get< 0 >(localFuncs), testBase);
  }

  /**
   * \brief extracts the local function and calls the correct evaluate() method
   */
  template< class R, size_t rT, size_t rCT >
  void evaluate(const LocalfunctionTupleType& localFuncs,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicVector< R >& ret) const
  {
    evaluate(*std::get< 0 >(localFuncs), testBase, localPoint, ret);
  }

  /// \}
  /// \name Actual implementation of order
  /// \{

  /**
   * \note   for `LocalEvaluation::Codim0Interface< ..., 1 >`
   * \return localFunction.order() + testBase.order()-1
   */
  template< class R, size_t rL, size_t rCL, size_t rT, size_t rCT >
  size_t order(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rL, rCL >& localFunction,
               const Stuff::LocalfunctionSetInterface
                  < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase) const
  {
    return localFunction.order() + testBase.order()-1;
  }


  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
   * \brief computes a product between a local (l2) function and the gradients of a scalar test base
   */
  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& localFunction,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& testBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicVector< R >& ret) const
  {
    assert(r == dimDomain && "This cannot be computed!");
    typedef typename Stuff::LocalfunctionSetInterface
         < EntityType, DomainFieldType, dimDomain, R, 1, 1 >::JacobianRangeType JacobianRangeType;
    // evaluate local function
    const auto functionValue = localFunction.evaluate(localPoint);
    // evaluate test base
    const size_t size = testBase.size();
    std::vector< JacobianRangeType > testgradients(size, JacobianRangeType(0));
    testBase.jacobian(localPoint, testgradients);
    // compute product
    assert(ret.size() >= size);
    for (size_t ii = 0; ii < size; ++ii) {
      ret[ii] = functionValue * testgradients[ii][0];
    }
  } // ... evaluate(...)

  /// \}

private:
  const LocalizableFunctionType& inducingFunction_;
}; // class L2grad


/**
 *  \brief  Computes a product evaluation between a vector valued local l2 function and the curls of a test space.
 */
template< class LocalizableFunctionImp >
class L2curl
  : public LocalEvaluation::Codim0Interface< internal::L2curlTraits< LocalizableFunctionImp >, 1 >
{
public:
  typedef internal::L2curlTraits< LocalizableFunctionImp > Traits;
  typedef typename Traits::LocalizableFunctionType          LocalizableFunctionType;
  typedef typename Traits::LocalfunctionTupleType           LocalfunctionTupleType;
  typedef typename Traits::EntityType                       EntityType;
  typedef typename Traits::DomainFieldType                  DomainFieldType;
  static const size_t                                       dimDomain = Traits::dimDomain;

  L2curl(const LocalizableFunctionType& inducingFunction)
    : inducingFunction_(inducingFunction)
  {}

  /// \name Required by all variants of LocalEvaluation::Codim0Interface
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(inducingFunction_.local_function(entity));
  }

  /// \}
  /// \name Required by LocalEvaluation::Codim0Interface< ..., 1 >
  /// \{

  /**
   * \brief extracts the local function and calls the correct order() method
   */
  template< class R, size_t rT, size_t rCT >
  size_t order(const LocalfunctionTupleType& localFuncs,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase) const
  {
    return order(*std::get< 0 >(localFuncs), testBase);
  }

  /**
   * \brief extracts the local function and calls the correct evaluate() method
   */
  template< class R, size_t rT, size_t rCT >
  void evaluate(const LocalfunctionTupleType& localFuncs,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicVector< R >& ret) const
  {
    evaluate(*std::get< 0 >(localFuncs), testBase, localPoint, ret);
  }

  /// \}
  /// \name Actual implementation of order
  /// \{

  /**
   * \note   for `LocalEvaluation::Codim0Interface< ..., 1 >`
   * \return localFunction.order() + testBase.order()-1
   */
  template< class R, size_t rL, size_t rCL, size_t rT, size_t rCT >
  size_t order(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rL, rCL >& localFunction,
               const Stuff::LocalfunctionSetInterface
                  < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase) const
  {
    return localFunction.order() + testBase.order()-1;
  }


  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
   * \brief computes a product between a local (l2) function and the curls of a vectorial test base
   */
  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& localFunction,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& testBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicVector< R >& ret) const
  {
    assert(r == dimDomain && dimDomain == 3 && "curl only defined form r^3 to r^3!");
    typedef typename Stuff::LocalfunctionSetInterface
         < EntityType, DomainFieldType, dimDomain, R, r, 1 >::JacobianRangeType JacobianRangeType;
    // evaluate local function
    const auto functionValue = localFunction.evaluate(localPoint);
    // evaluate test base
    const size_t size = testBase.size();
    std::vector< JacobianRangeType > testgradients(size, JacobianRangeType(0));
    testBase.jacobian(localPoint, testgradients);
    // compute product
    assert(ret.size() >= size);
    for (size_t ii = 0; ii < size; ++ii) {
      ret[ii] = functionValue[0] * (testgradients[ii][2][1]-testgradients[ii][1][2])
                + functionValue[1] * (testgradients[ii][0][2]-testgradients[ii][2][0])
                + functionValue[2] * (testgradients[ii][1][0]-testgradients[ii][0][1]);
    }
  } // ... evaluate(...)

  /// \}

private:
  const LocalizableFunctionType& inducingFunction_;
}; // class L2grad

} //namespace LocalEvaluation
} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_EVALUATION_PRODUCT_L2DERIV_HH
