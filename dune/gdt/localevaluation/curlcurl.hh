// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_EVALUATION_CURLCURL_HH
#define DUNE_GDT_EVALUATION_CURLCURL_HH

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
class CurlCurl;

namespace internal {

/**
 * \brief Traits for the CurlCurl evaluation
 * \note only implemented for scalar parameter functions at the moment
*/
template< class ParamFunctionType >
class CurlCurlTraits
{
  static_assert(Stuff::is_localizable_function< ParamFunctionType >::value,
                "ParamFunctionType has to be a localizable function!");
public:
  typedef CurlCurl< ParamFunctionType >               derived_type;
  typedef typename ParamFunctionType::EntityType      EntityType;
  typedef typename ParamFunctionType::DomainFieldType DomainFieldType;
  typedef std::tuple< std::shared_ptr< typename ParamFunctionType::LocalfunctionType > > LocalfunctionTupleType;
  static const size_t dimDomain = ParamFunctionType::dimDomain;
  static_assert(dimDomain == 3, "Curl only defined on r^3");
}; //class CurlCurlTraits

}  //namespace internal


/**
  * \brief Computes a curlcurl evaluation: (paramfunction*curl(ansatz function)* curl(test function))
  */
template< class ParamFunctionImp >
class CurlCurl
  : public LocalEvaluation::Codim0Interface< internal::CurlCurlTraits< ParamFunctionImp >, 2 >
{
public:
  typedef ParamFunctionImp                             ParamType;
  typedef internal::CurlCurlTraits< ParamFunctionImp > Traits;
  typedef typename Traits::LocalfunctionTupleType      LocalfunctionTupleType;
  typedef typename Traits::EntityType                  EntityType;
  typedef typename Traits::DomainFieldType             DomainFieldType;
  static const size_t                                  dimDomain = Traits::dimDomain;

  explicit CurlCurl(const ParamType& permeab)
    : mu_(permeab)
  {}


  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(mu_.local_function(entity));
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
    */

  template< class R>
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunction,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    typedef typename Stuff::LocalfunctionSetInterface
         < EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >::JacobianRangeType JacobianRangeType;
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
        retRow[jj] = functionValue *(((tGrad[ii][2][1]-tGrad[ii][1][2])*(aGrad[jj][2][1]-aGrad[jj][1][2]))
                                     + ((tGrad[ii][0][2]-tGrad[ii][2][0])*(aGrad[jj][0][2]-aGrad[jj][2][0]))
                                     + ((tGrad[ii][1][0]-tGrad[ii][0][1])*(aGrad[jj][1][0]-aGrad[jj][0][1])));
      }
    }
  } // ... evaluate (...)

  /**
   *  \brief  Computes a curlcurl evaluation for a 3x3 matrix-valued local function and vector-valued basefunctionsets.
   *  \tparam R RangeFieldType
   */
  template< class R >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, dimDomain >& localFunction,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    //evaluate local function
    const auto functionValue = localFunction.evaluate(localPoint);
    //evaluate test gradient
    const size_t rows = testBase.size();
    const auto testGradients = testBase.jacobian(localPoint);
    //evaluate ansatz gradient
    const size_t cols = ansatzBase.size();
    const auto ansatzGradients = ansatzBase.jacobian(localPoint);
    //compute ansatzcurls
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >::RangeType RangeType;
    std::vector< RangeType > ansatzCurls(cols, RangeType(0));
    std::vector< RangeType > testCurls(rows, RangeType(0));
    for (size_t jj = 0; jj < cols; ++jj){
      ansatzCurls[jj][0] = ansatzGradients[jj][2][1] - ansatzGradients[jj][1][2];
      ansatzCurls[jj][1] = ansatzGradients[jj][0][2] - ansatzGradients[jj][2][0];
      ansatzCurls[jj][2] = ansatzGradients[jj][1][0] - ansatzGradients[jj][0][1];
    }
    //compute test curls and products
    assert(ret.rows() >= rows);
    assert(ret.cols() >= cols);
    RangeType product(0.0);
    for (size_t ii = 0; ii < rows; ++ii){
      testCurls[jj][0] = testGradients[jj][2][1] - testGradients[jj][1][2];
      testCurls[jj][1] = testGradients[jj][0][2] - testGradients[jj][2][0];
      testCurls[jj][2] = testGradients[jj][1][0] - testGradients[jj][0][1];
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj < cols; ++jj){
        functionValue.mv(ansatzCurls[jj], product);
        retRow[jj] = product * testCurls[ii];
      }
    }
  } //evaluate with matrix local function

  /// \}

private:
  const ParamType& mu_;
}; //class CurlCurl



} //namespace LocalEvaluation
} //namespace GDT
} //namespace Dune


#endif // DUNE_GDT_EVALUATION_CURLCURL_HH
