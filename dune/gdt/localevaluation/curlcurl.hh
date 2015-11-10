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
#include <dune/stuff/functions/constant.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalEvaluation {

//forward
/*template< class ParamFunctionImp >
class CurlCurl; */
template< class ScalarParamImp, class TensorParamImp = void >
class CurlCurl;

namespace internal {

/**
 * \brief Traits for the CurlCurl evaluation
*/
template< class ScalarParamImp, class TensorParamImp >
class CurlCurlTraits
{
  static_assert(Stuff::is_localizable_function< ScalarParamImp >::value,
                "ScalarParamType has to be a localizable function!");
public:
  typedef typename ScalarParamImp::EntityType      EntityType;
  typedef typename ScalarParamImp::DomainFieldType DomainFieldType;
  static const size_t                              dimDomain = ScalarParamImp::dimDomain;
  static_assert(dimDomain == 3, "Curl only defined on r^3");

  // we need to distinguish three cases here (since TensorParamImp may be void):
  // given a two functions, a factor and a tensor
  // given a single factor (then set the tensor to default)
  // given a single tensor (then set the factor to default)
private:

  template< bool factor, bool tensor, bool anything = true >
  struct Helper { static_assert(AlwaysFalse< TensorParamImp >::value, "Unsupported combination of functions given!"); };

  // given both
  template< bool anything >
  struct Helper< false, false, anything >
  {
    static_assert(Stuff::is_localizable_function< TensorParamImp >::value,
                  "TensorParamType has to be a localizable function!");
    static_assert(std::is_same< typename ScalarParamImp::EntityType,
                                typename TensorParamImp::EntityType >::value,
                  "EntityTypes have to agree!");
    static_assert(std::is_same< typename ScalarParamImp::DomainFieldType,
                                typename TensorParamImp::DomainFieldType >::value,
                  "DomainFieldTypes have to agree!");
    static_assert(ScalarParamImp::dimDomain == TensorParamImp::dimDomain,
                  "Dimensions have to agree!");
    static_assert(ScalarParamImp::dimRange     == 1, "ScalarParamType has to be scalar!");
    static_assert(ScalarParamImp::dimRangeCols == 1, "ScalarParamType has to be scalar!");
    static_assert(TensorParamImp::dimRange     == TensorParamImp::dimDomain,
                  "TensorParamType has to be matrix valued!");
    static_assert(TensorParamImp::dimRangeCols == TensorParamImp::dimDomain,
                  "TensorParamType has to be matrix valued!");
    typedef ScalarParamImp FactorType;
    typedef TensorParamImp TensorType;
  };

  // given only one, and this is scalar
  template< bool anything >
  struct Helper< true, false, anything >
  {
    typedef ScalarParamImp                                                                                              FactorType;
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, DomainFieldType, dimDomain, dimDomain > TensorType;
  };

  // given only one, and this is a tensor
  template< bool anything >
  struct Helper< false, true, anything >
  {
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, DomainFieldType, 1, 1 > FactorType;
    typedef ScalarParamImp                                                                              TensorType;
  };

  static const bool single_factor_given =    ScalarParamImp::dimRange == 1
                                          && ScalarParamImp::dimRangeCols == ScalarParamImp::dimRange
                                          && std::is_same< TensorParamImp, void >::value;
  static const bool single_tensor_given =    ScalarParamImp::dimRange != 1
                                          && ScalarParamImp::dimRange == ScalarParamImp::dimRangeCols
                                          && std::is_same< TensorParamImp, void >::value;
public:
  typedef typename Helper< single_factor_given, single_tensor_given >::FactorType      ParamFactorType;
  typedef typename Helper< single_factor_given, single_tensor_given >::TensorType      ParamTensorType;
  typedef CurlCurl< ParamFactorType, ParamTensorType >                                 derived_type;
  typedef std::tuple< std::shared_ptr< typename ParamFactorType::LocalfunctionType >,
                      std::shared_ptr< typename ParamTensorType::LocalfunctionType > > LocalfunctionTupleType;
}; // class CurlCurlTraits

}  //namespace internal


/**
  * \brief Computes a curlcurl evaluation: (paramfunction * curl(ansatz function) * curl(test function))
  */
template< class ScalarParamImp, class TensorParamImp >
class CurlCurl
  : public LocalEvaluation::Codim0Interface< internal::CurlCurlTraits< ScalarParamImp, TensorParamImp >, 2 >
{
public:
  typedef internal::CurlCurlTraits< ScalarParamImp, TensorParamImp > Traits;
  typedef typename Traits::ParamFactorType                           ParamFactorType;
  typedef typename Traits::ParamTensorType                           ParamTensorType;
  typedef typename Traits::LocalfunctionTupleType                    LocalfunctionTupleType;
  typedef typename Traits::EntityType                                EntityType;
  typedef typename Traits::DomainFieldType                           DomainFieldType;
  static const size_t                                                dimDomain = Traits::dimDomain;
private:
  typedef Stuff::Common::ConstStorageProvider< ParamFactorType > ParamFactorProvider;
  typedef Stuff::Common::ConstStorageProvider< ParamTensorType > ParamTensorProvider;

public:
  CurlCurl(const ParamFactorType& permeab_factor, const ParamTensorType& permeab_matrix)
    : mu_factor_(permeab_factor)
    , mu_matrix_(permeab_matrix)
  {}

  CurlCurl(const ParamFactorType& permeab_factor)
    : mu_factor_(permeab_factor)
    , mu_matrix_(new ParamTensorType(Stuff::Functions::internal::UnitMatrix< typename ParamTensorType::RangeFieldType, dimDomain >::value()))
  {}

  template< typename ParamType // This disables the ctor if dimDomain == 1, since factor and tensor are then identical
          , typename = typename std::enable_if< (std::is_same< ParamType, ParamTensorType >::value) // and the ctors
                                                && (dimDomain > 1) && sizeof(ParamType) >::type >       // ambiguous.
  CurlCurl(const ParamType& permeab_matrix)
    : mu_factor_(new ParamFactorType(1.))
    , mu_matrix_(permeab_matrix)
  {}


  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(mu_factor_.access().local_function(entity),
                           mu_matrix_.access().local_function(entity));
  }

  /**
   * \brief extracts the local functions and calls the correct order() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const LocalfunctionTupleType& local_functions_tuple,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase) const
  {
    const auto local_mu_factor = std::get< 0 >(local_functions_tuple);
    const auto local_mu_matrix = std::get< 1 >(local_functions_tuple);
    return order(*local_mu_factor, *local_mu_matrix, testBase, ansatzBase);
  }

  /**
   * \brief extracts the local functions and calls the correct evaluate() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  void evaluate(const LocalfunctionTupleType& local_functions_tuple,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    const auto local_mu_factor = std::get< 0 >(local_functions_tuple);
    const auto local_mu_matrix = std::get< 1 >(local_functions_tuple);
    evaluate(*local_mu_factor, *local_mu_matrix, testBase, ansatzBase, localPoint, ret);
  }

  /// \}
  /// \name Actual implmentation of order
  /// \{

  /**
    * \return localFunction.order()+(testBase.order()-1)+(ansatzBase.order()-1)
    */
  template< class R, size_t rF, size_t rCF, size_t rM, size_t rCM, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rF, rCF >& local_mu_factor,
               const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rM, rCM >& local_mu_matrix,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase)
  const
  {
    return local_mu_factor.order()
     + local_mu_matrix.order()
     + boost::numeric_cast< size_t >(std::max(ssize_t(testBase.order()) -1, ssize_t(0)))
     + boost::numeric_cast< size_t >(std::max(ssize_t(ansatzBase.order()) - 1, ssize_t(0)));
  } // ...order(....)


  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
    * \brief Computes a curlcurl evaluation for scalar local function, matrix-valued local function and vector valued ansatz and test spaces
    * \tparam R RangeFieldType
    */

  template< class R>
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& local_mu_factor,
                const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, dimDomain >& local_mu_matrix,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    typedef typename Stuff::LocalfunctionSetInterface
         < EntityType, DomainFieldType, dimDomain, R, dimDomain, 1 >::JacobianRangeType JacobianRangeType;
    typedef Stuff::Common::FieldMatrix< R, dimDomain, dimDomain > TensorType;
    // evaluate local functions
    const auto       mu_factor_value = local_mu_factor.evaluate(localPoint);
    const TensorType mu_matrix_value = local_mu_matrix.evaluate(localPoint);
    const auto       mu_value        = mu_matrix_value * mu_factor_value;
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
    for (size_t ii = 0; ii < rows; ++ii){
      testCurls[ii][0] = testGradients[ii][2][1] - testGradients[ii][1][2];
      testCurls[ii][1] = testGradients[ii][0][2] - testGradients[ii][2][0];
      testCurls[ii][2] = testGradients[ii][1][0] - testGradients[ii][0][1];
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj < cols; ++jj){
        retRow[jj] = (mu_value * ansatzCurls[jj]) * testCurls[ii];
      }
    }
  } // ... evaluate (...)

  /// \}

private:
  const ParamFactorProvider mu_factor_;
  const ParamTensorProvider mu_matrix_;
}; //class CurlCurl

} //namespace LocalEvaluation
} //namespace GDT
} //namespace Dune


#endif // DUNE_GDT_EVALUATION_CURLCURL_HH
