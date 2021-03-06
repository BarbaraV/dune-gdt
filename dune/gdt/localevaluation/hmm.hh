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
template< class FunctionImp, class CellProblemType >
class HMMCurlcurl;

template< class FunctionImp, class CellProblemType >
class HMMIdentity;

namespace internal {


template< class FunctionType, class CellProblemType >
class HMMCurlcurlTraits
{
  static_assert(Stuff::is_localizable_function< FunctionType >::value,
                "FunctionType has to be a localizable function!");
public:
  typedef HMMCurlcurl< FunctionType, CellProblemType >                              derived_type;
  typedef typename FunctionType::EntityType                                         EntityType;
  typedef typename FunctionType::DomainFieldType                                    DomainFieldType;
  typedef std::tuple< std::shared_ptr< typename FunctionType::LocalfunctionType > > LocalfunctionTupleType;
  static const size_t                                                               dimDomain = FunctionType::dimDomain;

  typedef typename CellProblemType::ScalarFct               FineFunctionType;
  typedef typename CellProblemType::PeriodicViewType        FineGridViewType;
  typedef typename CellProblemType::CellSolutionStorageType CellSolutionsStorageType;
}; //class HMMCurlcurlTraits

template< class FunctionType, class CellProblemType >
class HMMIdentityTraits
{
  static_assert(Stuff::is_localizable_function< FunctionType >::value,
                "FunctionType has to be a localizable function!");
public:
  typedef HMMIdentity< FunctionType, CellProblemType >                              derived_type;
  typedef typename FunctionType::EntityType                                         EntityType;
  typedef typename FunctionType::DomainFieldType                                    DomainFieldType;
  typedef std::tuple< std::shared_ptr< typename FunctionType::LocalfunctionType >,
                      std::shared_ptr< typename FunctionType::LocalfunctionType > > LocalfunctionTupleType;
  static const size_t                                                               dimDomain = FunctionType::dimDomain;

  typedef typename CellProblemType::ScalarFct               FineFunctionType;
  typedef typename CellProblemType::PeriodicViewType        FineGridViewType;
  typedef typename CellProblemType::CellSolutionStorageType CellSolutionsStorageType;
}; //class HMMIdentityTraits


}//namespace internal


/** \brief Class to compute a local (w.r.t. the macroscopic grid) evaluation of the curl-curl part in the HMM
 *
 * \tparan FunctionImp The macroscopic type of parameter functions
 * \tparam CellProblemType The type of cell reconstruction to use for the correctors
 * \note This class solves the cell problem separately for each quadrature point. If you have a periodic or separable parameter
 * dependence you shoud use the class \ref HMMCurlcurlPeriodic
 */
template< class FunctionImp, class CellProblemType >
class HMMCurlcurl
  : public LocalEvaluation::Codim0Interface< internal::HMMCurlcurlTraits< FunctionImp, CellProblemType >, 2 >
{
public:
  typedef internal::HMMCurlcurlTraits< FunctionImp, CellProblemType > Traits;
  typedef FunctionImp                             FunctionType;
  typedef typename Traits::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename Traits::EntityType             EntityType;
  typedef typename Traits::DomainFieldType        DomainFieldType;
  static const size_t                             dimDomain = Traits::dimDomain;

  typedef typename Traits::FineFunctionType         FineFunctionType;
  typedef typename Traits::FineGridViewType         FineGridViewType;
  typedef typename Traits::CellSolutionsStorageType CellSolutionsStorageType;

  explicit HMMCurlcurl(const CellProblemType& cell_problem,
                       const FineFunctionType& periodic_mu,
                       const FineFunctionType& periodic_divparam,
                       const FunctionType& macro_mu)
    : cell_problem_(cell_problem)
    , periodic_mu_(periodic_mu)
    , periodic_divparam_(periodic_divparam)
    , macro_mu_(macro_mu)
  {}

  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(macro_mu_.local_function(entity));
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
    * \brief Computes the HMM curl-curl evaluation
    * \note Contra-intuitively, we first iterate over the microscopic enities and then over the rows and columns (base size of the macroscopic space),
    * but in most applications the number of entities is large in comparison to the size of the macroscopic space, so this is much faster
    * \tparam R RangeFieldType
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunction,
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
    //clear return matrix
    ret *= 0.;
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
    //prepare cell solutions
    assert(rows==cols);
    CellSolutionsStorageType cell_solutions(rows);
    for (auto& it : cell_solutions) {
      if (!cell_problem_.is_complex()){
        std::vector<DiscreteFunction< typename CellProblemType::CellSpaceType, typename CellProblemType::RealVectorType > >
                it1(1, DiscreteFunction< typename CellProblemType::CellSpaceType, typename CellProblemType::RealVectorType >(cell_problem_.cell_space()));
        it = DSC::make_unique< typename CellProblemType::CellDiscreteFunctionType >(it1);
      }
      else {
        typename CellProblemType::CellDiscreteFunctionType
                  it1(2, DiscreteFunction< typename CellProblemType::CellSpaceType, typename CellProblemType::RealVectorType >(cell_problem_.cell_space()));
        it = DSC::make_unique< typename CellProblemType::CellDiscreteFunctionType >(it1);
      }
    }
    //solve cell reconstructions
    cell_problem_.solve_all_at_single_point(testBase.entity(), cell_solutions, localPoint);
    auto cube_grid_view = cell_solutions[0]->operator[](0).space().grid_view();
    // perpare ansatz test curls
    for (size_t jj = 0; jj<cols; ++jj) {
      ansatzcurl[jj][0] = aGrad[jj][2][1]-aGrad[jj][1][2];
      ansatzcurl[jj][1] = aGrad[jj][0][2]-aGrad[jj][2][0];
      ansatzcurl[jj][2] = aGrad[jj][1][0]-aGrad[jj][0][1];
    }
    //prepare test curls
    for (size_t ii = 0; ii<rows; ++ii) {
      testcurl[ii][0] = tGrad[ii][2][1]-tGrad[ii][1][2];
      testcurl[ii][1] = tGrad[ii][0][2]-tGrad[ii][2][0];
      testcurl[ii][2] = tGrad[ii][1][0]-tGrad[ii][0][1];
    }
    auto macro_function_value = localFunction.evaluate(localPoint);
    //integrate over unit cube
    for (const auto& entity : DSC::entityRange(cube_grid_view) ) {
      const auto local_mu = periodic_mu_.local_function(entity);
      const auto local_divparam = periodic_divparam_.local_function(entity);
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const size_t integrand_order = local_mu->order() + 2 * (cell_solutions[0]->operator[](0).local_function(entity)->order() - 1);
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(integrand_order));
      // evaluate the jacobians of all local solutions in all quadrature points
      std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations(
         cell_solutions.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
      for (auto lsNum : DSC::valueRange(cell_solutions.size())) {
        const auto local_cell_function = cell_solutions[lsNum]->operator[](0).local_function(entity);
        local_cell_function->jacobian(volumeQuadrature, allLocalSolutionEvaluations[lsNum]);
      }
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      size_t kk = 0;
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++kk) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        for (size_t ii = 0; ii < rows; ++ii) {
          auto& retRow = ret[ii];
          for (size_t jj = 0; jj < cols; ++jj) {
            auto tmp_result = (macro_function_value * local_mu->evaluate(x));
            tmp_result *= ((ansatzcurl[jj][0] + allLocalSolutionEvaluations[jj][kk][2][1] - allLocalSolutionEvaluations[jj][kk][1][2])
                              * (testcurl[ii][0] + allLocalSolutionEvaluations[ii][kk][2][1] - allLocalSolutionEvaluations[ii][kk][1][2])
                            + (ansatzcurl[jj][1] + allLocalSolutionEvaluations[jj][kk][0][2] - allLocalSolutionEvaluations[jj][kk][2][0])
                              * (testcurl[ii][1] + allLocalSolutionEvaluations[ii][kk][0][2] - allLocalSolutionEvaluations[ii][kk][2][0])
                            + (ansatzcurl[jj][2] + allLocalSolutionEvaluations[jj][kk][1][0] - allLocalSolutionEvaluations[jj][kk][0][1])
                              * (testcurl[ii][2] + allLocalSolutionEvaluations[ii][kk][1][0] - allLocalSolutionEvaluations[ii][kk][0][1]));
            tmp_result += local_divparam->evaluate(x) * (allLocalSolutionEvaluations[jj][kk][0][0] * allLocalSolutionEvaluations[ii][kk][0][0]
                                                         + allLocalSolutionEvaluations[jj][kk][1][1] * allLocalSolutionEvaluations[ii][kk][1][1]
                                                         + allLocalSolutionEvaluations[jj][kk][2][2] * allLocalSolutionEvaluations[ii][kk][2][2]);
            tmp_result *= (quadrature_weight * integration_factor);
            retRow[jj] += tmp_result;
          } //loop over cols
        } //loop over rows
      } //loop over quadrature points
    } //loop over cube entities
  } // ... evaluate (...)

protected:
  const CellProblemType&  cell_problem_;
  const FineFunctionType& periodic_mu_;
  const FineFunctionType& periodic_divparam_;
  const FunctionType&     macro_mu_;
}; //class HMMCurlcurl


/** \brief Class to compute a local (w.r.t. the macroscopic grid) evaluation of the curl-curl part in the HMM for periodic problems
 *
 * \tparan FunctionImp The macroscopic type of parameter functions
 * \tparam CellProblemType The type of cell reconstruction to use for the correctors
 * \sa HMMCurlcurl
 */
template< class FunctionImp, class CellProblemType >
class HMMCurlcurlPeriodic
  : public HMMCurlcurl< FunctionImp, CellProblemType >
{
public:
  typedef HMMCurlcurl< FunctionImp, CellProblemType > BaseType;
  using typename BaseType::FunctionType;
  using typename BaseType::EntityType;
  using typename BaseType::DomainFieldType;
  using BaseType::dimDomain;
  using typename BaseType::FineFunctionType;
  using typename BaseType::FineGridViewType;
  using typename BaseType::CellSolutionsStorageType;

  using BaseType::periodic_mu_;
  using BaseType::periodic_divparam_;

  explicit HMMCurlcurlPeriodic(const CellProblemType& cell_problem,
                               const FineFunctionType& periodic_mu,
                               const FineFunctionType& periodic_divparam,
                               const FunctionType& macro_mu,
                               const CellSolutionsStorageType& cell_solutions)
    : BaseType(cell_problem, periodic_mu, periodic_divparam, macro_mu)
    , cell_solutions_(cell_solutions)
  {}

  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
    * \brief Computes the HMM curl-curl evaluation for periodic problems
    * \note Contra-intuitively, we first iterate over the microscopic enities and then over the rows and columns (base size of the macroscopic space),
    * but in most applications the number of entities is large in comparison to the size of the macroscopic space, so this is much faster
    * \tparam R RangeFieldType
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunction,
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
    //clear return matrix
    ret *= 0.;
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
    assert(cell_solutions_.size()==dimDomain);
    auto cube_grid_view = cell_solutions_[0]->operator[](0).space().grid_view();
    // perpare ansatz test curls
    for (size_t jj = 0; jj<cols; ++jj) {
      ansatzcurl[jj][0] = aGrad[jj][2][1]-aGrad[jj][1][2];
      ansatzcurl[jj][1] = aGrad[jj][0][2]-aGrad[jj][2][0];
      ansatzcurl[jj][2] = aGrad[jj][1][0]-aGrad[jj][0][1];
    }
    //prepare test curls
    for (size_t ii = 0; ii<rows; ++ii) {
      testcurl[ii][0] = tGrad[ii][2][1]-tGrad[ii][1][2];
      testcurl[ii][1] = tGrad[ii][0][2]-tGrad[ii][2][0];
      testcurl[ii][2] = tGrad[ii][1][0]-tGrad[ii][0][1];
    }
    auto macro_function_value = localFunction.evaluate(localPoint);
    //integrate over unit cube
    for (const auto& entity : DSC::entityRange(cube_grid_view) ) {
      const auto local_mu = BaseType::periodic_mu_.local_function(entity);
      const auto local_divparam = BaseType::periodic_divparam_.local_function(entity);
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const size_t integrand_order = local_mu->order() + 2 * (cell_solutions_[0]->operator[](0).local_function(entity)->order() - 1);
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(integrand_order));
      // evaluate the jacobians of all local solutions in all quadrature points
      std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations(
         cell_solutions_.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
      for (auto lsNum : DSC::valueRange(cell_solutions_.size())) {
        const auto local_cell_function = cell_solutions_[lsNum]->operator[](0).local_function(entity);
        local_cell_function->jacobian(volumeQuadrature, allLocalSolutionEvaluations[lsNum]);
      }
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      size_t kk = 0;
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++kk) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        for (size_t ii = 0; ii < rows; ++ii) {
          auto& retRow = ret[ii];
          for (size_t jj = 0; jj < cols; ++jj) {
            auto tmp_result = (macro_function_value * local_mu->evaluate(x));
            auto allCellCorrecJacobii = allLocalSolutionEvaluations[ii][kk];
            auto allCellCorrecJacobjj = allLocalSolutionEvaluations[jj][kk];
            allCellCorrecJacobii *= 0.;
            allCellCorrecJacobjj *= 0.;
            for (size_t ll = 0; ll < dimDomain; ++ll){
              allCellCorrecJacobii.axpy(testcurl[ii][ll], allLocalSolutionEvaluations[ll][kk]);
              allCellCorrecJacobjj.axpy(ansatzcurl[jj][ll], allLocalSolutionEvaluations[ll][kk]);
            }
            tmp_result *= ((ansatzcurl[jj][0] + allCellCorrecJacobjj[2][1] - allCellCorrecJacobjj[1][2])
                              * (testcurl[ii][0] + allCellCorrecJacobii[2][1] - allCellCorrecJacobii[1][2])
                            + (ansatzcurl[jj][1] + allCellCorrecJacobjj[0][2] - allCellCorrecJacobjj[2][0])
                              * (testcurl[ii][1] + allCellCorrecJacobii[0][2] - allCellCorrecJacobii[2][0])
                            + (ansatzcurl[jj][2] + allCellCorrecJacobjj[1][0] - allCellCorrecJacobjj[0][1])
                              * (testcurl[ii][2] + allLocalSolutionEvaluations[ii][kk][1][0] - allLocalSolutionEvaluations[ii][kk][0][1]));
            tmp_result += local_divparam->evaluate(x) * (allCellCorrecJacobjj[0][0] * allCellCorrecJacobii[0][0]
                                                         + allCellCorrecJacobjj[1][1] * allCellCorrecJacobii[1][1]
                                                         + allCellCorrecJacobjj[2][2] * allCellCorrecJacobjj[2][2]);
            tmp_result *= (quadrature_weight * integration_factor);
            retRow[jj] += tmp_result;
          } //loop over cols
        } //loop over rows
      } //loop over quadrature points
    } //loop over cube entities
  } // ... evaluate (...)

  using BaseType::evaluate;

private:
  const CellSolutionsStorageType& cell_solutions_;
}; //class HMMCurlcurlPeriodic


/** \brief Class to compute a local evaluation (w.r.t the macroscopic grid) of the identity part of the HMM for curl-curl-problems
 *
 * \tparam FunctionImp Type of the macroscopic parameter function
 * \tparam CellProblemType Type of cell reconstruction for the correctors
 * \note This class solves the cell problem separately for each quadrature point. If you have aperiodic or separable parameter
 * dependence, you should use the class \ref HMMIdentityPeriodic
 */
template< class FunctionImp, class CellProblemType >
class HMMIdentity
  : public LocalEvaluation::Codim0Interface< internal::HMMIdentityTraits< FunctionImp, CellProblemType >, 2 >
{
public:
  typedef internal::HMMIdentityTraits< FunctionImp, CellProblemType > Traits;
  typedef FunctionImp                             FunctionType;
  typedef typename Traits::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename Traits::EntityType             EntityType;
  typedef typename Traits::DomainFieldType        DomainFieldType;
  static const size_t                             dimDomain = Traits::dimDomain;

  typedef typename Traits::FineFunctionType         FineFunctionType;
  typedef typename Traits::FineGridViewType         FineGridViewType;
  typedef typename Traits::CellSolutionsStorageType CellSolutionsStorageType;

  explicit HMMIdentity(const CellProblemType& cell_problem,
                       const FineFunctionType& periodic_kappa_real,
                       const FineFunctionType& periodic_kappa_imag,
                       const bool real_part,
                       const FunctionType& macro_kappa_real,
                       const FunctionType& macro_kappa_imag)
    : cell_problem_(cell_problem)
    , periodic_kappa_real_(periodic_kappa_real)
    , periodic_kappa_imag_(periodic_kappa_imag)
    , real_part_(real_part)
    , macro_kappa_real_(macro_kappa_real)
    , macro_kappa_imag_(macro_kappa_imag)
  {}

  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(macro_kappa_real_.local_function(entity), macro_kappa_imag_.local_function(entity));
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
    return order(*std::get< 0 >(localFuncs), *std::get< 1 >(localFuncs), testBase, ansatzBase);
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
    evaluate(*std::get< 0 >(localFuncs), *std::get< 1 >(localFuncs), testBase, ansatzBase, localPoint, ret);
  }

  /// \}
  /// \name Actual implmentation of order
  /// \{

  /**
    * \return localFunction.order()+(testBase.order()-1)+(ansatzBase.order()-1)
    */
  template< class R, size_t rL, size_t rCl, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rL, rCl >& localFunctionreal,
               const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, rL, rCl >& localFunctionimag,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase)
  const
  {
    return boost::numeric_cast< size_t >(std::max(ssize_t(localFunctionreal.order()), ssize_t(localFunctionimag.order())))
     + boost::numeric_cast< size_t >(std::max(ssize_t(testBase.order()) -1, ssize_t(0)))
     + boost::numeric_cast< size_t >(std::max(ssize_t(ansatzBase.order()) - 1, ssize_t(0)));
  } // ...order(....)


  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
    * \brief Computes the evaluation for the identity part of the HMM for curl-curl-problems
    * \note Contra-intuitively, we first iterate over the microscopic enities and then over the rows and columns (base size of the macroscopic space),
    * but in most applications the number of entities is large in comparison to the size of the macroscopic space, so this is much faster
    * \tparam R RangeFieldType
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunctionreal,
                const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunctionimag,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeType         RangeType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeFieldType    RangeFieldType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, 1, 1 >::JacobianRangeType JacobianRangeType;
    //clear return matrix
    ret *= 0.;
    //evaluate test functions
    const size_t rows = testBase.size();
    auto tValue = testBase.evaluate(localPoint);
    //evaluate ansatz functions
    const size_t cols = ansatzBase.size();
    auto aValue = ansatzBase.evaluate(localPoint);
    assert(ret.rows()>= rows);
    assert(ret.cols()>= cols);
    //prepare cell solutions
    assert(rows==cols);
    CellSolutionsStorageType cell_solutions(rows);
    for (auto& it : cell_solutions) {
      if (!cell_problem_.is_complex()){
        std::vector<DiscreteFunction< typename CellProblemType::CellSpaceType, typename CellProblemType::RealVectorType > >
                it1(1, DiscreteFunction< typename CellProblemType::CellSpaceType, typename CellProblemType::RealVectorType >(cell_problem_.cell_space()));
        it = DSC::make_unique< typename CellProblemType::CellDiscreteFunctionType >(it1);
      }
      else {
        typename CellProblemType::CellDiscreteFunctionType
                  it1(2, DiscreteFunction< typename CellProblemType::CellSpaceType, typename CellProblemType::RealVectorType >(cell_problem_.cell_space()));
        it = DSC::make_unique< typename CellProblemType::CellDiscreteFunctionType >(it1);
      }
    }
    //solve cell reconstructions
    cell_problem_.solve_all_at_single_point(testBase.entity(), cell_solutions, localPoint);
    auto cube_grid_view = cell_solutions[0]->operator[](0).space().grid_view();
    auto macro_real_value = localFunctionreal.evaluate(localPoint);
    auto macro_imag_value = localFunctionimag.evaluate(localPoint);
    //integrate over unit cube
    for (const auto& entity : DSC::entityRange(cube_grid_view) ) {
      const auto local_kappa_real = periodic_kappa_real_.local_function(entity);
      const auto local_kappa_imag = periodic_kappa_imag_.local_function(entity);
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const size_t integrand_order = local_kappa_real->order() + 2 * (cell_solutions[0]->operator[](0).local_function(entity)->order() - 1);
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(integrand_order));
      // evaluate the jacobians of all local solutions in all quadrature points
      std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations_real(
          cell_solutions.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
      std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations_imag(
          cell_solutions.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
      for (auto lsNum : DSC::valueRange(cell_solutions.size())) {
        const auto localFunction_real = cell_solutions[lsNum]->operator[](0).local_function(entity);
        const auto localFunction_imag = cell_solutions[lsNum]->operator[](1).local_function(entity);
        localFunction_real->jacobian(volumeQuadrature, allLocalSolutionEvaluations_real[lsNum]);
        localFunction_imag->jacobian(volumeQuadrature, allLocalSolutionEvaluations_imag[lsNum]);
      }
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      size_t kk = 0;
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++kk) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        auto value_real = (macro_real_value * local_kappa_real->evaluate(x));
        auto value_imag = (macro_imag_value * local_kappa_imag->evaluate(x));
        for (size_t ii = 0; ii<rows; ++ii) {
          auto& retRow = ret[ii];
          for (size_t jj = 0; jj< cols; ++jj) {
            auto reconii_real = tValue[ii] + allLocalSolutionEvaluations_real[ii][kk][0];
            auto reconii_imag = allLocalSolutionEvaluations_imag[ii][kk][0];
            auto reconjj_real = aValue[jj] + allLocalSolutionEvaluations_real[jj][kk][0];
            auto reconjj_imag = allLocalSolutionEvaluations_imag[jj][kk][0];
            if (real_part_) {
              auto tmp_result = value_real * (reconjj_real * reconii_real);
              tmp_result += value_real * (reconjj_imag * reconii_imag);
              tmp_result += value_imag * (reconjj_real * reconii_imag);
              tmp_result -= value_imag * (reconjj_imag * reconii_real);
              tmp_result *= (quadrature_weight * integration_factor);
              retRow[jj] += tmp_result;
            }
            else {
              auto tmp_result = value_imag * (reconjj_imag * reconii_imag);
              tmp_result += value_imag * (reconjj_real * reconii_real);
              tmp_result += value_real * (reconjj_imag * reconii_real);
              tmp_result -= value_real * (reconjj_real * reconii_imag);
              tmp_result *= (quadrature_weight * integration_factor);
              retRow[jj] += tmp_result;
            }
          } //loop over cols
        } //loop over rows
      } //loop over micro quadrature points
    } //loop over entities
  } // ... evaluate (...)

protected:
  const CellProblemType&  cell_problem_;
  const FineFunctionType& periodic_kappa_real_;
  const FineFunctionType& periodic_kappa_imag_;
  const bool              real_part_;
  const FunctionType&     macro_kappa_real_;
  const FunctionType&     macro_kappa_imag_;
}; //class HMMIdentity


/** \brief Class to compute a local evaluation (w.r.t the macroscopic grid) of the identity part of the HMM for periodic curl-curl-problems
 *
 * \tparam FunctionImp Type of the macroscopic parameter function
 * \tparam CellProblemType Type of cell reconstruction for the correctors
 * \sa HMMIdentity
 */
template< class FunctionImp, class CellProblemType >
class HMMIdentityPeriodic
  : public LocalEvaluation::HMMIdentity< FunctionImp, CellProblemType >
{
public:
  typedef HMMIdentity< FunctionImp, CellProblemType > BaseType;
  using typename BaseType::FunctionType;
  using typename BaseType::EntityType;
  using typename BaseType::DomainFieldType;
  using BaseType::dimDomain;
  using typename BaseType::FineFunctionType;
  using typename BaseType::FineGridViewType;
  using typename BaseType::CellSolutionsStorageType;

  using BaseType::periodic_kappa_real_;
  using BaseType::periodic_kappa_imag_;
  using BaseType::real_part_;

  explicit HMMIdentityPeriodic(const CellProblemType& cell_problem,
                               const FineFunctionType& periodic_kappa_real,
                               const FineFunctionType& periodic_kappa_imag,
                               const bool real_part,
                               const FunctionType& macro_kappa_real,
                               const FunctionType& macro_kappa_imag,
                               const CellSolutionsStorageType& cell_solutions)
    : BaseType(cell_problem, periodic_kappa_real, periodic_kappa_imag, real_part, macro_kappa_real, macro_kappa_imag)
    , cell_solutions_(cell_solutions)
  {}

  /// \}
  /// \name Actual implementation of evaluate
  /// \{

  /**
    * \brief Computes the evaluation for the identity part of the HMM for periodic curl-curl-problems
    * \note Contra-intuitively, we first iterate over the microscopic enities and then over the rows and columns (base size of the macroscopic space),
    * but in most applications the number of entities is large in comparison to the size of the macroscopic space, so this is much faster
    * \tparam R RangeFieldType
    */

  template< class R, size_t r >
  void evaluate(const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunctionreal,
                const Stuff::LocalfunctionInterface< EntityType, DomainFieldType, dimDomain, R, 1, 1 >& localFunctionimag,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, r, 1 >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeType         RangeType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, r, 1 >::RangeFieldType    RangeFieldType;
    typedef typename Stuff::LocalfunctionSetInterface
        < EntityType, DomainFieldType, dimDomain, R, 1, 1 >::JacobianRangeType JacobianRangeType;
    //clear return matrix
    ret *= 0.;
    //evaluate test functions
    const size_t rows = testBase.size();
    auto tValue = testBase.evaluate(localPoint);
    //evaluate ansatz functions
    const size_t cols = ansatzBase.size();
    auto aValue = ansatzBase.evaluate(localPoint);
    assert(ret.rows()>= rows);
    assert(ret.cols()>= cols);
    assert(cell_solutions_.size()==dimDomain);
    auto cube_grid_view = cell_solutions_[0]->operator[](0).space().grid_view();
    auto macro_real_value = localFunctionreal.evaluate(localPoint);
    auto macro_imag_value = localFunctionimag.evaluate(localPoint);
    //integrate over unit cube
    for (const auto& entity : DSC::entityRange(cube_grid_view) ) {
      const auto local_kappa_real = periodic_kappa_real_.local_function(entity);
      const auto local_kappa_imag = periodic_kappa_imag_.local_function(entity);
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const size_t integrand_order = local_kappa_real->order() + 2 * (cell_solutions_[0]->operator[](0).local_function(entity)->order() - 1);
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(integrand_order));
      // evaluate the jacobians of all local solutions in all quadrature points
      std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations_real(
          cell_solutions_.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
      std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations_imag(
          cell_solutions_.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
      for (auto lsNum : DSC::valueRange(cell_solutions_.size())) {
        const auto localFunction_real = cell_solutions_[lsNum]->operator[](0).local_function(entity);
        const auto localFunction_imag = cell_solutions_[lsNum]->operator[](1).local_function(entity);
        localFunction_real->jacobian(volumeQuadrature, allLocalSolutionEvaluations_real[lsNum]);
        localFunction_imag->jacobian(volumeQuadrature, allLocalSolutionEvaluations_imag[lsNum]);
      }
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      size_t kk = 0;
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++kk) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        auto value_real = (macro_real_value * local_kappa_real->evaluate(x));
        auto value_imag = (macro_imag_value * local_kappa_imag->evaluate(x));
        for (size_t ii = 0; ii<rows; ++ii) {
          auto& retRow = ret[ii];
          for (size_t jj = 0; jj< cols; ++jj) {
            auto reconii_real = tValue[ii];
            auto reconii_imag = reconii_real;
            reconii_imag *= 0.;
            auto reconjj_real = aValue[jj];
            auto reconjj_imag = reconjj_real;
            reconjj_imag *= 0.;
            for (size_t ll = 0; ll< dimDomain; ++ll){
              reconii_real.axpy(tValue[ii][ll], allLocalSolutionEvaluations_real[ll][kk][0]);
              reconii_imag.axpy(tValue[ii][ll], allLocalSolutionEvaluations_imag[ll][kk][0]);
              reconjj_real.axpy(aValue[jj][ll], allLocalSolutionEvaluations_real[ll][kk][0]);
              reconjj_imag.axpy(aValue[jj][ll], allLocalSolutionEvaluations_imag[ll][kk][0]);
            }
            if (real_part_) {
              auto tmp_result = value_real * (reconjj_real * reconii_real);
              tmp_result += value_real * (reconjj_imag * reconii_imag);
              tmp_result += value_imag * (reconjj_real * reconii_imag);
              tmp_result -= value_imag * (reconjj_imag * reconii_real);
              tmp_result *= (quadrature_weight * integration_factor);
              retRow[jj] += tmp_result;
            }
            else {
              auto tmp_result = value_imag * (reconjj_imag * reconii_imag);
              tmp_result += value_imag * (reconjj_real * reconii_real);
              tmp_result += value_real * (reconjj_imag * reconii_real);
              tmp_result -= value_real * (reconjj_real * reconii_imag);
              tmp_result *= (quadrature_weight * integration_factor);
              retRow[jj] += tmp_result;
            }
          } //loop over cols
        } //loop over rows
      } //loop over micro quadrature points
    } //loop over entities
  } // ... evaluate (...)

  using BaseType::evaluate;

private:
  const CellSolutionsStorageType& cell_solutions_;
}; //class HMMIdentityPeriodic

}//namespace LocalEvaluation
}//namespace GDT
}//namespace Dune

#endif // DUNE_GDT_EVALUATION_HMM_HH
