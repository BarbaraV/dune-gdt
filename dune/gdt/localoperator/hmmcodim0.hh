// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_LOCALOPERATOR_HMMCODIM0_HH
#define DUNE_GDT_LOCALOPERATOR_HMMCODIM0_HH

#include <vector>
#include <utility>
#include <type_traits>
#include <limits>

#include <boost/numeric/conversion/cast.hpp>

#include <dune/common/densematrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/stuff/functions/interfaces.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalOperator {


//forward to be used in the traits
template< class HMMEvaluation, class CoarseGridView >
class HMMCodim0Integral;


namespace internal {


template< class HMMEvaluation, class CoarseGridView >
class HMMCodim0IntegralTraits
{
public:
  typedef HMMCodim0Integral< HMMEvaluation, CoarseGridView > derived_type;
};


} //namespace internal


template< class HMMEvaluation, class CoarseGridView >
class HMMCodim0Integral
  : public LocalOperator::Codim0Interface< internal::HMMCodim0IntegralTraits< HMMEvaluation, CoarseGridView > >
{
public:
  typedef internal::HMMCodim0IntegralTraits< HMMEvaluation, CoarseGridView > Traits;
  typedef HMMEvaluation                                                      HMMEvaluationType;
  typedef CoarseGridView                                                     GridViewType;

private:
  static const size_t numTmpObjectsRequired_ = 1;

public:
  template< class... Args >
  explicit HMMCodim0Integral(const GridViewType& grid_view, Args&& ...args)
    : evaluation_(std::forward< Args >(args)...)
    , grid_view_(grid_view)
    , over_integrate_(0)
  {}

  template< class... Args >
  explicit HMMCodim0Integral(const int over_integrate, const GridViewType& grid_view, Args&& ...args)
    : evaluation_(std::forward< Args >(args)...)
    , grid_view_(grid_view)
    , over_integrate_(boost::numeric_cast< size_t >(over_integrate))
  {}

  template< class... Args >
  explicit HMMCodim0Integral(const size_t over_integrate, const GridViewType& grid_view, Args&& ...args)
    : evaluation_(std::forward< Args >(args)...)
    , grid_view_(grid_view)
    , over_integrate_(over_integrate)
  {}

  size_t numTmpObjectsRequired() const
  {
    return numTmpObjectsRequired_;
  }

  template< class E, class D, size_t d, class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  void apply(const Stuff::LocalfunctionSetInterface< E, D, d, R, rT, rCT >& testBase,
             const Stuff::LocalfunctionSetInterface< E, D, d, R, rA, rCA >& ansatzBase,
             Dune::DynamicMatrix< R >& ret,
             std::vector< Dune::DynamicMatrix< R > >& tmpLocalMatrices) const
  {
    const auto& entity = ansatzBase.entity();
    size_t entity_index = grid_view_.indexSet().index(entity);
    const auto localFunctions = evaluation_.localFunctions(entity);
    // quadrature
    typedef Dune::QuadratureRules< D, d > VolumeQuadratureRules;
    typedef Dune::QuadratureRule< D, d > VolumeQuadratureType;
    const size_t integrand_order = evaluation_.order(localFunctions, ansatzBase, testBase) + over_integrate_;
    const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(),
                                                                               boost::numeric_cast< int >(integrand_order));
    // check matrix and tmp storage
    const size_t rows = testBase.size();
    const size_t cols = ansatzBase.size();
    ret *= 0.0;
    assert(ret.rows() >= rows);
    assert(ret.cols() >= cols);
    assert(tmpLocalMatrices.size() >= numTmpObjectsRequired_);
    auto& evaluationResult = tmpLocalMatrices[0];
    // loop over all quadrature points
    const auto quadPointEndIt = volumeQuadrature.end();
    size_t kk = 0;
    for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++kk) {
      const Dune::FieldVector< D, d > x = quadPointIt->position();
      // integration factors
      const double integrationFactor = entity.geometry().integrationElement(x);
      const double quadratureWeight = quadPointIt->weight();
      // evaluate the local operation
      auto key = std::make_pair(entity_index, kk);
      evaluation_.evaluate(localFunctions, ansatzBase, testBase, x, key, evaluationResult);
      // compute integral
      for (size_t ii = 0; ii < rows; ++ii) {
        auto& retRow = ret[ii];
        const auto& evaluationResultRow = evaluationResult[ii];
        for (size_t jj = 0; jj < cols; ++jj)
          retRow[jj] += evaluationResultRow[jj] * integrationFactor * quadratureWeight;
      } // compute integral
    } // loop over all quadrature points
  } // ... apply(...)

private:
  const HMMEvaluation evaluation_;
  const GridViewType& grid_view_;
  const size_t over_integrate_;
}; //class HMMCodim0Integral

} //namespace LocalOperator
} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_LOCALOPERATOR_HMMCODIM0_HH
