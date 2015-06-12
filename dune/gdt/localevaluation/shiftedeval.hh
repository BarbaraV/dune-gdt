// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_EVALUATION_SHIFTEDEVAL_HH
#define DUNE_GDT_EVALUATION_SHIFTEDEVAL_HH

#include <type_traits>

#include <dune/common/typetraits.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/stuff/functions/interfaces.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalEvaluation {


//forward
template< class LocalEvaluationImp, class Coordtypeext >
class ShiftedEval;


namespace internal {


/**
  * \brief Traits for the shifted evaluation
  */
template< class LocalEvaluationImp, class Coordtypeext >
class ShiftedEvalTraits
{
public:
  typedef LocalEvaluationImp                                   LocalEvaluationType;
  typedef Coordtypeext                                         givenCoordType;
  typedef ShiftedEval< LocalEvaluationType, givenCoordType >   derived_type;
  typedef typename LocalEvaluationType::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename LocalEvaluationType::EntityType             EntityType;
  typedef typename LocalEvaluationType::DomainFieldType        DomainFieldType;
  static const size_t                                          dimDomain = LocalEvaluationType::dimDomain;
  static_assert(std::is_same< givenCoordType, Dune::FieldVector< DomainFieldType, dimDomain > >::value,
                "The external coordinate type must be a FieldVector of right dimension!");
}; //class ShiftedEvalTraits


} //namespace internal


/**
  * \brief computes a shifted local evaluation
  *
  * the given evaluation is shifted by external coordinates, as in the case of the HMM.
  */
template< class LocalEvaluationImp, class Coordtypeext >
class ShiftedEval
  : public LocalEvaluation::Codim0Interface< internal::ShiftedEvalTraits< LocalEvaluationImp, Coordtypeext >, 1 >
  , public LocalEvaluation::Codim0Interface< internal::ShiftedEvalTraits< LocalEvaluationImp, Coordtypeext >, 2 >
{
public:
  typedef internal::ShiftedEvalTraits< LocalEvaluationImp, Coordtypeext > Traits;
  typedef typename Traits::LocalEvaluationType                            LocalEvaluationType;
  typedef typename Traits::givenCoordType                                 givenCoordType;
  typedef typename Traits::LocalfunctionTupleType                         LocalfunctionTupleType;
  typedef typename Traits::EntityType                                     EntityType;
  typedef typename Traits::DomainFieldType                                DomainFieldType;
  static const size_t                                                     dimDomain = Traits::dimDomain;
  static_assert(std::is_same< givenCoordType, Dune::FieldVector< DomainFieldType, dimDomain > >::value, "");

  ShiftedEval(const LocalEvaluationType& localeval,
              const DomainFieldType delta,
              const givenCoordType xt)
    : evaluation_(localeval)
    , shiftfactor_(delta)
    , localshift_(xt)
  {}

  /// \name Required by all variants of LocalEvaluation::Codim0Interface
  /// \{

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return evaluation_.localFunctions(entity);
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
    return evaluation_.order(localFuncs, testBase);
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
    auto tmp_point = localPoint;
    evaluation_.evaluate(localFuncs, testBase, tmp_point.axpy(shiftfactor_, localshift_), ret);
  }

  /// \}
  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  /**
   * \brief extracts the local function and calls the correct evaluate() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  void evaluate(const LocalfunctionTupleType& localFuncs,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
                const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase,
                const Dune::FieldVector< DomainFieldType, dimDomain >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    auto tmp_point = localPoint;
    evaluation_.evaluate(localFuncs, testBase, ansatzBase, tmp_point.axpy(shiftfactor_, localshift_), ret);
  }

  /// \}
  /// \name Required by LocalEvaluation::Codim0Interface< ..., 2 >
  /// \{

  /**
   * \brief extracts the local function and calls the correct order() method
   */
  template< class R, size_t rT, size_t rCT, size_t rA, size_t rCA >
  size_t order(const LocalfunctionTupleType& localFuncs,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rT, rCT >& testBase,
               const Stuff::LocalfunctionSetInterface< EntityType, DomainFieldType, dimDomain, R, rA, rCA >& ansatzBase)
  const
  {
    return evaluation_.order(localFuncs, testBase, ansatzBase);
  }

  /// \}


private:
  const LocalEvaluationType& evaluation_;
  const DomainFieldType      shiftfactor_;
  const givenCoordType       localshift_;
}; //class ShiftedEval


} //namespace LocalEvaluation
} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_EVALUATION_SHIFTEDEVAL_HH
