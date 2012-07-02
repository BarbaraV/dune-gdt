/**
  \file   integration.hh
  **/

#ifndef DUNE_DETAILED_DISCRETIZATIONS_DISCRETEFUNCTIONAL_LOCAL_CODIM0_INTEGRAL_HH
#define DUNE_DETAILED_DISCRETIZATIONS_DISCRETEFUNCTIONAL_LOCAL_CODIM0_INTEGRAL_HH

// dune-common includes
#include <dune/common/dynmatrix.hh>

// dune geometry includes
#include <dune/geometry/quadraturerules.hh>

// dune-helper-tools includes
#include <dune/helper-tools/common/vector.hh>

// dune-detailed-discretizations includes
#include <dune/detailed-discretizations/basefunctionset/local/wrapper.hh>

namespace Dune
{

namespace DetailedDiscretizations
{

namespace DiscreteFunctional
{

namespace Local
{

namespace Codim0
{

template< class LocalEvaluationImp >
class Integral
{
public:

  typedef LocalEvaluationImp
    LocalEvaluationType;

  typedef Integral< LocalEvaluationType >
    ThisType;

  typedef typename LocalEvaluationType::FunctionSpaceType
    FunctionSpaceType;

  typedef typename FunctionSpaceType::RangeFieldType
    RangeFieldType;

  typedef typename FunctionSpaceType::DomainType
    DomainType;

  Integral( const LocalEvaluationType localEvaluation )
    : localEvaluation_( localEvaluation )
  {
  }

  //! copy constructor
  Integral( const ThisType& other )
    : localEvaluation_( other.localEvaluation() )
  {
  }

  LocalEvaluationType localEvaluation() const
  {
    return localEvaluation_;
  }

  unsigned int numTmpObjectsRequired() const
  {
    return 1;
  }

  template< class LocalTestBaseFunctionSetType, class LocalVectorType >
  void applyLocal( const LocalTestBaseFunctionSetType localTestBaseFunctionSet,
                   LocalVectorType& localVector,
                   std::vector< LocalVectorType >& tmpLocalVectors ) const
  {
    // some types
    typedef typename LocalTestBaseFunctionSetType::DiscreteFunctionSpaceType
      DiscreteFunctionSpaceType;

    typedef typename DiscreteFunctionSpaceType::GridViewType
      GridViewType;

    typedef Dune::QuadratureRules< double, LocalTestBaseFunctionSetType::GridElementType::mydimension >
      VolumeQuadratureRules;

    typedef Dune::QuadratureRule< double, LocalTestBaseFunctionSetType::GridElementType::mydimension >
      VolumeQuadratureType;

    // some stuff
    const unsigned int size = localTestBaseFunctionSet.size();
    const unsigned int quadratureOrder = localEvaluation_.order() + localTestBaseFunctionSet.order();
    const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule( localTestBaseFunctionSet.gridElement().type(), 2*quadratureOrder+1 );

    // make sure target vector is big enough
    assert( localVector.size() >= size );

    // check tmp local vectors
    if( tmpLocalVectors.size() < 1 )
    {
      tmpLocalVectors.resize( 1,
                              LocalVectorType(  localTestBaseFunctionSet.baseFunctionSet().space().map().maxLocalSize(),
                                                RangeFieldType( 0.0 ) ) );
    }

    // clear target vector
    Dune::HelperTools::Common::Vector::clear( localVector );

    const typename VolumeQuadratureType::const_iterator quadratureEnd = volumeQuadrature.end();
    for (typename VolumeQuadratureType::const_iterator quadPoint = volumeQuadrature.begin(); quadPoint != quadratureEnd; ++quadPoint )
    {
      // local coordinates
      const DomainType x = quadPoint->position();

      // integration factors
      const double integrationFactor = localTestBaseFunctionSet.gridElement().geometry().integrationElement( x );
      const double quadratureWeight = quadPoint->weight();

      // evaluate the local evaluation
      localEvaluation_.evaluateLocal( localTestBaseFunctionSet, x, tmpLocalVectors[0] );

      // compute integral
      for( unsigned int i = 0; i < size; ++i )
      {
        localVector[i] += tmpLocalVectors[0][i] * integrationFactor * quadratureWeight;
      }
    } // done loop over all quadrature points
  } // end method applyLocal

private:

  //! assignment operator
  ThisType& operator=( const ThisType& other );

  const LocalEvaluationType localEvaluation_;

}; // end class Integral

template< class InducingOperatorImp, class InducingDiscreteFunctionImp >
class IntegralInduced
{
public:

  typedef InducingOperatorImp
    InducingOperatorType;

  typedef InducingDiscreteFunctionImp
    InducingDiscreteFunctionType;

  typedef IntegralInduced< InducingOperatorType, InducingDiscreteFunctionType >
    ThisType;

  typedef typename InducingDiscreteFunctionType::RangeFieldType
    RangeFieldType;

  typedef typename InducingDiscreteFunctionType::DomainType
    DomainType;

  IntegralInduced(  const InducingOperatorType& inducingOperator,
                    const InducingDiscreteFunctionType& inducingDiscreteFunction )
    : inducingOperator_( inducingOperator ),
      inducingDiscreteFunction_( inducingDiscreteFunction )
  {
  }

  //! copy constructor
  IntegralInduced( const ThisType& other )
    : inducingOperator_( other.inducingOperator() ),
      inducingDiscreteFunction_( other.inducingDiscreteFunction() )
  {
  }

  InducingOperatorType inducingOperator() const
  {
    return inducingOperator_;
  }

  InducingDiscreteFunctionType inducingDiscreteFunction() const
  {
    return inducingDiscreteFunction_;
  }

  unsigned int numTmpObjectsRequired() const
  {
    return 0;
  }

  template< class LocalTestBaseFunctionSetType, class LocalVectorType >
  void applyLocal( const LocalTestBaseFunctionSetType& localTestBaseFunctionSet,
                   LocalVectorType& localVector,
                   std::vector< LocalVectorType >& ) const
  {
    // some types
    typedef typename LocalTestBaseFunctionSetType::DiscreteFunctionSpaceType
      DiscreteFunctionSpaceType;

    typedef typename DiscreteFunctionSpaceType::GridViewType
      GridViewType;

    typedef Dune::QuadratureRules< double, GridViewType::dimension >
      QuadratureRules;

    typedef Dune::QuadratureRule< double, GridViewType::dimension >
      QuadratureType;

    typedef typename LocalTestBaseFunctionSetType::EntityType
      EntityType;

    typedef typename InducingDiscreteFunctionType::ConstLocalFunctionType
      InducingLocalFunctionType;

    typedef typename Dune::DetailedDiscretizations::BaseFunctionSet::Local::Wrapper< InducingLocalFunctionType >
      InducingBaseFunctionSetType;

    typedef Dune::DynamicMatrix< RangeFieldType >
      LocalMatrixType;

    const EntityType& entity = localTestBaseFunctionSet.entity();

    // wrap inducing local function
    const InducingLocalFunctionType inducingLocalFunction = inducingDiscreteFunction_.localFunction( entity );
    const InducingBaseFunctionSetType inducingBaseFunctionSet( inducingLocalFunction );

    // some stuff
    const unsigned int size = localTestBaseFunctionSet.size();
    const unsigned int quadratureOrder =
      inducingOperator_.localEvaluation().order() + inducingLocalFunction.order() + localTestBaseFunctionSet.order();
    const QuadratureType& volumeQuadrature = QuadratureRules::rule( localTestBaseFunctionSet.gridElement().type(), 2*quadratureOrder+1 );

    // make sure target vector is big enough
    assert( localVector.size() >= size );

    // clear target vector
    Dune::HelperTools::Common::Vector::clear( localVector );

    // do loop over all quadrature points
    LocalMatrixType tmpMatrix( 1, size );
    const typename QuadratureType::const_iterator quadratureEnd = volumeQuadrature.end();
    for (typename QuadratureType::const_iterator quadPoint = volumeQuadrature.begin(); quadPoint != quadratureEnd; ++quadPoint )
    {
      // local coordinates
      const DomainType x = quadPoint->position();

      // integration factors
      const double integrationFactor = entity.geometry().integrationElement( x );
      const double quadratureWeight = quadPoint->weight();

      // evaluate the local evaluation
      inducingOperator_.localEvaluation().evaluateLocal(  inducingBaseFunctionSet,
                                                          localTestBaseFunctionSet,
                                                          x,
                                                          tmpMatrix );

      // compute integral
      for( unsigned int i = 0; i < size; ++i )
      {
        localVector[i] += tmpMatrix[0][i] * integrationFactor * quadratureWeight;
      }
    } // done loop over all quadrature points

  } // end method applyLocal

private:
  //! assignment operator
  ThisType& operator=( const ThisType& );

  const InducingOperatorType inducingOperator_;
  const InducingDiscreteFunctionType inducingDiscreteFunction_;

}; // end class IntegralInduced

} // end namespace Codim0

} // end namespace Local

} // end namespace DiscreteFunctional

} // end namespace DetailedDiscretizations

} // end namespace Dune

#endif // end DUNE_DETAILED_DISCRETIZATIONS_DISCRETEFUNCTIONAL_LOCAL_CODIM0_INTEGRAL_HH