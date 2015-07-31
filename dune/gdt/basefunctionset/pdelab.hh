// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_BASEFUNCTIONSET_PDELAB_HH
#define DUNE_GDT_BASEFUNCTIONSET_PDELAB_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#if HAVE_DUNE_PDELAB
# include <dune/pdelab/gridfunctionspace/localfunctionspace.hh>
# include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
# include <dune/pdelab/constraints/conforming.hh>
#endif

#include <dune/stuff/common/type_utils.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace BaseFunctionSet {

#if HAVE_DUNE_PDELAB


// forwards, to be used in the traits and to allow for specialization
template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PdelabWrapper
{
  static_assert(Dune::AlwaysFalse< PdelabSpaceImp >::value, "Untested for arbitrary dimension!");
};


template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PiolaTransformedPdelabWrapper
{
  static_assert(Dune::AlwaysFalse< PdelabSpaceImp >::value, "Untested for these dimensions!");
};

//new
template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class Edges05PdelabWrapper
{
  static_assert(Dune::AlwaysFalse< PdelabSpaceImp >::value, "Untested for these dimensions!");
};

namespace internal {


// forward, to allow for specialization
template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols >
class PdelabWrapperTraits;


//! Specialization for dimRange = 1, dimRangeRows = 1
template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp >
class PdelabWrapperTraits< PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >
{
public:
  typedef PdelabWrapper < PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 > derived_type;
private:
  typedef PDELab::LocalFunctionSpace< PdelabSpaceImp, PDELab::TrialSpaceTag > PdelabLFSType;
  typedef FiniteElementInterfaceSwitch< typename PdelabSpaceImp::Traits::FiniteElementType > FESwitchType;
public:
  typedef typename FESwitchType::Basis BackendType;
  typedef EntityImp EntityType;
private:
  friend class PdelabWrapper < PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >;
};


template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim >
class PdelabWrapperTraits< PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
{
public:
  typedef PdelabWrapper < PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 > derived_type;
private:
  typedef PDELab::GridFunctionSpace< typename PdelabSpaceImp::Traits::GridViewType,
                                     typename PdelabSpaceImp::BaseT::ChildType::Traits::FiniteElementMapType, PDELab::OverlappingConformingDirichletConstraints > GFS;
  typedef PDELab::LocalFunctionSpace< PdelabSpaceImp, PDELab::TrialSpaceTag > PdelabLFSType;
  typedef FiniteElementInterfaceSwitch< typename GFS::Traits::FiniteElementType > FESwitchType;
public:
  typedef typename FESwitchType::Basis BackendType;
  typedef EntityImp EntityType;
private:
  friend class PdelabWrapper < PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >;
};


template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim >
class PiolaTransformedPdelabWrapperTraits
{
  static_assert(domainDim == rangeDim, "Untested!");
public:
  typedef PiolaTransformedPdelabWrapper < PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim >
      derived_type;
private:
  typedef PDELab::LocalFunctionSpace< PdelabSpaceImp, PDELab::TrialSpaceTag > PdelabLFSType;
  typedef FiniteElementInterfaceSwitch< typename PdelabSpaceImp::Traits::FiniteElementType > FESwitchType;
public:
  typedef typename FESwitchType::Basis BackendType;
  typedef EntityImp EntityType;
private:
  friend class PiolaTransformedPdelabWrapper < PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >;
};

//new
/** \brief Traits for the class Edges05PdelabWrapper
 *
 * \sa Edges05PdelabWrapper
*/
template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim >
class Edges05PdelabWrapperTraits
{
  static_assert(domainDim == rangeDim, "Untested!");
  static_assert(domainDim == 3, "Untested!");
public:
  typedef Edges05PdelabWrapper< PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim >
      derived_type;
private:
  typedef PDELab::LocalFunctionSpace< PdelabSpaceImp, PDELab::TrialSpaceTag > PdelabLFSType;
  typedef FiniteElementInterfaceSwitch< typename PdelabSpaceImp::Traits::FiniteElementType > FESwitchType;
public:
  typedef typename FESwitchType::Basis BackendType;
  typedef EntityImp EntityType;
private:
  friend class Edges05PdelabWrapper< PdelabSpaceImp, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >;
};


} // namespace internal


//! Specialization for dimRange = 1, dimRangeRows = 1
template< class PdelabSpaceType, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp >
class PdelabWrapper< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >
  : public BaseFunctionSetInterface< internal::PdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                                       DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >,
                                     DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >
{
  typedef PdelabWrapper < PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 > ThisType;
  typedef BaseFunctionSetInterface
      < internal::PdelabWrapperTraits< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >,
        DomainFieldImp, domainDim, RangeFieldImp, 1, 1 >
      BaseType;
public:
  typedef internal::PdelabWrapperTraits< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 > Traits;
  typedef typename Traits::BackendType   BackendType;
  typedef typename Traits::EntityType    EntityType;
private:
  typedef typename Traits::PdelabLFSType PdelabLFSType;
  typedef typename Traits::FESwitchType  FESwitchType;

public:
  typedef typename BaseType::DomainType        DomainType;
  typedef typename BaseType::RangeType         RangeType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;

  PdelabWrapper(const PdelabSpaceType& space, const EntityType& ent)
    : BaseType(ent)
    , tmp_domain_(0)
  {
    PdelabLFSType* lfs_ptr = new PdelabLFSType(space);
    lfs_ptr->bind(this->entity());
    lfs_ = std::unique_ptr< PdelabLFSType >(lfs_ptr);
    backend_ = std::unique_ptr< BackendType >(new BackendType(FESwitchType::basis(lfs_->finiteElement())));
  } // PdelabWrapper(...)

  PdelabWrapper(ThisType&& source) = default;
  PdelabWrapper(const ThisType& /*other*/) = delete;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  const BackendType& backend() const
  {
    return *backend_;
  }

  virtual size_t size() const override final
  {
    return backend_->size();
  }

  virtual size_t order() const override final
  {
    return backend_->order();
  }

  virtual void evaluate(const DomainType& xx, std::vector< RangeType >& ret) const override final
  {
    assert(ret.size() >= backend_->size());
    backend_->evaluateFunction(xx, ret);
  }

  using BaseType::evaluate;

  virtual void jacobian(const DomainType& xx, std::vector< JacobianRangeType >& ret) const override final
  {
    assert(ret.size() >= backend_->size());
    backend_->evaluateJacobian(xx, ret);
    const auto jacobian_inverse_transposed = this->entity().geometry().jacobianInverseTransposed(xx);
    for (size_t ii = 0; ii < ret.size(); ++ii) {
      jacobian_inverse_transposed.mv(ret[ii][0], tmp_domain_);
      ret[ii][0] = tmp_domain_;
    }
  } // ... jacobian(...)

  using BaseType::jacobian;

private:
  mutable DomainType tmp_domain_;
  std::unique_ptr< const PdelabLFSType > lfs_;
  std::unique_ptr< const BackendType > backend_;
}; // class PdelabWrapper


//! Specialization for dimRangeRows = 1
template< class PdelabSpaceType, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim >
class PdelabWrapper< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
  : public BaseFunctionSetInterface< internal::PdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                                       DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >,
                                     DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
{
  typedef PdelabWrapper < PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 > ThisType;
  typedef BaseFunctionSetInterface
      < internal::PdelabWrapperTraits< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >,
        DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
      BaseType;
public:
  typedef internal::PdelabWrapperTraits< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 > Traits;
  typedef typename Traits::BackendType               BackendType;
  typedef typename BackendType::Traits::RangeType    ScalarRangeType;
  typedef typename BackendType::Traits::JacobianType ScalarJacobianType;
  typedef typename Traits::EntityType                EntityType;
private:
  typedef typename Traits::PdelabLFSType PdelabLFSType;
  typedef typename Traits::FESwitchType  FESwitchType;

public:
  typedef typename BaseType::DomainType        DomainType;
  typedef typename BaseType::RangeType         RangeType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;

  PdelabWrapper(const PdelabSpaceType& space, const EntityType& ent)
    : BaseType(ent)
    , tmp_domain_(0)
  {
    PdelabLFSType* lfs_ptr = new PdelabLFSType(space);
    lfs_ptr->bind(this->entity());
    lfs_ = std::unique_ptr< PdelabLFSType >(lfs_ptr);
    backend_ = std::unique_ptr< BackendType >(new BackendType(FESwitchType::basis(lfs_->template child<0>().finiteElement())));
    tmp_ranges_ = std::vector< ScalarRangeType >(backend_->size(), ScalarRangeType(0));
    tmp_jacobian_ranges_ = std::vector< ScalarJacobianType >(backend_->size(), ScalarJacobianType(0));
  } // PdelabWrapper(...)

  PdelabWrapper(ThisType&& source) = default;
  PdelabWrapper(const ThisType& /*other*/) = delete;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  const BackendType& backend() const
  {
    return *backend_;
  }

  virtual size_t size() const override final
  {
    return (backend_->size())*rangeDim;
  }

  virtual size_t order() const override final
  {
    return backend_->order();
  }

  virtual void evaluate(const DomainType& xx, std::vector< RangeType >& ret) const override final
  {
    assert(ret.size() >= (backend_->size())*rangeDim);
    backend_->evaluateFunction(xx, tmp_ranges_);
    for (size_t ii = 0; ii < rangeDim; ++ii)
      for (size_t jj = 0; jj < backend_->size(); ++jj)
        ret[ii*(backend_->size())+jj][ii] = tmp_ranges_[jj];
  }

  using BaseType::evaluate;

  //untested for vector-valued!
  virtual void jacobian(const DomainType& xx, std::vector< JacobianRangeType >& ret) const override final
  {
    assert(ret.size() >= (backend_->size())*rangeDim);
    backend_->evaluateJacobian(xx, tmp_jacobian_ranges_);
    for (size_t ii =0; ii < rangeDim; ++ii)
      for (size_t jj = 0; jj < backend_->size(); ++jj)
        ret[ii*(backend_->size())+jj][ii] = tmp_jacobian_ranges_[jj][0];
    const auto jacobian_inverse_transposed = this->entity().geometry().jacobianInverseTransposed(xx);
    for (size_t ii = 0; ii < ret.size(); ++ii) {
      for (size_t jj = 0; jj < rangeDim; ++jj) {
        jacobian_inverse_transposed.mv(ret[ii][jj], tmp_domain_);
        ret[ii][jj] = tmp_domain_;
      }
    }
  } // ... jacobian(...)

  using BaseType::jacobian;

private:
  mutable DomainType tmp_domain_;
  std::unique_ptr< const PdelabLFSType > lfs_;
  std::unique_ptr< const BackendType > backend_;
  mutable std::vector< ScalarRangeType > tmp_ranges_;
  mutable std::vector< ScalarJacobianType > tmp_jacobian_ranges_;
}; // class PdelabWrapper


template< class PdelabSpaceType, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim >
class PiolaTransformedPdelabWrapper< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
  : public BaseFunctionSetInterface< internal::PiolaTransformedPdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                                                       DomainFieldImp, domainDim,
                                                                       RangeFieldImp, rangeDim >,
                                     DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
{
  typedef PiolaTransformedPdelabWrapper
    < PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 > ThisType;
  typedef BaseFunctionSetInterface< internal::PiolaTransformedPdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                                                      DomainFieldImp, domainDim,
                                                                      RangeFieldImp, rangeDim >,
                                    DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
      BaseType;
public:
  typedef internal::PiolaTransformedPdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                            DomainFieldImp, domainDim,
                                            RangeFieldImp, rangeDim >
                                         Traits;
  typedef typename Traits::BackendType   BackendType;
  typedef typename Traits::EntityType    EntityType;
private:
  typedef typename Traits::PdelabLFSType PdelabLFSType;
  typedef typename Traits::FESwitchType  FESwitchType;

public:
  using typename BaseType::DomainFieldType;
  using BaseType::dimDomain;
  using typename BaseType::DomainType;
  using typename BaseType::RangeType;
  using typename BaseType::JacobianRangeType;

  PiolaTransformedPdelabWrapper(const PdelabSpaceType& space, const EntityType& ent)
    : BaseType(ent)
    , tmp_domain_(DomainFieldType(0))
    , tmp_jacobian_transposed_(DomainFieldType(0))
    , tmp_jacobian_inverse_transposed_(DomainFieldType(0))
  {
    PdelabLFSType* lfs_ptr = new PdelabLFSType(space);
    lfs_ptr->bind(this->entity());
    lfs_ = std::unique_ptr< PdelabLFSType >(lfs_ptr);
    backend_ = std::unique_ptr< BackendType >(new BackendType(FESwitchType::basis(lfs_->finiteElement())));
    tmp_ranges_ = std::vector< RangeType >(backend_->size(), RangeType(0));
    tmp_jacobian_ranges_ = std::vector< JacobianRangeType >(backend_->size(), JacobianRangeType(0));
  } // PdelabWrapper(...)

  PiolaTransformedPdelabWrapper(ThisType&& source) = default;

  PiolaTransformedPdelabWrapper(const ThisType& /*other*/) = delete;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  const BackendType& backend() const
  {
    return *backend_;
  }

  virtual size_t size() const override final
  {
    return backend_->size();
  }

  virtual size_t order() const override final
  {
    return backend_->order();
  }

  virtual void evaluate(const DomainType& xx, std::vector< RangeType >& ret) const override final
  {
    assert(lfs_);
    assert(backend_);
    assert(tmp_ranges_.size() >= backend_->size());
    assert(ret.size() >= backend_->size());
    backend_->evaluateFunction(xx, tmp_ranges_);
    const auto geometry = this->entity().geometry();
    tmp_jacobian_transposed_ = geometry.jacobianTransposed(xx);
    const DomainFieldType integration_element = geometry.integrationElement(xx);
    for (size_t ii = 0; ii < backend_->size(); ++ii) {
      tmp_jacobian_transposed_.mtv(tmp_ranges_[ii], ret[ii]);
      ret[ii] /= integration_element;
    }
  } // ... evaluate(...)

  using BaseType::evaluate;

  virtual void jacobian(const DomainType& xx, std::vector< JacobianRangeType >& ret) const override final
  {
    assert(lfs_);
    assert(backend_);
    assert(ret.size() >= backend_->size());
    backend_->evaluateJacobian(xx, tmp_jacobian_ranges_);
    const auto geometry = this->entity().geometry();
    tmp_jacobian_transposed_ = geometry.jacobianTransposed(xx);
    tmp_jacobian_inverse_transposed_ = geometry.jacobianInverseTransposed(xx);
    const DomainFieldType integration_element = geometry.integrationElement(xx);
    for (size_t ii = 0; ii < backend_->size(); ++ii) {
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        tmp_jacobian_inverse_transposed_.mv(tmp_jacobian_ranges_[ii][jj], ret[ii][jj]);
        tmp_jacobian_transposed_.mv(ret[ii][jj], tmp_jacobian_ranges_[ii][jj]);
        tmp_jacobian_ranges_[ii][jj] /= integration_element;
        ret[ii][jj] = tmp_jacobian_ranges_[ii][jj];
      }
    }
  } // ... jacobian(...)

  using BaseType::jacobian;

private:
  mutable DomainType tmp_domain_;
  mutable typename EntityType::Geometry::JacobianTransposed tmp_jacobian_transposed_;
  mutable typename EntityType::Geometry::JacobianInverseTransposed tmp_jacobian_inverse_transposed_;
  std::unique_ptr< const PdelabLFSType > lfs_;
  std::unique_ptr< const BackendType > backend_;
  mutable std::vector< RangeType > tmp_ranges_;
  mutable std::vector< JacobianRangeType > tmp_jacobian_ranges_;
}; // class PiolaTransformedPdelabWrapper


/** \brief Wrapper-class for the Hcurl-confroming transformation, e.g. for member functions of the Nedelec spaces
 *
 * \note As the curl is only defined in dimension 3, the domainDim and the rangeDim have to equal 3
 *
 * \tparam PdelabSpaceType Type of function space, has to be implemented in dune-pdelab/finiteelementmap
 * \tparam EntityImp Type of Entity the transformation maps from
 * \tparam DomainFieldImp Type of the domain field
 * \tparam domainDim Dimension of the domain, has to be 3
 * \tparam RangeFieldImp Type of the range field
 * \tparam rangeDim Dimension of the range, has to be 3
 */
template< class PdelabSpaceType, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim >
class Edges05PdelabWrapper< PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
  : public BaseFunctionSetInterface< internal::Edges05PdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                                                       DomainFieldImp, domainDim,
                                                                       RangeFieldImp, rangeDim >,
                                     DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
{
  typedef Edges05PdelabWrapper
    < PdelabSpaceType, EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 > ThisType;
  typedef BaseFunctionSetInterface< internal::Edges05PdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                                                      DomainFieldImp, domainDim,
                                                                      RangeFieldImp, rangeDim >,
                                    DomainFieldImp, domainDim, RangeFieldImp, rangeDim, 1 >
      BaseType;
public:
  typedef internal::Edges05PdelabWrapperTraits< PdelabSpaceType, EntityImp,
                                            DomainFieldImp, domainDim,
                                            RangeFieldImp, rangeDim >
      Traits;
  typedef typename Traits::BackendType   BackendType;
  typedef typename Traits::EntityType    EntityType;
private:
  typedef typename Traits::PdelabLFSType PdelabLFSType;
  typedef typename Traits::FESwitchType  FESwitchType;

public:
  using typename BaseType::DomainFieldType;
  using typename BaseType::DomainType;
  using typename BaseType::RangeType;
  using typename BaseType::JacobianRangeType;
  using BaseType::dimDomain;

  Edges05PdelabWrapper(const PdelabSpaceType& space, const EntityType& ent)
    : BaseType(ent)
    , tmp_domain_(DomainFieldType(0))
    , tmp_jacobian_transposed_(DomainFieldType(0))
    , tmp_jacobian_inverse_transposed_(DomainFieldType(0))
  {
    PdelabLFSType* lfs_ptr = new PdelabLFSType(space);
    lfs_ptr->bind(this->entity());
    lfs_ = std::unique_ptr< PdelabLFSType >(lfs_ptr);
    backend_ = std::unique_ptr< BackendType >(new BackendType(FESwitchType::basis(lfs_->finiteElement())));
    tmp_ranges_ = std::vector< RangeType >(backend_->size(), RangeType(0));
    tmp_jacobian_ranges_ = std::vector< JacobianRangeType >(backend_->size(), JacobianRangeType(0));
  } // PdelabWrapper(...)

  Edges05PdelabWrapper(ThisType&& source) = default;

  Edges05PdelabWrapper(const ThisType& /*other*/) = delete;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  const BackendType& backend() const
  {
    return *backend_;
  }

  virtual size_t size() const override final
  {
    return backend_->size();
  }

  virtual size_t order() const override final
  {
    return backend_->order();
  }

  virtual void evaluate(const DomainType& xx, std::vector< RangeType >& ret) const override final
  {
    assert(lfs_);
    assert(backend_);
    assert(ret.size() >= backend_->size());
    backend_->evaluateFunction(xx, ret);
    //evaluateFunction for edges05 already gives the value on the element and not in the reference configuration, so no mapping here
  } // ... evaluate(...)

  using BaseType::evaluate;

  virtual void jacobian(const DomainType& xx, std::vector< JacobianRangeType >& ret) const override final
  {
    assert(lfs_);
    assert(backend_);
    assert(ret.size() >= backend_->size());
    backend_->evaluateJacobian(xx, ret);
    //again, evaluateJacobian for edge05 already gives the value on the local element and not on the reference element, so no mapping
  } // ... jacobian(...)

  using BaseType::jacobian;

private:
  mutable DomainType tmp_domain_;
  mutable typename EntityType::Geometry::JacobianTransposed        tmp_jacobian_transposed_;
  mutable typename EntityType::Geometry::JacobianInverseTransposed tmp_jacobian_inverse_transposed_;
  std::unique_ptr< const PdelabLFSType > lfs_;
  std::unique_ptr< const BackendType >   backend_;
  mutable std::vector< RangeType >         tmp_ranges_;
  mutable std::vector< JacobianRangeType > tmp_jacobian_ranges_;
}; // class Edges05PdelabWrapper

#else // HAVE_DUNE_PDELAB


template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PdelabWrapper
{
  static_assert(AlwaysFalse< PdelabSpaceImp >::value, "You are missing dune-pdelab!");
};


template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PiolaTransformedPdelabWrapper
{
  static_assert(AlwaysFalse< PdelabSpaceImp >::value, "You are missing dune-pdelab!");
};

//new
template< class PdelabSpaceImp, class EntityImp,
          class DomainFieldImp, size_t domainDim,
          class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class Edges05PdelabWrapper
{
  static_assert(AlwaysFalse< PdelabSpaceImp >::value, "You are missing dune-pdelab!");
};

#endif // HAVE_DUNE_PDELAB

} // namespace BaseFunctionSet
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_BASEFUNCTIONSET_PDELAB_HH
