// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_MAPPER_PDELAB_HH
#define DUNE_GDT_MAPPER_PDELAB_HH

#include <dune/common/dynvector.hh>
#include <dune/common/typetraits.hh>

#if HAVE_DUNE_PDELAB
# include <dune/pdelab/gridfunctionspace/localfunctionspace.hh>
#endif

#include <dune/stuff/common/parallel/threadstorage.hh>
#include <dune/stuff/common/type_utils.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace Mapper {

#if HAVE_DUNE_PDELAB


// forwards
template< class PdelabSpaceImp >
class ContinuousPdelabWrapper;

template< class PdelabSpaceImp >
class DiscontinuousPdelabWrapper;


namespace internal {


template< class PdelabSpaceImp >
class ContinuousPdelabWrapperTraits
{
public:
  typedef ContinuousPdelabWrapper< PdelabSpaceImp > derived_type;
  typedef PdelabSpaceImp                            BackendType;
  //typedef typename BackendType::Element             EntityType;
  typedef typename BackendType::Traits::GridViewType::template Codim< 0>::Entity EntityType;
};

template< class PdelabSpaceImp >
class DiscontinuousPdelabWrapperTraits
{
public:
  typedef DiscontinuousPdelabWrapper< PdelabSpaceImp > derived_type;
  typedef PdelabSpaceImp                               BackendType;
  typedef typename BackendType::Element                EntityType;
};


template< class ImpTraits >
class PdelabWrapperBase
  : public MapperInterface< ImpTraits >
{
  typedef MapperInterface< ImpTraits > InterfaceType;
public:
  typedef typename InterfaceType::EntityType  EntityType;
  typedef typename InterfaceType::BackendType BackendType;
private:
  typedef PDELab::LocalFunctionSpace< BackendType, PDELab::TrialSpaceTag > PdeLabLFSType;

public:
  explicit PdelabWrapperBase(const BackendType& pdelab_space)
    : backend_(pdelab_space)
    , lfs_(backend_)
  {}

  virtual ~PdelabWrapperBase(){}

  const BackendType& backend() const
  {
    return backend_;
  }

  size_t size() const
  {
    return backend_.size();
  }

  size_t numDofs(const EntityType& entity) const
  {
    lfs_.bind(entity);
    return lfs_.size();
  }

  size_t maxNumDofs() const
  {
    return backend_.maxLocalSize();
  }

  void globalIndices(const EntityType& entity, Dune::DynamicVector< size_t >& ret) const
  {
    lfs_.bind(entity);
    // some checks
    const size_t numLocalDofs = numDofs(entity);
    if (ret.size() < numLocalDofs)
      ret.resize(numLocalDofs);
    // compute
    for (size_t ii = 0; ii < numLocalDofs; ++ii)
      ret[ii] = mapToGlobal(entity, ii);
  } // ... globalIndices(...)

  using InterfaceType::globalIndices;

  size_t mapToGlobal(const EntityType& entity, const size_t& localIndex) const
  {
    lfs_.bind(entity);
    assert(localIndex < lfs_.size());
    return mapAfterBound(entity, localIndex);
  } // ... mapToGlobal(...)

protected:
  virtual size_t mapAfterBound(const EntityType& entity, const size_t& localIndex) const = 0;

  const BackendType& backend_;
  mutable PdeLabLFSType lfs_;
}; // class PdelabWrapperBase


} // namespace internal


template< class PdelabSpaceImp >
class ContinuousPdelabWrapper
  : public internal::PdelabWrapperBase< internal::ContinuousPdelabWrapperTraits< PdelabSpaceImp > >
{
public:
  typedef typename internal::ContinuousPdelabWrapperTraits< PdelabSpaceImp > Traits;
  typedef typename Traits::EntityType                                        EntityType;

  template< class... Args >
  ContinuousPdelabWrapper(Args&& ...args)
    : internal::PdelabWrapperBase< Traits >(std::forward< Args >(args)...)
  {}

protected:
  virtual size_t mapAfterBound(const EntityType& /*entity*/, const size_t& localIndex) const override
  {
    return this->lfs_.dofIndex(localIndex).entityIndex()[1];
  }
}; // class ContinuousPdelabWrapper


template< class PdelabSpaceImp >
class ContinuousPowerPdelabWrapper
  : public internal::PdelabWrapperBase< internal::ContinuousPdelabWrapperTraits< PdelabSpaceImp > >
{
public:
  typedef typename internal::ContinuousPdelabWrapperTraits< PdelabSpaceImp > Traits;
  typedef typename Traits::EntityType                                        EntityType;
  static const size_t                                                        dimRange = PdelabSpaceImp::Traits::CHILDREN;

  template< class... Args >
  ContinuousPowerPdelabWrapper(Args&& ...args)
    : internal::PdelabWrapperBase< Traits >(std::forward< Args >(args)...)
  {}

protected:
  virtual size_t mapAfterBound(const EntityType& entity, const size_t& localIndex) const override
  {
    auto numlocalscalardofs = this->numDofs(entity)/dimRange;
    auto numglobalscalardofs = this->size()/dimRange;
    auto numcomp = localIndex/numlocalscalardofs;
    return numcomp * numglobalscalardofs + this->lfs_.dofIndex(localIndex).entityIndex()[1];
  }
}; // class ContinuousPowerPdelabWrapper


template< class PdelabSpaceImp >
class DiscontinuousPdelabWrapper
  : public internal::PdelabWrapperBase< internal::DiscontinuousPdelabWrapperTraits< PdelabSpaceImp > >
{
public:
  typedef typename internal::DiscontinuousPdelabWrapperTraits< PdelabSpaceImp > Traits;
  typedef typename Traits::EntityType                                           EntityType;

  template< class... Args >
  DiscontinuousPdelabWrapper(Args&& ...args)
    : internal::PdelabWrapperBase< Traits >(std::forward< Args >(args)...)
  {}

protected:
  virtual size_t mapAfterBound(const EntityType& entity, const size_t& localIndex) const override
  {
    return this->lfs_.dofIndex(localIndex).entityIndex()[1] * this->numDofs(entity) + localIndex;
  }
}; // class DiscontinuousPdelabWrapper


#else // HAVE_DUNE_PDELAB


template< class PdelabSpaceImp >
class ContinuousPdelabWrapper
{
  static_assert(Dune::AlwaysFalse< PdelabSpaceImp >::value, "You are missing dune-pdelab!");
};

template< class PdelabSpaceImp >
class DiscontinuousPdelabWrapper
{
  static_assert(Dune::AlwaysFalse< PdelabSpaceImp >::value, "You are missing dune-pdelab!");
};


#endif // HAVE_DUNE_PDELAB

} // namespace Mapper
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_MAPPER_PDELAB_HH
