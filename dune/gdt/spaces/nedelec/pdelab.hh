// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_SPACES_NEDELEC_PDELAB_HH
#define DUNE_GDT_SPACES_NEDELEC_PDELAB_HH

#include <type_traits>
#include <limits>
#include <mutex>

#include <dune/geometry/genericgeometry/topologytypes.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/grid/utility/vertexorderfactory.hh>  //brauchen irgendwas fuer vertexorder
#include <dune/grid/common/grid.hh>
#include <dune/grid/common/capabilities.hh>

#if HAVE_DUNE_PDELAB
# include <dune/pdelab/finiteelementmap/edges0.5fem.hh>
# include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#endif // HAVE_DUNE_PDELAB

#include <dune/stuff/common/float_cmp.hh>
#include <dune/stuff/common/exceptions.hh>
#include <dune/stuff/common/type_utils.hh>

#include <dune/gdt/basefunctionset/pdelab.hh>
#include <dune/gdt/mapper/pdelab.hh>
#include <dune/gdt/spaces/parallel.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace Spaces {
namespace Nedelec {

#if HAVE_DUNE_PDELAB

// forward, to be used in the traits and to allow for specialization
template< class GridViewImp, int polynomialOrder, class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PdelabBased
{
  static_assert(AlwaysFalse< GridViewImp >::value, "Untested for these dimensions or polynomial order!");
}; // class PdelabBased


namespace internal {

/** \brief Traits for the PdelabBased class
 *\sa PdelabBased
 */
template< class GridViewImp, int polynomialOrder, class RangeFieldImp, size_t rangeDim, size_t rangeDimCols >
class PdelabBasedTraits
{
public:
  typedef PdelabBased< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, rangeDimCols > derived_type;
  typedef GridViewImp GridViewType;
  static const int polOrder = polynomialOrder;
  static_assert(polOrder == 1, "Untested!");
  static_assert(rangeDim == GridViewType::dimension, "Untested!");
  static_assert(rangeDim == 3, "Untested!");
  static_assert(rangeDimCols == 1, "Untested!");
private:
  typedef typename GridViewType::ctype DomainFieldType;
  static const size_t                  dimDomain = GridViewType::dimension;
public:
  typedef RangeFieldImp                 RangeFieldType;
private:
  template< class G, bool single_geom, bool is_simplex >
  struct FeMap
  {
    static_assert(AlwaysFalse< G >::value,
                  "This space is only implemented for fully simplicial grids!");
  };
  template< class G >
  struct FeMap< G, true, true >
  {
    typedef Dune::VertexOrderByIdFactory< typename G::GlobalIdSet, size_t> VertexOrderFactory;  //passt das so mit dem IndexSet?
    typedef PDELab::EdgeS0_5FiniteElementMap< typename G::template Codim<0>::Geometry,
                                              VertexOrderFactory, RangeFieldType > Type;   //richtige Template-Argumente?
                                                                       //wie VertexOrder?
  };

  typedef typename GridViewType::Grid GridType;
  static const bool single_geom_ = Dune::Capabilities::hasSingleGeometryType< GridType >::v;
  static const bool simplicial_ = (Dune::Capabilities::hasSingleGeometryType< GridType >::topologyId
                                   == GenericGeometry::SimplexTopology< dimDomain >::type::id);
  typedef typename FeMap< GridType, single_geom_, simplicial_ >::Type FEMapType;
public:
  typedef PDELab::GridFunctionSpace< GridViewType, FEMapType > BackendType;
  typedef Mapper::ContinuousPdelabWrapper< BackendType > MapperType;
  typedef typename GridViewType::template Codim< 0 >::Entity EntityType;
  typedef BaseFunctionSet::HcurlTransformedPdelabWrapper
      < BackendType, EntityType, DomainFieldType, dimDomain, RangeFieldType, rangeDim, rangeDimCols >
    BaseFunctionSetType;
  static const Stuff::Grid::ChoosePartView part_view_type = Stuff::Grid::ChoosePartView::view;
  static const bool needs_grid_view = true;
  typedef CommunicationChooser< GridViewType >    CommunicationChooserType;
  typedef typename CommunicationChooserType::Type CommunicatorType;
private:
  friend class PdelabBased< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, rangeDimCols >;
}; // class PdelabBasedTraits


} // namespace internal


/** \brief Class for a PdelabBased function space of lowest order Nedelec (edge) elements of the first family
* \tparam GridViewImp Type of the used Grid
* \tparam RangeFieldImp Type of the range field
* \tparam rangeDim Dimension of the range
*/
//specification for PDELabBased<.., 1....., 1>
template< class GridViewImp, class RangeFieldImp, size_t rangeDim >
class PdelabBased< GridViewImp, 1, RangeFieldImp, rangeDim, 1 >
  : public Spaces::NedelecInterface< internal::PdelabBasedTraits< GridViewImp, 1, RangeFieldImp, rangeDim, 1 >,
                                GridViewImp::dimension, rangeDim, 1 >
{
  typedef PdelabBased< GridViewImp, 1, RangeFieldImp, rangeDim, 1 >                  ThisType;
  typedef Spaces::NedelecInterface< internal::PdelabBasedTraits< GridViewImp, 1, RangeFieldImp, rangeDim, 1 >,
                               GridViewImp::dimension, rangeDim, 1 >                 BaseType;
public:
  typedef internal::PdelabBasedTraits< GridViewImp, 1, RangeFieldImp, rangeDim, 1 >  Traits;

  using BaseType::dimDomain;
  using BaseType::polOrder;

  using typename BaseType::GridViewType;
  using typename BaseType::BackendType;
  using typename BaseType::MapperType;
  using typename BaseType::BaseFunctionSetType;
  using typename BaseType::CommunicatorType;

  using typename BaseType::PatternType;
  using typename BaseType::EntityType;
  using typename BaseType::BoundaryInfoType;
private:
  typedef typename Traits::FEMapType FEMapType;

public:
  PdelabBased(GridViewType gV)
    : grid_view_(gV)
    , fe_map_(grid_view_)
    , backend_(grid_view_, fe_map_)
    , mapper_(backend_)
    , communicator_(CommunicationChooser< GridViewType >::create(grid_view_))
    , communicator_prepared_(false)
  {}

  /**
   * \brief Copy ctor.
   * \note  Manually implemented bc of the std::mutex and our space creation policy
   *        (see https://github.com/pymor/dune-gdt/issues/28)
   */
  PdelabBased(const ThisType& other)
    : grid_view_(other.grid_view_)
    , fe_map_(grid_view_)
    , backend_(grid_view_, fe_map_)
    , mapper_(backend_)
    , communicator_(CommunicationChooser< GridViewType >::create(grid_view_))
    , communicator_prepared_(false)
  {
    // make sure our new communicator is prepared if other's was
    if (other.communicator_prepared_)
      const auto& DUNE_UNUSED(comm) = this->communicator();
  }

  /**
   * \brief Move ctor.
   * \note  Manually implemented bc of the std::mutex and our space creation policy
   *        (see https://github.com/pymor/dune-gdt/issues/28)
   */
  PdelabBased(ThisType&& source)
    : grid_view_(source.grid_view_)
    , fe_map_(grid_view_)
    , backend_(grid_view_, fe_map_)
    , mapper_(backend_)
    , communicator_(std::move(source.communicator_))
    , communicator_prepared_(source.communicator_prepared_)
  {}

  ThisType& operator=(const ThisType& other) = delete;

  ThisType& operator=(ThisType&& source) = delete;

  const GridViewType& grid_view() const
  {
    return grid_view_;
  }

  const BackendType& backend() const
  {
    return backend_;
  }

  const MapperType& mapper() const
  {
    return mapper_;
  }

  std::set< size_t > local_dirichlet_DoFs(const EntityType& entity,
                                          const BoundaryInfoType& boundaryInfo) const
  {
    return BaseType::local_dirichlet_DoFs_order_1(entity, boundaryInfo);
  }

  BaseFunctionSetType base_function_set(const EntityType& entity) const
  {
    return BaseFunctionSetType(backend_, entity);
  }

  CommunicatorType& communicator() const
  {
    std::lock_guard< std::mutex > DUNE_UNUSED(gg)(communicator_mutex_);
    if (!communicator_prepared_)
      communicator_prepared_ = CommunicationChooser<GridViewType>::prepare(*this, *communicator_);
    return *communicator_;
  } // ... communicator(...)

//helper wie bei RT-PDELABbased benoetigt?

private:
  GridViewType grid_view_;
  const FEMapType fe_map_;
  const BackendType backend_;
  const MapperType mapper_;
  mutable std::unique_ptr< CommunicatorType > communicator_;
  mutable bool communicator_prepared_;
  mutable std::mutex communicator_mutex_;
}; // class PdelabBased< ..., 1, ..., 1 >


#else  //HAVE_DUNE_PDELAB
template< class GridViewImp, int polynomialOrder, class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PdelabBased
{
  static_assert(AlwaysFalse< GridViewImp >::value, "You are missing dune-pdelab!");
};

#endif //HAVE_DUNE_PDELAB
}  //namespace Nedelec
}  //namespace Spaces
}  //namespace GDT
}  //namespace Dune

#endif // DUNE_GDT_SPACES_NEDELEC_PDELAB_HH
