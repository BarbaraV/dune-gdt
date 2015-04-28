// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_SPACES_NEDELEC_INTERFACE_HH
#define DUNE_GDT_SPACES_NEDELEC_INTERFACE_HH

#include <boost/numeric/conversion/cast.hpp>
#include <dune/stuff/common/timedlogging.hh>
#include <dune/gdt/spaces/interface.hh>

namespace Dune {
namespace GDT {
namespace Spaces {

template< class ImpTraits, size_t domainDim, size_t rangeDim, size_t rangeDimCols = 1 >
class NedelecInterface
  :public SpaceInterface< ImpTraits, domainDim, rangeDim, rangeDimCols >
{
  typedef SpaceInterface< ImpTraits, domainDim, rangeDim, rangeDimCols > BaseType;
  typedef NedelecInterface< ImpTraits, domainDim, rangeDim, rangeDimCols > ThisType;
public:
  typedef ImpTraits Traits;

  using BaseType::polOrder;
  using BaseType::dimDomain;
  using typename BaseType::DomainFieldType;
  using typename BaseType::DomainType;
  using typename BaseType::RangeFieldType;
  using typename BaseType::EntityType;
  using typename BaseType::IntersectionType;
  using typename BaseType::BoundaryInfoType;
  using typename BaseType::BaseFunctionSetType;
  using typename BaseType::PatternType;

  //implement localDoFs and "Dirichlet"DoFs
  //implement constraints

  /**
   * \defgroup interface ´´These methods have to be implemented!''
   * @{
   **/
 /* std::vector< size_t > local_DoF_indices(const EntityType& entity) const
  {
    CHECK_CRTP(this->as_imp().local_DoF_indices(entity));
    return this->as_imp().local_DoF_indices(entity);
  } // ... local_DoF_indices(...)                                           // as in RTInterface
*/

  std::set< size_t > local_dirichlet_DoFs(const EntityType& entity,
                                          const BoundaryInfoType& boundaryInfo) const
  {
    CHECK_CRTP(this->as_imp().local_dirichlet_DoFs(entity, boundaryInfo));
    return this->as_imp().local_dirichlet_DoFs(entity, boundaryInfo);
  } // ... local_dirichlet_DoFs(...)                                       //as in CGInterface
  /** @} */

  /**
   * \defgroup provided ´´These methods are provided by the interface for convenience.''
   * @{
   */

/*  std::vector< size_t > local_DoF_indices_3dsimplex_order1(const EntityType& entity) const
  {
    static_assert(dimDomain == 3, "Not implemented");
    static_assert(polOrder == 1, "Not implemented");
    const auto num_edges = boost::numeric_cast< size_t >(entity.template count< 2 >());    // ergibt das die Zahl der codim2 entities?
    std::vector< size_t > local_DoF_index_of_edge(num_edges, std::numeric_limits< size_t >::infinity());
    //do something intelligent
    return local_DoF_index_of_edge;
  } //...local_DoF_indices_3dsimplex_order0(...)                                            //as in RTInterface
*/


  /**
   * @brief local_dirichlet_DoFs_order1 computes the local degrees of freedom on the dirichlet boundary for polynomial order 1
   * @param entity Entity on which the dirichlet dofs are computed
   * @param boundaryInfo Boundary Info to give the (local) Dirichlet boundary
   * @return a set of local indices which lie on the Dirichlet boundary
   * @todo implement this as it was recently done for elliptic problems to avoid problems with edges which are not part of a full Dirichlet face!
   */
  std::set< size_t > local_dirichlet_DoFs_order1(const EntityType& entity,
                                                 const BoundaryInfoType& boundaryInfo) const
  {
    static_assert(polOrder == 1, "Not tested for higher polynomial orders!");
    static_assert(dimDomain == 3, "Not implemented!");
    //check
    assert(this->grid_view().indexSet().contains(entity));
    //prepare
    std::set< size_t > localDirichletDoFs;
    std::vector< DomainType > vertexoppDirirchlet;
    DomainType corner(0);
    const auto num_vertices = boost::numeric_cast< size_t >(entity.template count< dimDomain >());
    //get all dirichlet edges of this entity
    //loop over all intersections
    const auto intersection_it_end = this->grid_view().iend(entity);
    for (auto intersection_it = this->grid_view().ibegin(entity);
           intersection_it != intersection_it_end;
           ++intersection_it) {
        //only work in dirichlet intersections
        const auto& intersection = *intersection_it;
        //actual dirichlet intersections+process bdries for parallel run
        if (boundaryInfo.dirichlet(intersection) || (!intersection.neighbor() && !intersection.boundary())) {
            const auto geometry = intersection.geometry();
            //get the vertex opposite to that intersection
            for (size_t vv = 0; vv < num_vertices; ++vv) {
                const auto vertex_ptr = entity.template subEntity< dimDomain >(boost::numeric_cast< int >(vv));
                const auto& vertex = *vertex_ptr;
                for (auto cc : DSC::valueRange(geometry.corners())) {
                    corner = geometry.corner(boost::numeric_cast< int >(cc));
                    if (!Stuff::Common::FloatCmp::eq(vertex.geometry().center(), corner))
                        vertexoppDirirchlet.emplace_back(vertex.geometry().center());
                } //loop over all corners of the intersection
            } //loop over all vertices
        } //only work on dirichlet intersections
    } //loop over all intersections
    // get all the basefunctions which evaluate to 0 there
    //(must be exactly dimDomain for polOrder 1!), these are added to localdirichletdofs
    const auto basis = this->base_function_set(entity);
    typedef typename BaseType::BaseFunctionSetType::RangeType RangeType;
    const RangeType one(1);
    std::vector< RangeType > tmp_basis_values(basis.size(), RangeType(0));
    for (size_t cc = 0; cc < vertexoppDirirchlet.size(); ++cc) {
        basis.evaluate(vertexoppDirirchlet[cc], tmp_basis_values);
        size_t zeros = 0;
        size_t nonzeros = 0;
        for (size_t ii = 0; ii < basis.size(); ++ii) {
            if (Stuff::Common::FloatCmp::eq(tmp_basis_values[ii] + one, one)) {
                localDirichletDoFs.insert(ii);
                ++zeros;
            } else
                ++nonzeros;
        }
        assert(zeros == dimDomain && "This must not happen for polynomial order 1!");
    }
   return localDirichletDoFs;
  } //... local_dirichlet_DoFs_order0(...)
    //might give a problem for an entity with a Dirichlet edge, which is not part of a Dirichlet intersection

  using BaseType::compute_pattern;

  template < class G, class S, size_t d, size_t r, size_t rC >
  PatternType compute_pattern(const GridView< G >& local_grid_view, const SpaceInterface< S, d, r, rC >& ansatz_space) const
  {
    DSC::TimedLogger().get("gdt.spaces.nedelec.pdelab.compute_pattern").warn() << "Returning largest possible pattern!"
                                                                            << std::endl;
    return BaseType::compute_face_and_volume_pattern(local_grid_view, ansatz_space);
  }                                                                         //as in RTInterface

  using BaseType::local_constraints;
  template< class S, size_t d, size_t r, size_t rC, class ConstraintsType >
  void local_constraints(const SpaceInterface< S, d, r, rC >& /*ansatz_space*/,
                         const EntityType& /*entity*/,
                         ConstraintsType& /*ret*/) const
  {
    static_assert(AlwaysFalse< S >::value, "Not implemented for these constraints!");
  }

  template< class S, size_t d, size_t r, size_t rC >
  void local_constraints(const SpaceInterface< S, d, r, rC >& other,
                         const EntityType& entity,
                         Constraints::Dirichlet< IntersectionType, RangeFieldType >& ret) const
  {
    static_assert(polOrder == 1, "Not tested for higher polynomial orders!");
    static_assert(dimDomain == 3, "Not implemented!");
    assert(this->grid_view().indexSet().contains(entity));
    const std::set< size_t > localDirichletDofs = this->local_dirichlet_DoFs(entity, ret.boundary_info());
    const size_t numRows = localDirichletDofs.size();
    Dune::DynamicVector< size_t > tmpMappedRows;
    Dune::DynamicVector< size_t > tmpMappedCols;
    if (numRows > 0) {
      const size_t numCols = this->mapper().numDofs(entity);
      ret.set_size(numRows, numCols);
      this->mapper().globalIndices(entity, tmpMappedRows);
      other.mapper().globalIndices(entity, tmpMappedCols);
      size_t localRow = 0;
      for (const size_t& localDirichletDofIndex : localDirichletDofs) {
        ret.global_row(localRow) = tmpMappedRows[localDirichletDofIndex];
        for (size_t jj = 0; jj < ret.cols(); ++jj) {
          ret.global_col(jj) = tmpMappedCols[jj];
          if (tmpMappedCols[jj] == tmpMappedRows[localDirichletDofIndex])
            ret.value(localRow, jj) = ret.set_row() ? 1 : 0;
          else
            ret.value(localRow, jj) = 0;
        }
        ++localRow;
      }
    } else {
      ret.set_size(0, 0);
    }
  } //... local_constraints(..., Constraints::Dirichlet<...>....)   //as in CGInterface

  /** @} */

}; //class NedelecInterface


} //namespace Spaces

//space helper as in rtinterface or cginterface
namespace internal {


template< class S >
struct is_nedelec_space_helper
{
  DSC_has_typedef_initialize_once(Traits)
  DSC_has_static_member_initialize_once(dimDomain)
  DSC_has_static_member_initialize_once(dimRange)
  DSC_has_static_member_initialize_once(dimRangeCols)

  static const bool is_candidate = DSC_has_typedef(Traits)< S >::value
                                   && DSC_has_static_member(dimDomain)< S >::value
                                   && DSC_has_static_member(dimRange)< S >::value
                                   && DSC_has_static_member(dimRangeCols)< S >::value;
}; // class is_nedelec_space_helper


} // namespace internal


template< class S, bool candidate = internal::is_nedelec_space_helper< S >::is_candidate >
struct is_nedelec_space
  : public std::is_base_of< Spaces::NedelecInterface< typename S::Traits, S::dimDomain, S::dimRange, S::dimRangeCols >
                          , S >
{};


template< class S >
struct is_nedelec_space< S, false >
  : public std::false_type
{};


} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_SPACES_NEDELEC_INTERFACE_HH
