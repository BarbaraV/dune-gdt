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

//static constexpr ChooseSpaceBackend default_nedelec_backend = default_space_backend;

template< class ImpTraits, size_t domainDim, size_t rangeDim, size_t rangeDimCols = 1 >
class NedelecInterface
  :public SpaceInterface< ImpTraits, domainDim, rangeDim, rangeDimCols >
{
  typedef SpaceInterface< ImpTraits, domainDim, rangeDim, rangeDimCols >   BaseType;
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
private:
  static const constexpr RangeFieldType compare_tolerance_ = 1e-13;
public:

  /**
   * \defgroup interface ´´These methods have to be implemented!''
   * @{
   **/


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


  /**
   * @brief local_dirichlet_DoFs_order1 computes the local degrees of freedom on the dirichlet boundary for polynomial order 1
   * @param entity Entity on which the dirichlet dofs are computed
   * @param boundaryInfo Boundary Info to give the (local) Dirichlet boundary
   * @return a set of local indices which lie on the Dirichlet boundary
   */
  std::set< size_t > local_dirichlet_DoFs_order_1(const EntityType& entity,
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
    const auto& reference_element = ReferenceElements< DomainFieldType, dimDomain >::general(entity.type());
    const auto num_vertices = boost::numeric_cast< size_t >(entity.template count< dimDomain >());
    std::vector< bool > liesonintersection(num_vertices, false);
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
        //check which vertices lie on the intersection
        for (size_t vv = 0; vv < num_vertices; ++vv) {
          const auto vertex_ptr = entity.template subEntity< dimDomain >(boost::numeric_cast< int >(vv));
          const auto& vertex = *vertex_ptr;
          for (auto cc : DSC::valueRange(geometry.corners())) {
            corner = geometry.corner(boost::numeric_cast< int >(cc));
            if (Stuff::Common::FloatCmp::eq(vertex.geometry().center(), corner))
              liesonintersection[vv] = true;
          } //loop over all corners of the intersection
        } //loop over all vertices
        //now get the vertex opposite to the intersection (i.e. the one which does not lie on it)
        size_t found = 0;
        size_t missed = 0;
        for (size_t vvv = 0; vvv < num_vertices; ++vvv) {
          if (!(liesonintersection[vvv])) {
            vertexoppDirirchlet.emplace_back(reference_element.position(vvv, dimDomain));
            ++found;
          } else
            ++missed;
            //clear for next intersection
            liesonintersection[vvv] = false;
        } //loop over all vertices
        //assert only one opposite vertex was found
        assert(found == 1 && missed == dimDomain && "This must not happen for tetrahedral meshes!");
      } //only work on dirichlet intersections
    } //loop over all intersections
    // get all the basefunctions which evaluate to 0 there
    //(must be exactly dimDomain for polOrder 1), these are added to localdirichletdofs
    const auto basis = this->base_function_set(entity);
    typedef typename BaseType::BaseFunctionSetType::RangeType RangeType;
    std::vector< RangeType > tmp_basis_values(basis.size(), RangeType(0));
    for (size_t cc = 0; cc < vertexoppDirirchlet.size(); ++cc) {
      basis.evaluate(vertexoppDirirchlet[cc], tmp_basis_values);
      size_t zeros = 0;
      size_t nonzeros = 0;
      for (size_t ii = 0; ii < basis.size(); ++ii) {
        if (tmp_basis_values[ii].two_norm() < compare_tolerance_) {
          localDirichletDoFs.insert(ii);
          ++zeros;
        } else
          ++nonzeros;
      }
      assert(zeros == dimDomain && "This must not happen for polynomial order 1!");
    }
    return localDirichletDoFs;
  } //... local_dirichlet_DoFs_order0(...)

  using BaseType::compute_pattern;

  template < class G, class S, size_t d, size_t r, size_t rC >
  PatternType compute_pattern(const GridView< G >& local_grid_view, const SpaceInterface< S, d, r, rC >& ansatz_space) const
  {
    DSC::TimedLogger().get("gdt.spaces.nedelec.pdelab.compute_pattern").warn() << "Returning largest possible pattern!"
                                                                            << std::endl;
    return BaseType::compute_face_and_volume_pattern(local_grid_view, ansatz_space);
  }

  using BaseType::local_constraints;
  template< class S, size_t d, size_t r, size_t rC, class ConstraintsType >
  void local_constraints(const SpaceInterface< S, d, r, rC >& /*ansatz_space*/,
                         const EntityType& /*entity*/,
                         ConstraintsType& /*ret*/) const
  {
    static_assert(AlwaysFalse< S >::value, "Not implemented for these constraints!");
  }

  template< class S, size_t d, size_t r, size_t rC >
  void local_constraints(const SpaceInterface< S, d, r, rC >& /*other*/,
                         const EntityType& entity,
                         DirichletConstraints< IntersectionType >& ret) const
  {
    static_assert(polOrder == 1, "Not tested for higher polynomial orders!");
    static_assert(dimDomain == 3, "Not implemented!");
    const auto local_DoFs = this->local_dirichlet_DoFs(entity, ret.boundary_info());
    if (local_DoFs.size() > 0) {
      const auto global_indices = this->mapper().globalIndices(entity);
      for (const auto& localDoF : local_DoFs) {
        ret.insert(global_indices[localDoF]);
      }
    }
  } //... local_constraints(..., Constraints::Dirichlet<...>....)

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
