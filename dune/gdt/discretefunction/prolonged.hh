// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_DISCRETEFUNCTION_PROLONGED_HH
#define DUNE_GDT_DISCRETEFUNCTION_PROLONGED_HH

#include <memory>
#include <type_traits>

#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/common/memory.hh>

#include <dune/gdt/discretefunction/default.hh>

namespace Dune {
namespace GDT {


/** \brief Class to prolong a discrete function to a finer grid
 *
 * This class fulfills the GlobalFucntionInterface. The methods evaluate and jacobian are implemented using
 * the EntityInLevelSearch and global coordinates.
 *
 * \note Due to the use of EntityInLevelSearch, this class may not be optimal
 *
 * \tparam FunctionImp The type of discrete function
 * \tparam RangeGridView The (finer) grid to map to
 */
template< class FunctionImp, class RangeGridView >
class ProlongedFunction
  : public Stuff::GlobalFunctionInterface< typename RangeGridView::template Codim<0>::Entity,
                                           typename FunctionImp::DomainFieldType,
                                           FunctionImp::dimDomain,
                                           typename FunctionImp::RangeFieldType,
                                           FunctionImp::dimRange >
{
  typedef Stuff::GlobalFunctionInterface< typename RangeGridView::template Codim<0>::Entity,
                                           typename FunctionImp::DomainFieldType, FunctionImp::dimDomain,
                                           typename FunctionImp::RangeFieldType, FunctionImp::dimRange > BaseType;
  typedef ProlongedFunction< FunctionImp, RangeGridView >                                                ThisType;

public:
  typedef typename BaseType::LocalfunctionType LocalfunctionType;
  typedef typename BaseType::DomainType        DomainType;
  typedef typename BaseType::RangeType         RangeType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;

  static_assert(is_const_discrete_function< FunctionImp >::value || is_discrete_function< FunctionImp >::value, "Function to be prolonged has to be a discrete function!");

  static const bool available = false;

  static std::string static_id()
  {
    return FunctionImp::static_id() + ".prolonged";
  }

  ProlongedFunction(const FunctionImp& source_function, const RangeGridView& grid_view)
    : source_function_(source_function)
    , range_grid_view_(grid_view)
  {}

  ProlongedFunction(const ThisType& /*other*/) = default;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  virtual std::string type() const override final
  {
    return FunctionImp::static_id() + ".prolonged";
  }

  virtual std::string name() const override final
  {
    return FunctionImp::static_id() + ".prolonged";
  }

  virtual size_t order() const override final
  {
    //guess the order by hoping that source's order is the same on all entities
    return source_function_.local_function(*source_function_.space().grid_view().template begin<0>())->order();
  }

  virtual void evaluate(const DomainType& xx, RangeType& ret) const override final
  {
    //create entity search in the source grid view
    typedef typename FunctionImp::SpaceType::GridViewType SourceGridViewType;
    typedef Stuff::Grid::EntityInlevelSearch< SourceGridViewType > EntitySearch;
    EntitySearch entity_search(source_function_.space().grid_view());
    //get source entity
    std::vector< FieldVector< typename SourceGridViewType::ctype, SourceGridViewType::dimension> > global_point(1);
    global_point[0] = xx;
    const auto source_entity_ptr = entity_search(global_point);
    assert(source_entity_ptr.size() == 1);
    const auto& source_entity_unique_ptr = source_entity_ptr[0];
    if(source_entity_unique_ptr){
      const auto source_entity_ptr1 = *source_entity_unique_ptr;
      const auto& source_entity = *source_entity_ptr1;
      //evaluate source function
      const auto local_source_point = source_entity.geometry().local(xx);
      const auto local_source = source_function_.local_function(source_entity);
      local_source->evaluate(local_source_point, ret);
    }
  }

  virtual void jacobian(const DomainType& xx, JacobianRangeType& ret) const override final
  {
    //create entity search in the source grid view
    typedef typename FunctionImp::SpaceType::GridViewType SourceGridViewType;
    typedef Stuff::Grid::EntityInlevelSearch< SourceGridViewType > EntitySearch;
    EntitySearch entity_search(source_function_.space().grid_view());
    //get source entity
    std::vector< FieldVector< typename SourceGridViewType::ctype, SourceGridViewType::dimension> > global_point(1);
    global_point[0] = xx;
    const auto source_entity_ptr = entity_search(global_point);
    assert(source_entity_ptr.size() == 1);
    const auto& source_entity_unique_ptr = source_entity_ptr[0];
    if(source_entity_unique_ptr){
      const auto source_entity_ptr1 = *source_entity_unique_ptr;
      const auto& source_entity = *source_entity_ptr1;
      //evaluate jacobain of the source function
      const auto local_source_point = source_entity.geometry().local(xx);
      const auto local_source = source_function_.local_function(source_entity);
      local_source->jacobian(local_source_point, ret);
    }
  }

private:
const FunctionImp&   source_function_;
const RangeGridView& range_grid_view_;
}; //class ProlongedFunction


} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_DISCRETEFUNCTION_PROLONGED_HH
