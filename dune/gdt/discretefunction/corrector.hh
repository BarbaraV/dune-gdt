#ifndef DUNE_GDT_DISCRETEFUNCTION_CORRECTOR_HH
#define DUNE_GDT_DISCRETEFUNCTION_CORRECTOR_HH

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/grid/search.hh>

#include <dune/gdt/discretefunction/default.hh>

namespace Dune {
namespace GDT {


//forward
template< class CoarseFunctionImp, class MicroFunctionImp >
class PeriodicCorrectorLocal;

/** class to represent the (discrete) corrector in an HMM from given macroscopic and cell problem solutions
 * \tparam CoarseFunctionImp Type of the macroscopic solution
 * \tparam MicroFunctionImp Type of the (discrete) solutions to the cell problem
 * \note this only works for periodic problem
 */
template< class CoarseFunctionImp, class MicroFunctionImp >
class PeriodicCorrector {
public:
  static_assert(Dune::GDT::is_const_discrete_function< CoarseFunctionImp >::value, "Macro Function has to be a discrete function");
  static_assert(Dune::GDT::is_discrete_function< MicroFunctionImp >::value, "Functiontype for cell solutions has to be discrete");

  typedef typename CoarseFunctionImp::EntityType CoarseEntityType;
  typedef typename CoarseFunctionImp::DomainType CoarseDomainType;
  typedef typename MicroFunctionImp::EntityType  FineEntityType;

  static_assert(std::is_same< typename CoarseFunctionImp::DomainFieldType, typename MicroFunctionImp::DomainFieldType >::value,
                "DomainFieldType has to be the same for macro and micro part");
  static_assert(CoarseFunctionImp::dimDomain == MicroFunctionImp::dimDomain, "Dimensions do not match");

  typedef typename CoarseFunctionImp::DomainFieldType DomainFieldType;
  static const size_t                                 dimDomain = CoarseFunctionImp::dimDomain;

  typedef PeriodicCorrectorLocal< CoarseFunctionImp, MicroFunctionImp > LocalfunctionType;

  PeriodicCorrector(const std::vector< CoarseFunctionImp >& macro_part,
                    const std::vector< std::vector< MicroFunctionImp > >& cell_solutions,
                    const std::string& type)
    : macro_part_(macro_part)
    , cell_solutions_(cell_solutions)
    , type_(type)
  {}

  PeriodicCorrector(const typename CoarseFunctionImp::SpaceType& coarse_space,
                    const typename MicroFunctionImp::SpaceType& fine_space,
                    const std::string& type)
    : macro_part_(2, CoarseFunctionImp(coarse_space))
    , cell_solutions_(dimDomain, std::vector< MicroFunctionImp >(2, MicroFunctionImp(fine_space)))
    , type_(type)
  {}

  std::unique_ptr< LocalfunctionType > local_function(const CoarseEntityType& coarse_entity)
  {
    return DSC::make_unique< LocalfunctionType >(macro_part_, cell_solutions_, type_, coarse_entity);
  }

  const typename CoarseFunctionImp::SpaceType& coarse_space() const
  {
    return macro_part_[0].space();
  }

  const typename MicroFunctionImp::SpaceType& cell_space() const
  {
    return cell_solutions_[0][0].space();
  }

  const std::vector< std::vector< MicroFunctionImp > > cell_solutions() const
  {
    return cell_solutions_;
  }

  const std::vector< CoarseFunctionImp > macro_function() const
  {
    return macro_part_;
  }

private:
  const std::vector< CoarseFunctionImp >               macro_part_;
  const std::vector< std::vector< MicroFunctionImp > > cell_solutions_;
  const std::string                                    type_;
}; //class PeriodicCorrector


/** class to represent the corrector in an HMM on a given entity of the macroscopic domain
 * \sa PeriodicCorrector
 * \todo implement this for other types of HMM than curl-curl-problems
 */
template< class CoarseFunctionImp, class MicroFunctionImp >
class PeriodicCorrectorLocal {
public:
  static_assert(Dune::Stuff::is_localizable_function< CoarseFunctionImp >::value, "Macro Function has to be localizable");
  static_assert(Dune::Stuff::is_localizable_function< MicroFunctionImp >::value, "Functiontype for cell solutions has to be localizable");

  typedef typename CoarseFunctionImp::EntityType CoarseEntityType;
  typedef typename CoarseFunctionImp::DomainType CoarseDomainType;
  typedef typename MicroFunctionImp::EntityType  FineEntityType;

  static_assert(std::is_same< typename CoarseFunctionImp::DomainFieldType, typename MicroFunctionImp::DomainFieldType >::value,
                "DomainFieldType has to be the same for macro and micro part");
  static_assert(CoarseFunctionImp::dimDomain == MicroFunctionImp::dimDomain, "Dimensions do not match");

  typedef typename CoarseFunctionImp::DomainFieldType DomainFieldType;
  static const size_t                                 dimDomain = CoarseFunctionImp::dimDomain;

  PeriodicCorrectorLocal(const std::vector< CoarseFunctionImp > & macro_part,
                         const std::vector< std::vector< MicroFunctionImp > >& cell_solutions,
                         const std::string& type,
                         const CoarseEntityType& coarse_entity)
    : local_macro_part_(macro_part.size())
    , cell_solutions_(cell_solutions)
    , type_(type)
  {
    for (size_t ii = 0; ii < macro_part.size(); ++ii)
      local_macro_part_[ii] = std::move(macro_part[ii].local_function(coarse_entity));
  }

  size_t order() const
  {
    if (type_ == "id")
      return local_macro_part_[0]->order();
    if (type_ == "curl")
      return boost::numeric_cast< size_t >(std::max(ssize_t(local_macro_part_[0]->order() -1), ssize_t(0)));
    else
      DUNE_THROW(Dune::NotImplemented, "This type of corrector needs to be implemented");
  }

  /**
   * @brief evaluate evaluates the corrector with respect to the macroscopic variable x
   * @param xx the (local) point in which the corrector is evaluated
   * @param ret a vector (2 items, for real and imaginary part) of (discrete) cell functions, into which the actusl corrector is evaluated
   *
   * the method takes the macroscopic part and the cell solutions and builds the corrector from those, depeending on the type_:
   * for "curl", the curl of the macrosocpic part is multiplied (componentwise) with the correcpsonding cell solution and then summed up
   * for "id", the evaluateion of the macroscopic patrt is nm=multiplied (componentwise) with the correspoding cell solution and then summed up
   */
  void evaluate(const CoarseDomainType& xx, std::vector< MicroFunctionImp >& ret) const
  {
    assert(local_macro_part_.size() > 1);
    assert(ret.size() > 1);
    //clear vectors
    ret[0].vector() *= 0;
    ret[1].vector() *= 0;
    if (type_ == "id") {
      auto macro_real = local_macro_part_[0]->evaluate(xx);
      auto macro_imag = local_macro_part_[1]->evaluate(xx);
      assert(macro_real.size() == cell_solutions_.size());
      for (size_t ii = 0; ii < cell_solutions_.size(); ++ii) {
        ret[0].vector().axpy(macro_real[ii], cell_solutions_[ii][0].vector());
        ret[1].vector().axpy(macro_imag[ii], cell_solutions_[ii][0].vector());
        if (cell_solutions_[ii].size() > 1) {
          ret[0].vector().axpy(-1*macro_imag[ii], cell_solutions_[ii][1].vector());
          ret[1].vector().axpy(macro_real[ii], cell_solutions_[ii][1].vector());
        }
      }
    }
    if (type_ == "curl") {
      auto macro_real = local_macro_part_[0]->jacobian(xx);
      auto macro_imag = local_macro_part_[1]->jacobian(xx);
      typename CoarseFunctionImp::RangeType macro_curl_real(0);
      typename CoarseFunctionImp::RangeType macro_curl_imag(0);
      macro_curl_real[0] = macro_real[2][1] - macro_real[1][2];
      macro_curl_real[1] = macro_real[0][2] - macro_real[2][0];
      macro_curl_real[2] = macro_real[1][0] - macro_real[0][1];
      macro_curl_imag[0] = macro_imag[2][1] - macro_imag[1][2];
      macro_curl_imag[1] = macro_imag[0][2] - macro_imag[2][0];
      macro_curl_imag[2] = macro_imag[1][0] - macro_imag[0][1];
      assert(macro_curl_real.size() == cell_solutions_.size());
      for (size_t ii = 0; ii < cell_solutions_.size(); ++ii) {
        ret[0].vector().axpy(macro_curl_real[ii], cell_solutions_[ii][0].vector());
        ret[1].vector().axpy(macro_curl_imag[ii], cell_solutions_[ii][0].vector());
        if (cell_solutions_[ii].size() > 1) {
          ret[0].vector().axpy(-1*macro_curl_imag[ii], cell_solutions_[ii][1].vector());
          ret[1].vector().axpy(macro_curl_real[ii], cell_solutions_[ii][1].vector());
        }
      }
    }
  } //evaluate

private:
  std::vector< std::unique_ptr< typename CoarseFunctionImp::LocalfunctionType > > local_macro_part_;
  std::vector< std::vector< MicroFunctionImp > > cell_solutions_;
  const std::string type_;
}; //class PeriodicCorrectorLocal


/** class to describe the zeroth order approximation to the heterogeneous solution, computed form the macroscopic part and its correctors of the HMM for a curl-curl-problem
 *
 * \tparam CoarseFunctionImp Type for the macroscopic part of the HMM solution
 * \tparam FineFunctionCurlImp Type for corrector to the curl of the HMM solution
 * \tparam FineFunctionIdImp Type for the corrector to the HMM solution itself (identity part)
 * \note this class describes a GloablFunction and its evaluation and jacobian use the EntityInlevelSearch, which may not be optimal
 */
template< class CoarseFunctionImp, class FineFunctionCurlImp, class FineFunctionIdImp >
class DeltaCorrectorCurl
  : public Dune::Stuff::GlobalFunctionInterface< typename CoarseFunctionImp::EntityType,
                                                 typename CoarseFunctionImp::DomainFieldType,
                                                 CoarseFunctionImp::dimDomain,
                                                 typename CoarseFunctionImp::RangeFieldType,
                                                 CoarseFunctionImp::dimRange >
{
  typedef Dune::Stuff::GlobalFunctionInterface< typename CoarseFunctionImp::EntityType,
                                                typename CoarseFunctionImp::DomainFieldType, CoarseFunctionImp::dimDomain,
                                                typename CoarseFunctionImp::RangeFieldType, CoarseFunctionImp::dimRange > BaseType;
  typedef DeltaCorrectorCurl< CoarseFunctionImp, FineFunctionCurlImp, FineFunctionIdImp >                                 ThisType;

public:
  using typename BaseType::LocalfunctionType;
  using typename BaseType::DomainType;
  using typename BaseType::DomainFieldType;
  using typename BaseType::RangeType;
  using typename BaseType::JacobianRangeType;

  static const bool available = false;

  static std::string static_id()
  {
    return CoarseFunctionImp::static_id() + ".corrector";
  }

  DeltaCorrectorCurl(const std::vector< CoarseFunctionImp >& macro_fct,
                     const std::vector< std::vector< FineFunctionCurlImp > >& curl_cell_solutions,
                     const std::vector< std::vector< FineFunctionIdImp > >& id_cell_solutions,
                     const DomainFieldType delta,
                     const std::string which_part)
    : macro_function_(macro_fct)
    , curl_cell_solutions_(curl_cell_solutions)
    , id_cell_solutions_(id_cell_solutions)
    , delta_(delta)
    , part_(which_part)
  {
    assert(macro_fct.size() > 1);
  }

  DeltaCorrectorCurl(const ThisType& /*other*/) = default;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  virtual std::string type() const override final
  {
    return CoarseFunctionImp::static_id() + ".corrector." + part_;
  }

  virtual std::string name() const override final
  {
    return CoarseFunctionImp::static_id() + ".corrector." + part_;
  }

  virtual size_t order() const override final
  {
    return macro_function_[0].local_function(*macro_function_[0].space().grid_view().template begin<0>())->order();
  }

  /** @brief evaluate evaluates the zeroth order HMM approximation to a heterogeneous curl-curl-problem
   *
   * @param xx global point of the macroscopic computational domain
   * @param ret vector the evaluation is stored in
   * @note only the zero'th order terms are considered, i.e. no delta * K_1 is evaluated
   */
  virtual void evaluate(const DomainType& xx, RangeType& ret) const override final
  {
    //clear ret
    ret *= 0;
    //tmp storage
    std::vector< RangeType > macro_total(macro_function_.size(), RangeType(0));
    //entity search in the macro grid view
    typedef typename CoarseFunctionImp::SpaceType::GridViewType MacroGridViewType;
    Dune::Stuff::Grid::EntityInlevelSearch< MacroGridViewType > entity_search(macro_function_[0].space().grid_view());
    std::vector< Dune::FieldVector< typename MacroGridViewType::ctype, MacroGridViewType::dimension > > global_point(1);
    global_point[0] = xx;
    const auto source_entity_ptr = entity_search(global_point);
    assert(source_entity_ptr.size() == 1);
    const auto& source_entity_unique_ptr = source_entity_ptr[0];
    if(source_entity_unique_ptr) {
      const auto source_entity_ptr1 = *source_entity_unique_ptr;
      const auto& source_entity = *source_entity_ptr1;
      const auto local_source_point = source_entity.geometry().local(xx);
      //evaluate macro function and its jacobian
      for (size_t ii = 0; ii < macro_function_.size(); ++ii)
        macro_function_[ii].local_function(source_entity)->evaluate(local_source_point, macro_total[ii]);
    }
    if (part_ == "real")
      ret += macro_total[0];
    else if (part_ == "imag")
      ret += macro_total[1];
    else
      DUNE_THROW(Dune::NotImplemented, "You can only compute real or imag part");
    //preparation for fine part
    DomainType yy(xx);
    yy /= delta_;
    DomainFieldType intpart;
    for (size_t ii = 0; ii < CoarseFunctionImp::dimDomain; ++ii) {
      auto fracpart = std::modf(yy[ii], &intpart);
      if (fracpart < 0)
        yy[ii] = 1 + fracpart;
      else
        yy[ii] = fracpart;
    }
    //now yy is a (global) point in the unit cube
    //do now the same entity search as before, but for the unit cube grid view
    std::vector< std::vector< typename FineFunctionIdImp::JacobianRangeType > >
      id_cell_jacobian(id_cell_solutions_.size(), std::vector< typename FineFunctionIdImp::JacobianRangeType >(id_cell_solutions_[0].size()));
    typedef typename FineFunctionIdImp::SpaceType::GridViewType FineGridViewType;
    Dune::Stuff::Grid::EntityInlevelSearch< FineGridViewType > entity_search_fine(id_cell_solutions_[0][0].space().grid_view());
    std::vector< Dune::FieldVector< typename FineGridViewType::ctype, FineGridViewType::dimension > > fine_point(1);
    fine_point[0] = yy;
    const auto source_entity_ptr_fine = entity_search_fine(fine_point);
    assert(source_entity_ptr_fine.size() == 1);
    const auto& source_entity_unique_ptr_fine = source_entity_ptr_fine[0];
    if(source_entity_unique_ptr_fine) {
      const auto source_entity_ptr1 = *source_entity_unique_ptr_fine;
      const auto& source_entity = *source_entity_ptr1;
      const auto local_source_point = source_entity.geometry().local(yy);
      //evaluate id cell solutions' jacobian
      for (size_t ii = 0; ii < id_cell_solutions_.size(); ++ii) {
        for (size_t jj = 0; jj < id_cell_solutions_[ii].size(); ++jj)
          id_cell_solutions_[ii][jj].local_function(source_entity)->jacobian(local_source_point, id_cell_jacobian[ii][jj]);
        if (part_ == "real") {
          ret.axpy(macro_total[0][ii], id_cell_jacobian[ii][0][0]); //real*real
          if (id_cell_solutions_[ii].size() > 1)
            ret.axpy(-1*macro_total[1][ii], id_cell_jacobian[ii][1][0]); //-imag*imag
        }
        else if (part_ == "imag") {
          ret.axpy(macro_total[1][ii], id_cell_jacobian[ii][0][0]); //imag*real
          if (id_cell_solutions_[ii].size() > 1)
            ret.axpy(macro_total[0][ii], id_cell_jacobian[ii][1][0]); //real*imag
        }
      }
    }
  } //evaluate


  /** @brief jacobian evaluates the zeroth order HMM approximation to the jacobian of the solution of a heterogeneous curl-curl-problem
   *
   * @param xx global point of the macroscopic computational domain
   * @param matrix vector the evaluation is stored in
   * @note only the zero'th order terms are considered, i.e. the jacobian of K_2 is neglected, which is only correct if this is used for curl computations
   */
  virtual void jacobian(const DomainType& xx, JacobianRangeType& ret) const override final
  {
    //clear ret
    ret *= 0;
    //tmp storage
    std::vector< JacobianRangeType > macro_jacobian(macro_function_.size(), JacobianRangeType(0));
    std::vector< RangeType > macro_curl(macro_function_.size(), RangeType(0));
    //entity search in the macro grid view
    typedef typename CoarseFunctionImp::SpaceType::GridViewType MacroGridViewType;
    Dune::Stuff::Grid::EntityInlevelSearch< MacroGridViewType > entity_search(macro_function_[0].space().grid_view());
    std::vector< Dune::FieldVector< typename MacroGridViewType::ctype, MacroGridViewType::dimension > > global_point(1);
    global_point[0] = xx;
    const auto source_entity_ptr = entity_search(global_point);
    assert(source_entity_ptr.size() == 1);
    const auto& source_entity_unique_ptr = source_entity_ptr[0];
    if(source_entity_unique_ptr) {
      const auto source_entity_ptr1 = *source_entity_unique_ptr;
      const auto& source_entity = *source_entity_ptr1;
      const auto local_source_point = source_entity.geometry().local(xx);
      //evaluate macro function and its jacobian
      for (size_t ii = 0; ii < macro_function_.size(); ++ii) {
        macro_function_[ii].local_function(source_entity)->jacobian(local_source_point, macro_jacobian[ii]);
        macro_curl[ii][0] = macro_jacobian[ii][2][1] - macro_jacobian[ii][1][2];
        macro_curl[ii][1] = macro_jacobian[ii][0][2] - macro_jacobian[ii][2][0];
        macro_curl[ii][2] = macro_jacobian[ii][1][0] - macro_jacobian[ii][0][1];
      }
    }
    if (part_ == "real")
      ret += macro_jacobian[0];
    else if (part_ == "imag")
      ret += macro_jacobian[1];
    else
      DUNE_THROW(Dune::NotImplemented, "You can only compute real or imag part");
    //preparation for fine part
    DomainType yy(xx);
    yy /= delta_;
    DomainFieldType intpart;
    for (size_t ii = 0; ii < CoarseFunctionImp::dimDomain; ++ii) {
      auto fracpart = std::modf(yy[ii], &intpart);
      if (fracpart < 0)
        yy[ii] = 1 + fracpart;
      else
        yy[ii] = fracpart;
    }
    //now yy is a (global) point in the unit cube
    //do now the same entity search as before, but for the unit cube grid view
    std::vector< std::vector< typename FineFunctionCurlImp::JacobianRangeType > >
      curl_cell_jacobian(curl_cell_solutions_.size(), std::vector< typename FineFunctionCurlImp::JacobianRangeType >(curl_cell_solutions_[0].size()));
    typedef typename FineFunctionCurlImp::SpaceType::GridViewType FineGridViewType;
    Dune::Stuff::Grid::EntityInlevelSearch< FineGridViewType > entity_search_fine(curl_cell_solutions_[0][0].space().grid_view());
    std::vector< Dune::FieldVector< typename FineGridViewType::ctype, FineGridViewType::dimension > > fine_point(1);
    fine_point[0] = yy;
    const auto source_entity_ptr_fine = entity_search_fine(fine_point);
    assert(source_entity_ptr_fine.size() == 1);
    const auto& source_entity_unique_ptr_fine = source_entity_ptr_fine[0];
    if(source_entity_unique_ptr_fine) {
      const auto source_entity_ptr1 = *source_entity_unique_ptr_fine;
      const auto& source_entity = *source_entity_ptr1;
      const auto local_source_point = source_entity.geometry().local(yy);
      //evaluate curl cell solutions' jacobian
      for (size_t ii = 0; ii < curl_cell_solutions_.size(); ++ii) {
        for (size_t jj = 0; jj < curl_cell_solutions_[ii].size(); ++jj)
          curl_cell_solutions_[ii][jj].local_function(source_entity)->jacobian(local_source_point, curl_cell_jacobian[ii][jj]);
        if (part_ == "real") {
          ret.axpy(macro_curl[0][ii], curl_cell_jacobian[ii][0]); //real*real
          if (curl_cell_solutions_[ii].size() > 1)
            ret.axpy(-1*macro_curl[1][ii], curl_cell_jacobian[ii][1]); //-imag*imag
        }
        else if (part_ == "imag") {
          ret.axpy(macro_curl[1][ii], curl_cell_jacobian[ii][0]); //imag*real
          if(curl_cell_solutions_[ii].size() > 1)
            ret.axpy(macro_curl[0][ii], curl_cell_jacobian[ii][1]); //real*imag
        }
      }
    }
  } //jacobian


private:
  const std::vector< CoarseFunctionImp >                  macro_function_;
  const std::vector< std::vector< FineFunctionCurlImp > > curl_cell_solutions_;
  const std::vector< std::vector< FineFunctionIdImp > >   id_cell_solutions_;
  const DomainFieldType                                   delta_;
  const std::string                                       part_;
};  //class DeltaCorrectorCurl


/** class to describe the zeroth order approximation to the heterogeneous solutionfor a highly heterogeneous Helmholtz problem
 *
 * \tparam CoarseFunctionImp Type for the macroscopic part of the HMM solution
 * \tparam FineFunctionGradImp Type for corrector to the gradient of the HMM solution
 * \tparam FineFunctionIdImp Type for the corrector to the HMM solution itself (identity part, in the inclusions)
 * \note this class describes a GlobalFunction and its evaluation and jacobian use the EntityInlevelSearch, which may not be optimal
 */
template< class CoarseFunctionImp, class FineFunctionGradImp, class FineFunctionIdImp >
class DeltaCorrectorHelmholtz
  : public Dune::Stuff::GlobalFunctionInterface< typename CoarseFunctionImp::EntityType,
                                                 typename CoarseFunctionImp::DomainFieldType,
                                                 CoarseFunctionImp::dimDomain,
                                                 typename CoarseFunctionImp::RangeFieldType,
                                                 CoarseFunctionImp::dimRange >
{
  typedef Dune::Stuff::GlobalFunctionInterface< typename CoarseFunctionImp::EntityType,
                                                typename CoarseFunctionImp::DomainFieldType, CoarseFunctionImp::dimDomain,
                                                typename CoarseFunctionImp::RangeFieldType, CoarseFunctionImp::dimRange > BaseType;
  typedef DeltaCorrectorHelmholtz< CoarseFunctionImp, FineFunctionGradImp, FineFunctionIdImp >                            ThisType;

public:
  using typename BaseType::LocalfunctionType;
  using typename BaseType::DomainType;
  using typename BaseType::DomainFieldType;
  using typename BaseType::RangeType;
  using typename BaseType::JacobianRangeType;

  static_assert(Dune::GDT::is_discrete_function< CoarseFunctionImp >::value, "macroscopic function has to be a discrete function");
  static_assert(Dune::GDT::is_discrete_function< FineFunctionGradImp >::value, "elliptic cell solution has to be a discrete function");
  static_assert(Dune::GDT::is_discrete_function< FineFunctionIdImp >::value, "helmholtz cell solution has to be a discrete function");

  typedef std::function< bool(const typename CoarseFunctionImp::SpaceType::GridViewType&, const typename CoarseFunctionImp::EntityType&) > MacroFilterType;
  typedef std::function< bool(const typename FineFunctionGradImp::SpaceType::GridViewType&, const typename FineFunctionGradImp::EntityType&) > FineFilterType;

  static const bool available = false;

  static std::string static_id()
  {
    return CoarseFunctionImp::static_id() + ".corrector";
  }

  DeltaCorrectorHelmholtz(const std::vector< CoarseFunctionImp >& macro_fct,
                          const std::vector< std::vector< FineFunctionGradImp > >& elliptic_cell_solutions,
                          const std::vector< std::vector< FineFunctionIdImp > >& incl_cell_solutions,
                          MacroFilterType filter_scatterer,
                          FineFilterType filter_inclusion,
                          const double& wavenumber,
                          const DomainFieldType delta,
                          const std::string which_part)
    : macro_function_(macro_fct)
    , elliptic_cell_solutions_(elliptic_cell_solutions)
    , inclusion_cell_solutions_(incl_cell_solutions)
    , filter_scatterer_(filter_scatterer)
    , filter_inclusion_(filter_inclusion)
    , wavenumber_(wavenumber)
    , delta_(delta)
    , part_(which_part)
  {
    assert(macro_fct.size() > 1);
  }

  DeltaCorrectorHelmholtz(const ThisType& /*other*/) = default;

  ThisType& operator=(const ThisType& /*other*/) = delete;

  virtual std::string type() const override final
  {
    return CoarseFunctionImp::static_id() + ".corrector." + part_;
  }

  virtual std::string name() const override final
  {
    return CoarseFunctionImp::static_id() + ".corrector." + part_;
  }

  virtual size_t order() const override final
  {
    return macro_function_[0].local_function(*macro_function_[0].space().grid_view().template begin<0>())->order();
  }

  /** @brief evaluate evaluates the zeroth order HMM approximation to a highly heterogeneous Helmholtz problem
   *
   * @param xx global point of the macroscopic computational domain
   * @param ret vector the evaluation is stored in
   * @note only the zero'th order terms are considered
   */
  virtual void evaluate(const DomainType& xx, RangeType& ret) const override final
  {
    //clear ret
    ret *= 0;
    //tmp storage
    std::vector< RangeType > macro_total(macro_function_.size(), RangeType(0));
    bool inside_scatterer;
    //bool outside_inclusion;
    //entity search in the macro grid view
    typedef typename CoarseFunctionImp::SpaceType::GridViewType MacroGridViewType;
    Dune::Stuff::Grid::EntityInlevelSearch< MacroGridViewType > entity_search(macro_function_[0].space().grid_view());
    std::vector< Dune::FieldVector< typename MacroGridViewType::ctype, MacroGridViewType::dimension > > global_point(1);
    global_point[0] = xx;
    const auto source_entity_ptr = entity_search(global_point);
    assert(source_entity_ptr.size() == 1);
    const auto& source_entity_unique_ptr = source_entity_ptr[0];
    if(source_entity_unique_ptr) {
      const auto source_entity_ptr1 = *source_entity_unique_ptr;
      const auto& source_entity = *source_entity_ptr1;
      inside_scatterer = filter_scatterer_(macro_function_[0].space().grid_view(), source_entity);
      const auto local_source_point = source_entity.geometry().local(xx);
      //evaluate macro function and its jacobian
      for (size_t ii = 0; ii < macro_function_.size(); ++ii)
        macro_function_[ii].local_function(source_entity)->evaluate(local_source_point, macro_total[ii]);
    }
    if (part_ == "real")
      ret += macro_total[0];
    else if (part_ == "imag")
      ret += macro_total[1];
    else
      DUNE_THROW(Dune::NotImplemented, "You can only compute real or imag part");
    if (inside_scatterer) {
      //preparation for fine part
      DomainType yy(xx);
      yy /= delta_;
      DomainFieldType intpart;
      for (size_t ii = 0; ii < CoarseFunctionImp::dimDomain; ++ii) {
        auto fracpart = std::modf(yy[ii], &intpart);
        if (fracpart < 0)
          yy[ii] = 1 + fracpart;
        else
          yy[ii] = fracpart;
      }
      //now yy is a (global) point in the unit cube
      //do now the same entity search as before, but for the grid view of the inclusions
      std::vector< std::vector< typename FineFunctionIdImp::RangeType > >
        incl_cell_evaluation(inclusion_cell_solutions_.size(), std::vector< typename FineFunctionIdImp::RangeType >(inclusion_cell_solutions_[0].size()));
      typedef typename FineFunctionIdImp::SpaceType::GridViewType FineGridViewType;
      Dune::Stuff::Grid::EntityInlevelSearch< FineGridViewType > entity_search_fine(inclusion_cell_solutions_[0][0].space().grid_view());
      std::vector< Dune::FieldVector< typename FineGridViewType::ctype, FineGridViewType::dimension > > fine_point(1);
      fine_point[0] = yy;
      const auto source_entity_ptr_fine = entity_search_fine(fine_point);
      assert(source_entity_ptr_fine.size() == 1);
      const auto& source_entity_unique_ptr_fine = source_entity_ptr_fine[0];
      if(source_entity_unique_ptr_fine) {
        const auto source_entity_ptr1 = *source_entity_unique_ptr_fine;
        const auto& source_entity = *source_entity_ptr1;
        const auto local_source_point = source_entity.geometry().local(yy);
        //evaluate id cell solutions
        assert(inclusion_cell_solutions_.size() == 1);
        for (size_t jj = 0; jj < inclusion_cell_solutions_[0].size(); ++jj)
          inclusion_cell_solutions_[0][jj].local_function(source_entity)->evaluate(local_source_point, incl_cell_evaluation[0][jj]);
        if (part_ == "real") {
          ret += wavenumber_ * wavenumber_ * macro_total[0] * incl_cell_evaluation[0][0]; //real*real
          if (inclusion_cell_solutions_[0].size() > 1)
            ret -= wavenumber_ * wavenumber_ *macro_total[1] * incl_cell_evaluation[0][1]; //-imag*imag
        }
        else if (part_ == "imag") {
          ret += wavenumber_ * wavenumber_ * macro_total[1] * incl_cell_evaluation[0][0]; //imag*real
          if (inclusion_cell_solutions_[0].size() > 1)
            ret += wavenumber_ * wavenumber_ * macro_total[0] * incl_cell_evaluation[0][1]; //real*imag
        }
      }
    } //inside scatterer
  } //evaluate


  /** @brief jacobian evaluates the zeroth order HMM approximation to the jacobian of the solution of a highly heterogeneous helmholtz problem
   *
   * @param xx global point of the macroscopic computational domain
   * @param ret matrix vector the evaluation is stored in
   * @note only the zero'th order terms are considered
   */
  virtual void jacobian(const DomainType& xx, JacobianRangeType& ret) const override final
  {
    //clear ret
    ret *= 0;
    //tmp storage
    std::vector< JacobianRangeType > macro_jacobian(macro_function_.size(), JacobianRangeType(0));
    std::vector< RangeType > macro_total(macro_function_.size(), RangeType(0));
    bool inside_scatterer;
    //entity search in the macro grid view
    typedef typename CoarseFunctionImp::SpaceType::GridViewType MacroGridViewType;
    Dune::Stuff::Grid::EntityInlevelSearch< MacroGridViewType > entity_search(macro_function_[0].space().grid_view());
    std::vector< Dune::FieldVector< typename MacroGridViewType::ctype, MacroGridViewType::dimension > > global_point(1);
    global_point[0] = xx;
    const auto source_entity_ptr = entity_search(global_point);
    assert(source_entity_ptr.size() == 1);
    const auto& source_entity_unique_ptr = source_entity_ptr[0];
    if(source_entity_unique_ptr) {
      const auto source_entity_ptr1 = *source_entity_unique_ptr;
      const auto& source_entity = *source_entity_ptr1;
      const auto local_source_point = source_entity.geometry().local(xx);
      inside_scatterer = filter_scatterer_(macro_function_[0].space().grid_view(), source_entity);
      //evaluate macro function and its jacobian
      for (size_t ii = 0; ii < macro_function_.size(); ++ii) {
        macro_function_[ii].local_function(source_entity)->jacobian(local_source_point, macro_jacobian[ii]);
        macro_function_[ii].local_function(source_entity)->evaluate(local_source_point, macro_total[ii]);
      }
    }
    if (part_ == "real")
      ret += macro_jacobian[0];
    else if (part_ == "imag")
      ret += macro_jacobian[1];
    else
      DUNE_THROW(Dune::NotImplemented, "You can only compute real or imag part");
    if (inside_scatterer) {
      //preparation for fine part
      DomainType yy(xx);
      yy /= delta_;
      DomainFieldType intpart;
      for (size_t ii = 0; ii < CoarseFunctionImp::dimDomain; ++ii) {
        auto fracpart = std::modf(yy[ii], &intpart);
        if (fracpart < 0)
          yy[ii] = 1 + fracpart;
        else
          yy[ii] = fracpart;
      }
      //now yy is a (global) point in the unit cube
      //do now the same entity search as before, but for the unit cube grid view
      std::vector< std::vector< typename FineFunctionGradImp::JacobianRangeType > >
        ell_cell_jacobian(elliptic_cell_solutions_.size(), std::vector< typename FineFunctionGradImp::JacobianRangeType >(elliptic_cell_solutions_[0].size()));
      std::vector< std::vector< typename FineFunctionIdImp::JacobianRangeType > >
        incl_cell_jacobian(inclusion_cell_solutions_.size(), std::vector< typename FineFunctionIdImp::JacobianRangeType >(inclusion_cell_solutions_[0].size()));
     // std::vector< std::vector< typename FineFunctionIdImp::RangeType > >
     //   incl_cell_eval(inclusion_cell_solutions_.size(), std::vector< typename FineFunctionIdImp::RangeType >(inclusion_cell_solutions_[0].size()));
      bool inside_inclusion;
      typedef typename FineFunctionGradImp::SpaceType::GridViewType FineGridViewType;
      Dune::Stuff::Grid::EntityInlevelSearch< FineGridViewType > entity_search_fine(elliptic_cell_solutions_[0][0].space().grid_view());
      std::vector< Dune::FieldVector< typename FineGridViewType::ctype, FineGridViewType::dimension > > fine_point(1);
      fine_point[0] = yy;
      const auto source_entity_ptr_fine = entity_search_fine(fine_point);
      assert(source_entity_ptr_fine.size() == 1);
      const auto& source_entity_unique_ptr_fine = source_entity_ptr_fine[0];
      if(source_entity_unique_ptr_fine) {
        const auto source_entity_ptr1 = *source_entity_unique_ptr_fine;
        const auto& source_entity = *source_entity_ptr1;
        const auto local_source_point = source_entity.geometry().local(yy);
        inside_inclusion = !filter_inclusion_(elliptic_cell_solutions_[0][0].space().grid_view(), source_entity);
        if (!inside_inclusion) {
          //evaluate elliptic cell solutions' jacobian
          for (size_t ii = 0; ii < elliptic_cell_solutions_.size(); ++ii) {
            for (size_t jj = 0; jj < elliptic_cell_solutions_[ii].size(); ++jj)
              elliptic_cell_solutions_[ii][jj].local_function(source_entity)->jacobian(local_source_point, ell_cell_jacobian[ii][jj]);
            if (part_ == "real") {
              ret.axpy(macro_jacobian[0][0][ii], ell_cell_jacobian[ii][0]); //real*real
              if (elliptic_cell_solutions_[ii].size() > 1)
                ret.axpy(-1*macro_jacobian[1][0][ii], ell_cell_jacobian[ii][1]); //-imag*imag
            }
            else if (part_ == "imag") {
              ret.axpy(macro_jacobian[1][0][ii], ell_cell_jacobian[ii][0]); //imag*real
              if(elliptic_cell_solutions_[ii].size() > 1)
                ret.axpy(macro_jacobian[0][0][ii], ell_cell_jacobian[ii][1]); //real*imag
            }
          }
        } //only if we are not in the inclusions
      }
      if (inside_inclusion) {
        //now do an entitysearch in the inclusion grid_view
        typedef typename FineFunctionIdImp::SpaceType::GridViewType FineInclGridViewType;
        Dune::Stuff::Grid::EntityInlevelSearch< FineInclGridViewType > entity_search_fine_incl(inclusion_cell_solutions_[0][0].space().grid_view());
        std::vector< Dune::FieldVector< typename FineInclGridViewType::ctype, FineInclGridViewType::dimension > > fine_point_incl(1);
        fine_point_incl[0] = yy;
        const auto source_entity_ptr_fine_incl = entity_search_fine_incl(fine_point_incl);
        assert(source_entity_ptr_fine_incl.size() == 1);
        const auto& source_entity_unique_ptr_fine_incl = source_entity_ptr_fine_incl[0];
        if(source_entity_unique_ptr_fine_incl) {
          const auto source_entity_ptr1 = *source_entity_unique_ptr_fine_incl;
          const auto& source_entity = *source_entity_ptr1;
          const auto local_source_point = source_entity.geometry().local(yy);
          assert(inclusion_cell_solutions_.size() == 1);
          for (size_t jj = 0; jj < inclusion_cell_solutions_[0].size(); ++jj) {
            inclusion_cell_solutions_[0][jj].local_function(source_entity)->jacobian(local_source_point, incl_cell_jacobian[0][jj]);
            //inclusion_cell_solutions_[0][jj].local_function(source_entity)->evaluate(local_source_point, incl_cell_eval[0][jj]);
          }
          if (part_ == "real") {
            ret -= macro_jacobian[0];
            ret.axpy(wavenumber_ * wavenumber_ * macro_total[0]/delta_, incl_cell_jacobian[0][0]); //real*real
            //ret.axpy(wavenumber_ * wavenumber_ * incl_cell_eval[0][0], macro_jacobian[0]); //real*real
            if (inclusion_cell_solutions_[0].size() > 1) {
              ret.axpy(-1* wavenumber_ * wavenumber_ * macro_total[1]/delta_, incl_cell_jacobian[0][1]); //-imag*imag
              //ret.axpy(-1* wavenumber_ * wavenumber_ * incl_cell_eval[0][1], macro_jacobian[1]); //-imag*imag
            }
          }
          else if (part_ == "imag") {
            ret -= macro_jacobian[1];
            ret.axpy(wavenumber_ * wavenumber_ * macro_total[1]/delta_,  incl_cell_jacobian[0][0]); //imag*real
            //ret.axpy(wavenumber_ * wavenumber_ * incl_cell_eval[0][0],  macro_jacobian[1]); //imag*real
            if (inclusion_cell_solutions_[0].size() > 1) {
              ret.axpy(wavenumber_ * wavenumber_ * macro_total[0]/delta_, incl_cell_jacobian[0][1]); //real*imag
              //ret.axpy(wavenumber_ * wavenumber_ * incl_cell_eval[0][1], macro_jacobian[0]); //real*imag
            }
          }
        }
      }  //inside the inclusion
    }
  } //jacobian


private:
  const std::vector< CoarseFunctionImp >                  macro_function_;
  const std::vector< std::vector< FineFunctionGradImp > > elliptic_cell_solutions_;
  const std::vector< std::vector< FineFunctionIdImp > >   inclusion_cell_solutions_;
  MacroFilterType                                         filter_scatterer_;
  FineFilterType                                          filter_inclusion_;
  const DomainFieldType                                   wavenumber_;
  const DomainFieldType                                   delta_;
  const std::string                                       part_;
};  //class DeltaCorrectorHelmholtz



} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_DISCRETEFUNCTION_CORRECTOR_HH
