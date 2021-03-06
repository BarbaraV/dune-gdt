// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_CURLCURL_HH
#define DUNE_GDT_OPERATORS_CURLCURL_HH

#include <type_traits>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/la/container/interfaces.hh>

#include <dune/gdt/spaces/interface.hh>
#include <dune/gdt/localevaluation/curlcurl.hh>
#include <dune/gdt/localevaluation/product.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/system.hh>

#include "base.hh"

namespace Dune {
namespace GDT {
namespace Operators {

//forward
template< class FunctionType, class MatrixImp, class SourceSpaceImp,
          class RangeSpaceImp = SourceSpaceImp,
          class GridViewImp = typename SourceSpaceImp::GridViewType >
class CurlCurl;



namespace internal {

/** \brief Traits for the CurlCurl-operator
 *
 *\sa CurlCurl
 */
template< class FunctionType, class MatrixImp, class SourceSpaceImp,
          class RangeSpaceImp, class GridViewImp >
class CurlCurlTraits
{
  static_assert(Stuff::is_localizable_function< FunctionType >::value,
                "FunctionType has to be derived from Stuff::LocalizableFunctionInterface!");
  static_assert(Stuff::LA::is_matrix< MatrixImp >::value,
                "MatrixImp has to be derived from Stuff::LA::MatrixInterface!");
  static_assert(is_space< SourceSpaceImp >::value, "SourceSpaceImp has to be derived from SpaceInterface!");
  static_assert(is_space< RangeSpaceImp >::value, "RangeSpaceImp has to be derived from SpaceInterface!");
public:
  typedef CurlCurl< FunctionType, MatrixImp, SourceSpaceImp, RangeSpaceImp, GridViewImp >
          derived_type;
  typedef MatrixImp      MatrixType;
  typedef SourceSpaceImp SourceSpaceType;
  typedef RangeSpaceImp  RangeSpaceType;
  typedef GridViewImp    GridViewType;
}; //class CurlCurlTraits

} //namespace internal


/** \brief Implements a (global) curlcurl operator
 * \note only scalar parameters are supported for sure
 *
 * \tparam FunctionType Type of the parameter function
 * \tparam MatrixImp Type for the system matrix everything is assembled in
 * \tparam SourceSpaceImp Type of the ansatz space
 * \tparam RangeSpaceImp Type of the test space
 * \tparam GridViewImp Type of the grid
 */
template< class FunctionType, class MatrixImp, class SourceSpaceImp,
          class RangeSpaceImp, class GridViewImp >
class CurlCurl
  : Stuff::Common::StorageProvider< MatrixImp >
  , public Operators::MatrixBased< internal::CurlCurlTraits< FunctionType, MatrixImp,
                                                             SourceSpaceImp, RangeSpaceImp, GridViewImp > >
  , public SystemAssembler< RangeSpaceImp, GridViewImp, SourceSpaceImp >
{
  typedef Stuff::Common::StorageProvider< MatrixImp >                                        StorageProvider;
  typedef SystemAssembler< RangeSpaceImp, GridViewImp, SourceSpaceImp >                      AssemblerBaseType;
  typedef Operators::MatrixBased< internal::CurlCurlTraits< FunctionType, MatrixImp, SourceSpaceImp,
                                                            RangeSpaceImp, GridViewImp > >   OperatorBaseType;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::CurlCurl< FunctionType > >         LocalCurlOperatorType;
  typedef LocalAssembler::Codim0Matrix< LocalCurlOperatorType >                              LocalCurlAssemblerType;
  public:
    typedef internal::CurlCurlTraits< FunctionType, MatrixImp, SourceSpaceImp, RangeSpaceImp, GridViewImp>
        Traits;

  typedef typename Traits::MatrixType      MatrixType;
  typedef typename Traits::SourceSpaceType SourceSpaceType;
  typedef typename Traits::RangeSpaceType  RangeSpaceType;
  typedef typename Traits::GridViewType    GridViewType;

  using OperatorBaseType::pattern;

  static Stuff::LA::SparsityPatternDefault pattern(const RangeSpaceType& range_space,
                                                   const SourceSpaceType& source_space,
                                                   const GridViewType& grid_view)
  {
    return range_space.compute_face_and_volume_pattern(grid_view, source_space);
  }

  CurlCurl(const FunctionType& mu,
           MatrixType& mtrx,
           const SourceSpaceType& src_space,
           const RangeSpaceType& rng_space,
           const GridViewType& grid_view)
    : StorageProvider(mtrx)
    , OperatorBaseType(this->storage_access(), src_space, rng_space, grid_view)
    , AssemblerBaseType(rng_space, grid_view, src_space)
    , mu_(mu)
    , local_curl_operator_(mu_)
    , local_curl_assembler_(local_curl_operator_)
    , assembled_(false)
  {
    setup();
  }

  CurlCurl(const FunctionType& mu,
           const SourceSpaceType& src_space,
           const RangeSpaceType& rng_space,
           const GridViewType& grid_view)
    : StorageProvider(new MatrixType(rng_space.mapper().size(),
                                     src_space.mapper().size(),
                                     pattern(rng_space, src_space, grid_view)))
    , OperatorBaseType(this->storage_access(), src_space, rng_space, grid_view)
    , AssemblerBaseType(rng_space, grid_view, src_space)
    , mu_(mu)
    , local_curl_operator_(mu_)
    , local_curl_assembler_(local_curl_operator_)
    , assembled_(false)
  {
    setup();
  }

  CurlCurl(const FunctionType& mu,
           MatrixType& mtrx,
           const SourceSpaceType& src_space,
           const RangeSpaceType& rng_space)
    : StorageProvider(mtrx)
    , OperatorBaseType(this->storage_access(), src_space, rng_space)
    , AssemblerBaseType(rng_space, src_space)
    , mu_(mu)
    , local_curl_operator_(mu_)
    , local_curl_assembler_(local_curl_operator_)
    , assembled_(false)
  {
    setup();
  }

  CurlCurl(const FunctionType& mu,
           const SourceSpaceType& src_space,
           const RangeSpaceType& rng_space)
    : StorageProvider(new MatrixType(rng_space.mapper().size(),
                                     src_space.mapper().size(),
                                     pattern(rng_space, src_space)))
    , OperatorBaseType(this->storage_access(), src_space, rng_space)
    , AssemblerBaseType(rng_space, src_space)
    , mu_(mu)
    , local_curl_operator_(mu_)
    , local_curl_assembler_(local_curl_operator_)
    , assembled_(false)
  {
    setup();
  }

  CurlCurl(const FunctionType& mu,
           MatrixType& mtrx,
           const SourceSpaceType& src_space)
    : StorageProvider(mtrx)
    , OperatorBaseType(this->storage_access(), src_space)
    , AssemblerBaseType(src_space)
    , mu_(mu)
    , local_curl_operator_(mu_)
    , local_curl_assembler_(local_curl_operator_)
    , assembled_(false)
  {
    setup();
  }

  CurlCurl(const FunctionType& mu,
           const SourceSpaceType& src_space)
    : StorageProvider(new MatrixType(src_space.mapper().size(), src_space.mapper().size(),
                                     pattern(src_space)))
    , OperatorBaseType(this->storage_access(), src_space)
    , AssemblerBaseType(src_space)
    , mu_(mu)
    , local_curl_operator_(mu_)
    , local_curl_assembler_(local_curl_operator_)
    , assembled_(false)
  {
    setup();
  }

  virtual ~CurlCurl() {}

  virtual void assemble() override final
  {
    if(!assembled_) {
      AssemblerBaseType::assemble(true);
      assembled_ = true;
    }
  } // ... assemble(...)


private:
  void setup()
  {
    this->add(local_curl_assembler_, this->matrix());
  } //... setup()

  const FunctionType& mu_;
  const LocalCurlOperatorType local_curl_operator_;
  const LocalCurlAssemblerType local_curl_assembler_;
  bool assembled_;
};  //class CurlCurl


} //namespace Operators
} //namespace GDT
} //namespace Dune




#endif // DUNE_GDT_OPERATORS_CURLCURL_HH
