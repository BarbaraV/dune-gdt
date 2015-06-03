// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_PRODUCTS_HCURL_INTERNAL_HH
#define DUNE_GDT_PRODUCTS_HCURL_INTERNAL_HH

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/functions/constant.hh>

#include "curlcurl-internal.hh"

namespace Dune {
namespace GDT {
namespace Products {
namespace internal {


/**
 * \brief Base class for all semi Hcurl products.
 *
 *        This class is implemented using CurlcurlBase with a Stuff::Functions::Constant of value 1 as the weight.
 * \note  Most likely you do not want to use this class directly, but Products::HcurlSemiLocalizable, Products::HcurlSemiAssemblable
 *        or Products::HcurlSemi instead!
 */
template< class GV, class FieldImp >
class HcurlSemiBase
  : DSC::ConstStorageProvider
        < Stuff::Functions::Constant
              < typename GV::template Codim< 0 >::Entity, typename GV::ctype, GV::dimension, FieldImp, 1 > >
  , public CurlcurlBase< GV,
                           Stuff::Functions::Constant< typename GV::template Codim< 0 >::Entity,
                                                       typename GV::ctype,
                                                       GV::dimension,
                                                       FieldImp,
                                                       1 >,
                           FieldImp >
{
  typedef DSC::ConstStorageProvider< Stuff::Functions::Constant
          < typename GV::template Codim< 0 >::Entity, typename GV::ctype, GV::dimension, FieldImp, 1 > >
                                 StorageBaseType;
  typedef CurlcurlBase< GV, Stuff::Functions::Constant
          < typename GV::template Codim< 0 >::Entity, typename GV::ctype, GV::dimension, FieldImp, 1 >, FieldImp >
                                 CurlcurlBaseType;
  typedef HcurlSemiBase< GV, FieldImp > ThisType;

public:
  HcurlSemiBase(const size_t over_integrate = 0)
    : StorageBaseType(new typename CurlcurlBaseType::FunctionType(1))
    , CurlcurlBaseType(this->storage_access(), over_integrate)
    , over_integrate_(over_integrate)
  {}

  /**
   * \note We need the manual copy ctor bc of the Stuff::Common::ConstStorageProvider
   */
  HcurlSemiBase(const ThisType& other)
    : StorageBaseType(new typename CurlcurlBaseType::FunctionType(1))
    , CurlcurlBaseType(this->storage_access(), other.over_integrate_)
    , over_integrate_(other.over_integrate_)
  {}

private:
  const size_t over_integrate_; //!< needed to provide manual copy ctor
}; // HcurlSemiBase


} // namespace internal
} // namespace Products
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_PRODUCTS_HCURL_INTERNAL_HH
