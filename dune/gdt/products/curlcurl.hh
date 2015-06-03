// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_PRODUCTS_CURLCURL_HH
#define DUNE_GDT_PRODUCTS_CURLCURL_HH

#include "base.hh"
#include "curlcurl-internal.hh"

namespace Dune {
namespace GDT {
namespace Products {


/**
 * \brief A localizable curlcurl product.
 *
 *        Possible ctor signaturer are a combination of the ones from \sa LocalizableBase first and then \sa
 *        internal::CurlcurlBase.
 * \todo  Add more documentation, especially a mathematical definition.
 */
template< class GV, class F, class R, class S = R, class FieldType = double >
class CurlcurlLocalizable
  : public LocalizableBase< internal::CurlcurlBase< GV, F, FieldType >, R, S >
{
  typedef LocalizableBase< internal::CurlcurlBase< GV, F, FieldType >, R, S > BaseType;

public:
  template< class... Args >
  CurlcurlLocalizable(Args&& ...args)
    : BaseType(std::forward< Args >(args)...)
  {}
};


/**
 * \brief An assemblable curlcurl product.
 *
 *        Possible ctor signaturer are a combination of the ones from \sa AssemblableBase first and then \sa
 *        internal::CurlcurlBase.
 * \todo  Add more documentation, especially a mathematical definition.
 */
template< class M, class F, class R, class GV = typename R::GridViewType, class S = R, class FieldType = double >
class CurlcurlAssemblable
  : public AssemblableBase< internal::CurlcurlBase< GV, F, FieldType >, M, R, S >
{
  typedef AssemblableBase< internal::CurlcurlBase< GV, F, FieldType >, M, R, S > BaseType;

public:
  template< class... Args >
  CurlcurlAssemblable(Args&& ...args)
    : BaseType(std::forward< Args >(args)...)
  {}
};


/**
 * \brief A curlcurl product.
 *
 *        Possible ctor signaturer are a combination of the ones from \sa GenericBase first and then \sa
 *        internal::CurlcurlBase.
 * \todo  Add more documentation, especially a mathematical definition.
 */
template< class GV, class F, class FieldType = double >
class Curlcurl
  : public GenericBase< internal::CurlcurlBase< GV, F, FieldType > >
{
  typedef GenericBase< internal::CurlcurlBase< GV, F, FieldType > > BaseType;

public:
  template< class... Args >
  Curlcurl(Args&& ...args)
    : BaseType(std::forward< Args >(args)...)
  {}
};


} // namespace Products
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_PRODUCTS_CURLCURL_HH
