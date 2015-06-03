// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_PRODUCTS_HCURL_HH
#define DUNE_GDT_PRODUCTS_HCURL_HH

#include "base.hh"
#include "hcurl-internal.hh"

namespace Dune {
namespace GDT {
namespace Products {


/**
 * \brief A localizable semi Hcurl product.
 *
 *        Possible ctor signaturer are a combination of the ones from \sa LocalizableBase first and then \sa
 *        internal::HcurlSemiBase.
 * \todo  Add more documentation, especially a mathematical definition.
 */
template< class GV, class R, class S = R, class FieldType = double >
class HcurlSemiLocalizable
  : public LocalizableBase< internal::HcurlSemiBase< GV, FieldType >, R, S >
{
  typedef LocalizableBase< internal::HcurlSemiBase< GV, FieldType >, R, S > BaseType;

public:
  template< class... Args >
  HcurlSemiLocalizable(Args&& ...args)
    : BaseType(std::forward< Args >(args)...)
  {}
};


/**
 * \brief An assemblable semi Hcurl product.
 *
 *        Possible ctor signaturer are a combination of the ones from \sa AssemblableBase first and then \sa
 *        internal::HcurlSemiBase.
 * \todo  Add more documentation, especially a mathematical definition.
 */
template< class M, class R, class GV = typename R::GridViewType , class S = R, class FieldType = double >
class HcurlSemiAssemblable
  : public AssemblableBase< internal::HcurlSemiBase< GV, FieldType >, M, R, S >
{
  typedef AssemblableBase< internal::HcurlSemiBase< GV, FieldType >, M, R, S > BaseType;

public:
  template< class... Args >
  HcurlSemiAssemblable(Args&& ...args)
    : BaseType(std::forward< Args >(args)...)
  {}
};


/**
 * \brief A semi Hcurl product.
 *
 *        Possible ctor signaturer are a combination of the ones from \sa GenericBase first and then \sa
 *        internal::HcurlSemiBase.
 * \todo  Add more documentation, especially a mathematical definition.
 */
template< class GV, class FieldType = double >
class HcurlSemi
  : public GenericBase< internal::HcurlSemiBase< GV, FieldType > >
{
  typedef GenericBase< internal::HcurlSemiBase< GV, FieldType > > BaseType;

public:
  template< class... Args >
  HcurlSemi(Args&& ...args)
    : BaseType(std::forward< Args >(args)...)
  {}
};


} // namespace Products
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_PRODUCTS_HCURL_HH
