// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH
#define DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH

#if !HAVE_DUNE_PDELAB
# error "This one requires dune-pdelab!"
#endif

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/localevaluation/elliptic.hh>
#include <dune/gdt/localevaluation/curlcurl.hh>
#include <dune/gdt/localevaluation/divdiv.hh>           // koennte auch noch divdiv-operator schreiben um tw. Konstruktionen zu verkuerzen
#include <dune/gdt/localevaluation/product-l2deriv.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/discretefunction/default.hh>

namespace Dune {
namespace GDT {
namespace Operators {


enum class ChooseCellProblem {Elliptic, CurlcurlDivreg};

//forward
template< class GridViewImp, int polynomialOrder, ChooseCellProblem >
class Cell;

namespace internal {


template< class GridViewImp, int polynomialOrder >
class CellTraits
{
public:
  typedef GridViewImp                                          GridViewType;
  typedef typename GridViewType::template Codim< 0 >::Entity   EntityType;
  typedef typename GridViewType::ctype                         DomainFieldType;
  static const size_t                                          dimDomain = GridViewType::dimension;

  static const unsigned int polOrder = polynomialOrder;
  typedef double            RangeFieldType;

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;

};


} //namespace internal


template< class GridViewImp, int polynomialOrder >
class Cell< GridViewImp, polynomialOrder, ChooseCellProblem::Elliptic >
{
public:
  typedef internal::CellTraits< GridViewImp, polynomialOrder> Traits;
  typedef typename Traits::EntityType EntityType;
  typedef typename Traits::DomainFieldType DomainFieldType;
  typedef typename Traits::RangeFieldType RangeFieldType;
  typedef typename Traits::MatrixType MatrixType;
  typedef typename Traits::VectorType VectorType;

  static const size_t       dimDomain = Traits::dimDomain;
  static const unsigned int polOrder = Traits::polOrder;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, RangeFieldType, 1 > SpaceType;

  Cell(const GridViewImp& gridview, const ScalarFct& kappa)
    : space_(gridview)
    , kappa_(kappa)
    , is_assembled_(false)
    , system_matrix_(0,0)
    , rhs_vector_(0)
  {}

  const SpaceType& space() const
  {
    return space_;
  }

  VectorType create_vector() const
  {
    return VectorType(space_.mapper().size());
  }

  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      rhs_vector_ = VectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //lhs
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > > EllipticOp;
      EllipticOp ellipticop(kappa_);
      LocalAssembler::Codim0Matrix< EllipticOp > matrixassembler(ellipticop);
      walker.add(matrixassembler, system_matrix_);

      walker.assemble();
      is_assembled_ = true;
    }
  } //assemble

  bool is_assembled() const
  {
    return is_assembled_;
  }

  const MatrixType& system_matrix() const
  {
    return system_matrix_;
  }

  const VectorType& rhs_vector() const
  {
    return rhs_vector_;
  }

  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, DiscreteFunction< SpaceType, VectorType > reconstruction) const
  {
    if(!is_assembled_)
      assemble();
    // set up rhs
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, typename RhsVectorType::field_type, dimDomain > ConstFct;
    externfctvalue *= -1.0;
    ConstFct constrhs(externfctvalue);
    typedef Stuff::Functions::Product< ScalarFct, ConstFct > RhsFuncType;
    RhsFuncType rhsfunc(kappa_, constrhs);
    typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2grad< RhsFuncType > > L2gradOp;
    L2gradOp l2gradop(rhsfunc);
    LocalAssembler::Codim0Vector< L2gradOp > vectorassembler(l2gradop);

    //assemble rhs
    SystemAssembler< SpaceType > walker(space_);
    walker.add(vectorassembler, rhs_vector_);
    walker.assemble();

    //solve
    auto solution = create_vector();
    Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "lu.sparse");

    //make discrete function
    reconstruction.vector() = solution;
  }

  Dune::FieldMatrix< DomainFieldType, dimDomain, dimDomain > effective_matrix() const
  {
    auto unit_mat = Dune::Stuff::Functions::internal::UnitMatrix< double, dimDomain >.unit_matrix();
    Dune::FieldMatrix< DomainFieldType, dimDomain, dimDomain > ret;
    if(!is_assembled_)
      assemble();
    const auto averageparam = averageparameter();
    //prepare temporary storage
    VectorType tmp_vector(space_.mapper().size());
    typedef DiscreteFunction< SpaceType, VectorType > DiscrFct;
    std::vector< DiscrFct > reconstr(dimDomain, DiscrFct(space_, tmp_vector));
    std::vector< VectorType > tmp_rhs;
    //compute solutions of cell problems
    for (size_t ii =0; ii < dimDomain; ++ii) {
      reconstruct(unit_mat[ii], reconstr[ii]);
      tmp_rhs.emplace_back(rhs_vector_);
      tmp_rhs[ii].scal(-1.0); //necessary because rhs was -kappa*e_i and we want kappa*e_i
    }
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += tmp_rhs[jj] * reconstr[ii].vector();
        retRow[jj] += reconstr[jj].vector() * tmp_rhs[ii]; //for complex, this has to be conjugated!
        system_matrix_.mv(reconstr[ii].vector(), tmp_vector);
        retRow[jj] += reconstr[jj] * tmp_vector;
      }
    }
  } //effective_matrix()

  const typename ScalarFct::RangeFieldType averageparameter() const
  {
    typename ScalarFct::RangeFieldType result(0.0);
    const auto entity_it_end = space_.grid_view().template end<0>();
    //integrate
    for (auto entity_it = space_.grid_view().template begin<0>(); entity_it != entity_it_end; ++entity_it) {
      const auto entity = *entity_it;
      const auto localparam = kappa_.local_function(entity);
      const size_t int_order = localparam->order();
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(int_order));
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        const auto evaluation_result = localparam->evaluate(x);
        result += evaluation_result * quadrature_weight * integration_factor;
      } //loop over quadrature points
    } //loop over entities
    return result;
  }

private:
  const SpaceType    space_;
  const ScalarFct&   kappa_;
  mutable bool       is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
}; //class Cell<... ChoosecellProblem::Elliptic >


template< class GridViewImp, int polynomialOrder >
class Cell< GridViewImp, polynomialOrder, ChooseCellProblem::CurlcurlDivreg >
{
public:
  typedef internal::CellTraits< GridViewImp, polynomialOrder> Traits;
  typedef typename Traits::EntityType EntityType;
  typedef typename Traits::DomainFieldType DomainFieldType;
  typedef typename Traits::RangeFieldType RangeFieldType;
  typedef typename Traits::MatrixType MatrixType;
  typedef typename Traits::VectorType VectorType;

  static const size_t       dimDomain = Traits::dimDomain;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = Traits::polOrder;

  static_assert(dimDomain == 3, "This cell problem is only defined in 3d");

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, RangeFieldType, dimRange, 1 > SpaceType;

  Cell(const GridViewImp& gridview, const ScalarFct& mu,
       const ScalarFct& divparam = Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1>(1.0))
    : space_(gridview)
    , mu_(mu)
    , divparam_(divparam)
    , is_assembled_(false)
    , system_matrix_(0,0)
    , rhs_vector_(0)
  {}

  const SpaceType& space() const
  {
    return space_;
  }

  VectorType create_vector() const
  {
    return VectorType(space_.mapper().size());
  }

  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      rhs_vector_ = VectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //lhs
      typedef LocalOperator::Codim0Integral< LocalEvaluation::CurlCurl< ScalarFct > > CurlOp;
      CurlOp curlop(mu_);
      LocalAssembler::Codim0Matrix< CurlOp > matrixassembler1(curlop);
      walker.add(matrixassembler1, system_matrix_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Divdiv< ScalarFct > > DivOp;
      DivOp divop(divparam_);
      LocalAssembler::Codim0Matrix< DivOp > matrixassembler2(divop);
      walker.add(matrixassembler2, system_matrix_);

      walker.assemble();
      is_assembled_ = true;
    }
  } //assemble

  bool is_assembled() const
  {
    return is_assembled_;
  }

  const MatrixType& system_matrix() const
  {
    return system_matrix_;
  }

  const VectorType& rhs_vector() const
  {
    return rhs_vector_;
  }

  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, DiscreteFunction< SpaceType, VectorType > reconstruction) const
  {
    if(!is_assembled_)
      assemble();
    // set up rhs
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, typename RhsVectorType::field_type, dimDomain > ConstFct;
    externfctvalue *= -1.0;
    ConstFct constrhs(externfctvalue);
    typedef Stuff::Functions::Product< ScalarFct, ConstFct > RhsFuncType;
    RhsFuncType rhsfunc(mu_, constrhs);
    typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2curl< RhsFuncType > > L2curlOp;
    L2curlOp l2curlop(rhsfunc);
    LocalAssembler::Codim0Vector< L2curlOp > vectorassembler(l2curlop);

    //assemble rhs
    SystemAssembler< SpaceType > walker(space_);
    walker.add(vectorassembler, rhs_vector_);
    walker.assemble();

    //solve
    auto solution = create_vector();
    Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "lu.sparse");

    //make discrete function
    reconstruction.vector() = solution;
  }

  Dune::FieldMatrix< DomainFieldType, dimDomain, dimDomain > effective_matrix() const
  {
    auto unit_mat = Dune::Stuff::Functions::internal::UnitMatrix< double, dimDomain >.unit_matrix();
    Dune::FieldMatrix< DomainFieldType, dimDomain, dimDomain > ret;
    if(!is_assembled_)
      assemble();
    const auto averageparam = averageparameter();
    //prepare temporary storage
    VectorType tmp_vector(space_.mapper().size());
    typedef DiscreteFunction< SpaceType, VectorType > DiscrFct;
    std::vector< DiscrFct > reconstr(dimDomain, DiscrFct(space_, tmp_vector));
    std::vector< VectorType > tmp_rhs;
    //compute solutions of cell problems
    for (size_t ii =0; ii < dimDomain; ++ii) {
      reconstruct(unit_mat[ii], reconstr[ii]);
      tmp_rhs.emplace_back(rhs_vector_);
      tmp_rhs[ii].scal(-1.0); //necessary because rhs was -mu*e_i and we want mu*e_i
    }
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += tmp_rhs[jj] * reconstr[ii].vector();
        retRow[jj] += reconstr[jj].vector() * tmp_rhs[ii]; //for complex, this has to be conjugated!
        system_matrix_.mv(reconstr[ii].vector(), tmp_vector);
        retRow[jj] += reconstr[jj] * tmp_vector;
      }
    }
  } //effective_matrix()

  const typename ScalarFct::RangeFieldType averageparameter() const
  {
    typename ScalarFct::RangeFieldType result(0.0);
    const auto entity_it_end = space_.grid_view().template end<0>();
    //integrate
    for (auto entity_it = space_.grid_view().template begin<0>(); entity_it != entity_it_end; ++entity_it) {
      const auto entity = *entity_it;
      const auto localparam = mu_.local_function(entity);
      const size_t int_order = localparam->order();
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(int_order));
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        //intergation factors
        const double integration_factor = entity.geometry().integrationElement(x);
        const double quadrature_weight = quadPointIt->weight();
        //evaluate
        const auto evaluation_result = localparam->evaluate(x);
        result += evaluation_result * quadrature_weight * integration_factor;
      } //loop over quadrature points
    } //loop over entities
    return result;
  }

private:
  const SpaceType    space_;
  const ScalarFct&   mu_;
  const ScalarFct&   divparam_;
  mutable bool       is_assembled_;
  mutable MatrixType system_matrix_;
  mutable VectorType rhs_vector_;
}; //class Cell<... ChoosecellProblem::CurlcurlDivreg >


} //namespace Operators
} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH
