// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_TEST_HMM_HELMHOLTZ_HH
#define DUNE_GDT_TEST_HMM_HELMHOLTZ_HH

#if !HAVE_DUNE_PDELAB
# error "This one requires dune-pdelab!"
#endif

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>

#include <dune/common/timer.hh>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/common/ranges.hh>
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localevaluation/product.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/operators/elliptic-cg.hh>
#include <dune/gdt/operators/cellreconstruction.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

//for error computation
#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/h1.hh>
#include <dune/gdt/discretefunction/corrector.hh>

namespace Dune {
namespace GDT {


template< class GridViewImp >
class HelmholtzInclusionCell
{
public:
  typedef typename GridViewImp::template Codim< 0 >::Entity EntityType;
  typedef typename GridViewImp::ctype                       DomainFieldType;
  static const size_t                                       dimDomain = GridViewImp::dimension;

  typedef std::complex< double > complextype;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType      MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType      VectorType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, double, 1 > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType >            DiscreteFunctionType;
  typedef std::vector< DiscreteFunctionType >                             CellDiscreteFunctionType;
  typedef std::vector< std::shared_ptr< CellDiscreteFunctionType > >      CellSolutionStorageType;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, double, 1 > ScalarFct;
  typedef Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, double, 1 >          ConstantFct;

  typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > >  EllipticOperator;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ConstantFct > > IdOperator;

  typedef LocalAssembler::Codim0Matrix< EllipticOperator > EllipticAssembler;
  typedef LocalAssembler::Codim0Matrix< IdOperator >       IdAssembler;


  HelmholtzInclusionCell(const GridViewImp& cell_gridview, const ScalarFct& a_real, const ScalarFct& a_imag,
                         const ConstantFct& wavenumber_squared_neg)
    : cell_space_(cell_gridview)
    , a_real_(a_real)
    , a_imag_(a_imag)
    , wavenumber_squared_neg_(wavenumber_squared_neg)
    , system_matrix_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_assembler_(cell_space_)
    , system_matrix_real_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_matrix_imag_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , elliptic_operator_real_(a_real_)
    , elliptic_operator_imag_(a_imag_)
    , id_operator_(wavenumber_squared_neg_)
    , elliptic_assembler_real_(elliptic_operator_real_)
    , elliptic_assembler_imag_(elliptic_operator_imag_)
    , id_assembler_(id_operator_)
  {
    system_assembler_.add(elliptic_assembler_real_, system_matrix_real_);
    system_assembler_.add(elliptic_assembler_imag_, system_matrix_imag_);
    system_assembler_.add(id_assembler_, system_matrix_real_);
  }

  const SpaceType& cell_space() const
  {
    return cell_space_;
  }

  void compute_cell_solutions(CellSolutionStorageType& cell_solutions) const
  {
    assert(cell_solutions.size() > 0);
    //clear return argument
    for (auto& localSol : cell_solutions) {
      assert(localSol->size() > 1);
      localSol->operator[](0).vector() *= 0;
      localSol->operator[](1).vector() *= 0;
    }
    //assemble rhs and system
    ConstantFct one(1.0);
    VectorType rhs_vector_real(cell_space_.mapper().size());
    VectorTypeComplex rhs_vector_total(cell_space_.mapper().size());
    auto rhs_functional_real = Dune::GDT::Functionals::make_l2_volume(one, rhs_vector_real, cell_space_);
    system_assembler_.add(*rhs_functional_real);
    Spaces::DirichletConstraints< typename GridViewImp::Intersection >
           dirichlet_constraints(DSG::BoundaryInfos::AllDirichlet< typename GridViewImp::Intersection >(), cell_space_.mapper().size());
    system_assembler_.add(dirichlet_constraints);
    system_assembler_.assemble();
    dirichlet_constraints.apply(system_matrix_real_, rhs_vector_real);
    dirichlet_constraints.apply(system_matrix_imag_);
    //make complex matrix and vector
    rhs_vector_total.backend() = rhs_vector_real.backend().template cast< complextype >();
    complextype im(0.0, 1.0);
    system_matrix_.backend() = system_matrix_imag_.backend().template cast< complextype >();
    system_matrix_.scal(im);
    system_matrix_.backend() += system_matrix_real_.backend().template cast< complextype >();
    //solve
    assert(cell_solutions.size() == 1);
    auto& current_solution = *cell_solutions[0];
    VectorTypeComplex tmp_solution(rhs_vector_total.size());
    if(!rhs_vector_total.valid())
      DUNE_THROW(Dune::InvalidStateException, "RHS vector invalid!");
    Stuff::LA::Solver< MatrixTypeComplex > solver(system_matrix_);
    solver.apply(rhs_vector_total, tmp_solution, "bicgstab.diagonal");
    if(!tmp_solution.valid())
      DUNE_THROW(Dune::InvalidStateException, "Solution vector invalid!");
    //make discrete functions
    assert(current_solution.size() > 1);
    current_solution[0].vector().backend() = tmp_solution.backend().real();
    current_solution[1].vector().backend() = tmp_solution.backend().imag();
  } //compute_cell_solutions

  template< class FunctionType >
  typename FunctionType::RangeType average(FunctionType& function) const
  {
    typename FunctionType::RangeType result(0.0);
    //integrate
    for (const auto& entity : DSC::entityRange(cell_space_.grid_view()) ) {
      const auto localparam = function.local_function(entity);
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
        auto evaluation_result = localparam->evaluate(x);
        evaluation_result *= (quadrature_weight * integration_factor);
        result += evaluation_result;
      } //loop over quadrature points
    } //loop over entities
    return result;
  } //average

  complextype effective_param() const
  {
    CellSolutionStorageType cell_solution(1);
    for (auto& it : cell_solution){
      std::vector< DiscreteFunctionType > it1(2, DiscreteFunctionType(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    compute_cell_solutions(cell_solution);
    typedef Dune::Stuff::Functions::Product< ConstantFct, DiscreteFunctionType > ProductFct;
    ProductFct real_integrand(wavenumber_squared_neg_, cell_solution[0]->operator[](0));
    ProductFct imag_integrand(wavenumber_squared_neg_, cell_solution[0]->operator[](1));
    double real_result = 1 - average(real_integrand);
    double imag_result = -1*average(imag_integrand);
    return complextype(real_result, imag_result);
  }

private:
  const SpaceType                      cell_space_;
  const ScalarFct&                     a_real_;
  const ScalarFct&                     a_imag_;
  const ConstantFct&                   wavenumber_squared_neg_;
  mutable MatrixTypeComplex            system_matrix_;
  mutable SystemAssembler< SpaceType > system_assembler_;
  mutable MatrixType                   system_matrix_real_;
  mutable MatrixType                   system_matrix_imag_;
  mutable EllipticOperator             elliptic_operator_real_;
  mutable EllipticOperator             elliptic_operator_imag_;
  mutable IdOperator                   id_operator_;
  mutable EllipticAssembler            elliptic_assembler_real_;
  mutable EllipticAssembler            elliptic_assembler_imag_;
  mutable IdAssembler                  id_assembler_;
};


} // namespace GDT
} //namespace Dune


#endif // DUNE_GDT_TEST_HMM_HELMHOLTZ_HH
