// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_TEST_HELMHOLTZ_DISCRETIZATION_HH
#define DUNE_GDT_TEST_HELMHOLTZ_DISCRETIZATION_HH

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
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/pdelab.hh>
#include <dune/gdt/operators/elliptic-cg.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/spaces/constraints.hh>


/** \brief Class to discretize Helmholtz problems
 *
 * Problems of the form -div(a grad u)-k^2 mu u= f are solved with Dirichlet or (complex) Robin-type boundary condition.
 * The functions and parameters can be complex-valued.
 *
 * \tparam GridViewType  Type of grid
 * \tparam polynomialOrder Polynomial order of the function space to be used, only polynomialOrder=1 is possible at the moment
 */

template< class GridViewType, int polynomialOrder >
class HelmholtzDiscretization{
public:
  typedef typename GridViewType::ctype DomainFieldType;
  static const size_t       dimDomain = GridViewType::dimension;
  static const size_t       dimRange = 1;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewType::Intersection >                                                               BoundaryInfoType;
  typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 >        ScalarFct;
  typedef Dune::Stuff::Functions::Constant< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 >                 ConstantFct;
  typedef Dune::Stuff::Functions::Product< ConstantFct, ScalarFct >                                                                                     ProductFct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 VectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewType, polOrder, double, dimRange > SpaceType;

  typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType >       DiscreteFunctionType;

  HelmholtzDiscretization(const GridViewType& gp,
                          const BoundaryInfoType& info,
                          const ScalarFct& a_real,
                          const ScalarFct& a_imag,
                          const ScalarFct& mu_real,
                          const ScalarFct& mu_imag,
                          const double wavenumber,
                          const ScalarFct& bdry_fct_real,
                          const ScalarFct& bdry_fct_imag,
                          const ScalarFct& source_fct_real,
                          const ScalarFct& source_fct_imag)
    : space_(gp)
    , boundary_info_(info)
    , a_real_(a_real)
    , a_imag_(a_imag)
    , mu_real_(mu_real)
    , mu_imag_(mu_imag)
    , wavenumber_(wavenumber)
    , bdry_fct_real_(bdry_fct_real)
    , bdry_fct_imag_(bdry_fct_imag)
    , source_fct_real_(source_fct_real)
    , source_fct_imag_(source_fct_imag)
    , is_assembled_(false)
    , system_matrix_real_(0,0)
    , system_matrix_imag_(0,0)
    , system_matrix_total_(0,0)
    , rhs_vector_real_(0)
    , rhs_vector_imag_(0)
    , rhs_vector_total_(0)
  {}

  const SpaceType& space() const
  {
    return space_;
  }

  VectorType create_vector() const   //oder besser VectorTypeComplex?
  {
    return VectorType(space_.mapper().size());
  }

  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if (!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_real_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_imag_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_total_ = MatrixTypeComplex(space_.mapper().size(), space_.mapper().size());
      rhs_vector_real_ = VectorType(space_.mapper().size());
      rhs_vector_imag_ = VectorType(space_.mapper().size());
      rhs_vector_total_ = VectorTypeComplex(space_.mapper().size());
      SystemAssembler< SpaceType > grid_walker(space_);

      //rhs
      auto source_functional_real = Dune::GDT::Functionals::make_l2_volume(source_fct_real_, rhs_vector_real_, space_);
      grid_walker.add(*source_functional_real);
      auto source_functional_imag = Dune::GDT::Functionals::make_l2_volume(source_fct_imag_, rhs_vector_imag_, space_);
      grid_walker.add(*source_functional_imag);
      auto bdry_functional_real = Dune::GDT::Functionals::make_l2_face(bdry_fct_real_, rhs_vector_real_, space_,
                                                                       new Stuff::Grid::ApplyOn::NeumannIntersections< GridViewType >(boundary_info_));
      grid_walker.add(*bdry_functional_real);
      auto bdry_functional_imag = Dune::GDT::Functionals::make_l2_face(bdry_fct_imag_, rhs_vector_imag_, space_,
                                                                       new Stuff::Grid::ApplyOn::NeumannIntersections< GridViewType >(boundary_info_));
      grid_walker.add(*bdry_functional_imag);

      //lhs
      //gradient part
      typedef GDT::Operators::EllipticCG< ScalarFct, MatrixType, SpaceType > EllipticOperatorType;
      EllipticOperatorType elliptic_operator_real(a_real_, system_matrix_real_, space_);
      EllipticOperatorType elliptic_operator_imag(a_imag_, system_matrix_imag_, space_);
      grid_walker.add(elliptic_operator_real);
      grid_walker.add(elliptic_operator_imag);
      //identity part
      ConstantFct wavenumber_fct(-1.0*wavenumber_);
      ConstantFct wavenumber_fct_squared(-1.0*wavenumber_ * wavenumber_);
      ProductFct id_param_real(wavenumber_fct_squared, mu_real_, "stuff.functions.product");
      ProductFct id_param_imag(wavenumber_fct_squared, mu_imag_, "stuff.functions.product");
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ProductFct > > IdOperatorType;
      const IdOperatorType identity_operator_real(id_param_real);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler_real(identity_operator_real);
      grid_walker.add(idMatrixAssembler_real, system_matrix_real_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ProductFct > > IdOperatorType;
      const IdOperatorType identity_operator_imag(id_param_imag);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler_imag(identity_operator_imag);
      grid_walker.add(idMatrixAssembler_imag, system_matrix_imag_);
      //boundary part for complex Robin-type condition
      typedef LocalOperator::Codim1BoundaryIntegral< LocalEvaluation::Product< ConstantFct > > BdryOperatorType;
      const BdryOperatorType bdry_operator(wavenumber_fct);
      const LocalAssembler::Codim1BoundaryMatrix< BdryOperatorType > bdry_assembler(bdry_operator);
      grid_walker.add(bdry_assembler, system_matrix_imag_, new Stuff::Grid::ApplyOn::NeumannIntersections< GridViewType >(boundary_info_));


      //for non-homogeneous dirichlet boundary values you have to use the DirichletProjection!
      //apply the dirichlet constraints, atm only for homogenous dirichlet constraints!
      Spaces::DirichletConstraints< typename GridViewType::Intersection >
             dirichlet_constraints(boundary_info_, space_.mapper().size());
      grid_walker.add(dirichlet_constraints);
      grid_walker.assemble();
      dirichlet_constraints.apply(system_matrix_real_, rhs_vector_real_);
      dirichlet_constraints.apply(system_matrix_imag_, rhs_vector_imag_);

      // total (block) matrix and (block) vector have to be assembled
      std::complex< double > im(0.0, 1.0);
      system_matrix_total_.backend() = system_matrix_imag_.backend().template cast<std::complex< double > >();
      system_matrix_total_.scal(im);
      system_matrix_total_.backend() += system_matrix_real_.backend().template cast< std::complex< double > >();
      rhs_vector_total_.backend() = rhs_vector_imag_.backend().template cast<std::complex< double > >();
      rhs_vector_total_.scal(im);
      rhs_vector_total_.backend() += rhs_vector_real_.backend().template cast< std::complex< double > >();

      is_assembled_ = true;
    }
  } //assemble()

  bool assembled() const
  {
    return is_assembled_;
  }


  const MatrixTypeComplex& system_matrix() const
  {
    return system_matrix_total_;
  }


  const VectorTypeComplex& rhs_vector() const
  {
    return rhs_vector_total_;
  }

  void solve(VectorTypeComplex& solution) const
  {
    if(!is_assembled_)
      assemble();

    // instantiate solver and options
    typedef Dune::Stuff::LA::Solver< MatrixTypeComplex > SolverType;
    SolverType solver(system_matrix_total_);
    //solve
    solver.apply(rhs_vector_total_, solution, "bicgstab.diagonal");
  } //solve()

  void visualize(const VectorTypeComplex& vector, const std::string filename, const std::string name) const
  {
    VectorType realvector(vector.size());
    VectorType imagvector(vector.size());
    realvector.backend() = vector.backend().real();
    imagvector.backend() = vector.backend().imag();
    DiscreteFunctionType functionreal(space_, realvector, name+"real");
    DiscreteFunctionType functionimag(space_, imagvector, name+"imag");
    functionreal.visualize(filename+"real");
    functionimag.visualize(filename+"imag");
  }

private:
  const SpaceType           space_;
  const BoundaryInfoType&   boundary_info_;
  const ScalarFct&          a_real_;
  const ScalarFct&          a_imag_;
  const ScalarFct&          mu_real_;
  const ScalarFct&          mu_imag_;
  const double              wavenumber_;
  const ScalarFct&          bdry_fct_real_;
  const ScalarFct&          bdry_fct_imag_;
  const ScalarFct&          source_fct_real_;
  const ScalarFct&          source_fct_imag_;
  mutable bool              is_assembled_;
  mutable MatrixType        system_matrix_real_;
  mutable MatrixType        system_matrix_imag_;
  mutable MatrixTypeComplex system_matrix_total_;
  mutable VectorType        rhs_vector_real_;
  mutable VectorType        rhs_vector_imag_;
  mutable VectorTypeComplex rhs_vector_total_;
};


#endif // DUNE_GDT_TEST_HELMHOLTZDISCRETIZATION_HH
