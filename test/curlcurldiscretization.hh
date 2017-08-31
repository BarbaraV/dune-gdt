// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_TEST_CURLCURLDISCRETIZATION_HH
#define DUNE_GDT_TEST_CURLCURLDISCRETIZATION_HH

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

#include <dune/gdt/spaces/nedelec/pdelab.hh>
#include <dune/gdt/operators/curlcurl.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/spaces/constraints.hh>
#include <dune/gdt/localoperator/codim1.hh>




/** \brief Class to discretize curl-curl problems only using real matrices
 *
 * Problems of the form curl(mu curl E)+kappa E= f are solved, which admit a unique solution if the complex parameter kappa has positive imaginary part.
 * The discretization uses Nedelec spaces of the first family in lowest order on simplicial grids. The functions and parameters can be complex-valued.
 * The parameters mu and kappa have to be computed out of the material parameters permability, permittivity and conductivity by the user.
 * Note that mu in most cases will be the inverse (!) permability.
 * \note As boundary conditions, only a homogenous Dirichlet condition is supported at the moment.
 *
 * \tparam GridViewType  Type of grid
 * \tparam polynomialOrder Polynomial order of the function space to be used, only polynomialOrder=1 is possible at the moment
 * \tparam MatrixImp Type of the system matrix
 * \tparam VectorImp Type of the vectors for the right hand side and the solution
 */

template< class GridViewType, int polynomialOrder,
          bool is_matrix_curl = false, bool is_matrix_id = false >
class Discretization{
public:
  typedef typename GridViewType::ctype DomainFieldType;
  static const size_t       dimDomain = GridViewType::dimension;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewType::Intersection >                                                               BoundaryInfoType;
  typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimRange > Vectorfct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 VectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< GridViewType, polOrder, double, dimRange > SpaceType;

  typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType >       DiscreteFunctionType;
  typedef Dune::GDT::ConstDiscreteFunction < SpaceType, VectorType > ConstDiscreteFunctionType;

private:
  template< bool is_matrix, bool anything = true >
  struct Helper {
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 > ParameterFctType;
  };

  template< bool anything >
  struct Helper< true, anything > {
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimRange, dimRange > ParameterFctType;
  };
public:
  typedef typename Helper< is_matrix_curl >::ParameterFctType 		  CurlParameterType;
  typedef typename Helper< is_matrix_id >::ParameterFctType  		  IdParameterType;

  Discretization(const GridViewType& gp,
                 const BoundaryInfoType& info,
                 const CurlParameterType& mu,
                 const IdParameterType& kappareal,
                 const IdParameterType& kappaimag,
                 const Vectorfct& srcreal,
                 const Vectorfct& srcimag)
    : space_(gp)
    , boundary_info_(info)
    , mu_(mu)
    , kappa_real_(kappareal)
    , kappa_imag_(kappaimag)
    , sourceterm_real_(srcreal)
    , sourceterm_imag_(srcimag)
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
      system_matrix_total_ = MatrixTypeComplex(space_.mapper().size(), space_.mapper().size()); //,sparsity_pattern);
      rhs_vector_real_ = VectorType(space_.mapper().size());
      rhs_vector_imag_ = VectorType(space_.mapper().size());
      rhs_vector_total_ = VectorTypeComplex(space_.mapper().size());
      SystemAssembler< SpaceType > grid_walker(space_);

      //rhs
      auto source_functional_real = Dune::GDT::Functionals::make_l2_volume(sourceterm_real_, rhs_vector_real_, space_);
      grid_walker.add(*source_functional_real);
      auto source_functional_imag = Dune::GDT::Functionals::make_l2_volume(sourceterm_imag_, rhs_vector_imag_, space_);
      grid_walker.add(*source_functional_imag);

      //lhs
      typedef GDT::Operators::CurlCurl< CurlParameterType, MatrixType, SpaceType > CurlOperatorType;
      CurlOperatorType curlcurl_operator(mu_, system_matrix_real_, space_);
      grid_walker.add(curlcurl_operator);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< IdParameterType > > IdOperatorType;
      const IdOperatorType identity_operator1(kappa_real_);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler1(identity_operator1);
      grid_walker.add(idMatrixAssembler1, system_matrix_real_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< IdParameterType > > IdOperatorType;
      const IdOperatorType identity_operator2(kappa_imag_);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler2(identity_operator2);
      grid_walker.add(idMatrixAssembler2, system_matrix_imag_);


      //for non-homogeneous dirichlet boundary values you have to implement an appropriate DirichletProjection!
      //afterwards, the same procedure as in elliptic-cg-discretization can be used


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

  void solve(VectorTypeComplex& solution, 
             const Dune::Stuff::Common::Configuration& options = Dune::Stuff::LA::Solver< MatrixTypeComplex >::options("bicgstab.diagonal")) const
  {
    if(!is_assembled_)
      assemble();
    //solve
    Dune::Stuff::LA::Solver< MatrixTypeComplex > solver(system_matrix_total_);
    solver.apply(rhs_vector_total_, solution, options);
  } //solve()

  void visualize(const VectorTypeComplex& vector, const std::string filename, const std::string name) const
  {
    VectorType realvector(vector.size());
    VectorType imagvector(vector.size());
    realvector.backend() = vector.backend().real();
    imagvector.backend() = vector.backend().imag();
    ConstDiscreteFunctionType functionreal(space_, realvector, name+"real");
    ConstDiscreteFunctionType functionimag(space_, imagvector, name+"imag");
    functionreal.visualize(filename+"real");
    functionimag.visualize(filename+"imag");
  }


private:
  const SpaceType           space_;
  const BoundaryInfoType&   boundary_info_;
  const CurlParameterType&  mu_;
  const IdParameterType&    kappa_real_;
  const IdParameterType&    kappa_imag_;
  const Vectorfct&          sourceterm_real_;
  const Vectorfct&          sourceterm_imag_;
  mutable bool              is_assembled_;
  mutable MatrixType        system_matrix_real_;
  mutable MatrixType        system_matrix_imag_;
  mutable MatrixTypeComplex system_matrix_total_;
  mutable VectorType        rhs_vector_real_;
  mutable VectorType        rhs_vector_imag_;
  mutable VectorTypeComplex rhs_vector_total_;
}; //class Discretization


/** \brief Class to discretize scattering curl-curl problems
 *
 * Problems of the form curl(mu curl E)+kappa E= 0  with impedance boundary condition are solved.
 * The discretization uses Nedelec spaces of the first family in lowest order on simplicial grids. The functions and parameters can be complex-valued.
 * The parameters mu and kappa have to be computed out of the material parameters permability, permittivity and conductivity by the user.
 * Note that mu in most cases will be the inverse (!) permability.
 *
 * \tparam GridViewType  Type of grid
 * \tparam polynomialOrder Polynomial order of the function space to be used, only polynomialOrder=1 is possible at the moment
 * \tparam is_matrix_curl is the parameter in the curl operator matrix-valued?
 * \tparam is_matrix_id is the parameter in the identity operator matrix-valued?
 */

template< class GridViewType, int polynomialOrder, bool is_matrix_curl, bool is_matrix_id >
class ScatteringDiscretization{
public:
  typedef typename GridViewType::ctype DomainFieldType;
  static const size_t       dimDomain = GridViewType::dimension;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewType::Intersection >                                                               BoundaryInfoType;
  typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimRange > Vectorfct;
  typedef Dune::Stuff::Functions::Constant< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 > 	        ConstantFct;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 MatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 VectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< GridViewType, polOrder, double, dimRange > SpaceType;

  typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType >       DiscreteFunctionType;
  typedef Dune::GDT::ConstDiscreteFunction < SpaceType, VectorType > ConstDiscreteFunctionType;

private:
  template< bool is_matrix, bool anything = true >
  struct Helper {
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 > ParameterFctType;
  };

  template< bool anything >
  struct Helper< true, anything > {
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimRange, dimRange > ParameterFctType;
  };

public:
  typedef typename Helper< is_matrix_curl >::ParameterFctType 		  CurlParameterType;
  typedef typename Helper< is_matrix_id >::ParameterFctType  		  IdParameterType;
  typedef Dune::Stuff::Functions::Product< ConstantFct, IdParameterType > ProductFct;

  ScatteringDiscretization(const GridViewType& gp,
                           const BoundaryInfoType& info,
                 	   const CurlParameterType& mu_real,
 			   const CurlParameterType& mu_imag,
    			   const double wavenumber,
                 	   const IdParameterType& kappareal,
                 	   const IdParameterType& kappaimag,
                 	   const Vectorfct& bdryreal,
                 	   const Vectorfct& bdryimag)
    : space_(gp)
    , boundary_info_(info)
    , mu_real_(mu_real)
    , mu_imag_(mu_imag)
    , wavenumber_(wavenumber)
    , kappa_real_(kappareal)
    , kappa_imag_(kappaimag)
    , bdryterm_real_(bdryreal)
    , bdryterm_imag_(bdryimag)
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
      typedef LocalFunctional::Codim1Integral< LocalEvaluation::ProductTangential < Vectorfct > > LocalFunctionalType;
      LocalFunctionalType bdry_fctnal_real(bdryterm_real_);
      LocalFunctionalType bdry_fctnal_imag(bdryterm_imag_);
      const LocalAssembler::Codim1Vector< LocalFunctionalType > bdry_vector_real(bdry_fctnal_real);
      const LocalAssembler::Codim1Vector< LocalFunctionalType > bdry_vector_imag(bdry_fctnal_imag);
      grid_walker.add(bdry_vector_real, rhs_vector_real_, new Stuff::Grid::ApplyOn::NeumannIntersections< GridViewType >(boundary_info_));
      grid_walker.add(bdry_vector_imag, rhs_vector_imag_, new Stuff::Grid::ApplyOn::NeumannIntersections< GridViewType >(boundary_info_));

      //lhs
      //curlcurl part
      typedef GDT::Operators::CurlCurl< CurlParameterType, MatrixType, SpaceType > CurlOperatorType;
      CurlOperatorType curlcurl_operator_real(mu_real_, system_matrix_real_, space_);
      CurlOperatorType curlcurl_operator_imag(mu_imag_, system_matrix_imag_, space_);
      grid_walker.add(curlcurl_operator_real);
      grid_walker.add(curlcurl_operator_imag);
      //identity part
      ConstantFct wavenumber_fct(-1.0*wavenumber_);
      ConstantFct wavenumber_fct_squared(-1.0*wavenumber_ * wavenumber_);
      ProductFct id_param_real(wavenumber_fct_squared, kappa_real_, "stuff.functions.product");
      ProductFct id_param_imag(wavenumber_fct_squared, kappa_imag_, "stuff.functions.product");
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ProductFct > > IdOperatorType;
      const IdOperatorType identity_operator_real(id_param_real);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler_real(identity_operator_real);
      grid_walker.add(idMatrixAssembler_real, system_matrix_real_);
      const IdOperatorType identity_operator_imag(id_param_imag);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler_imag(identity_operator_imag);
      grid_walker.add(idMatrixAssembler_imag, system_matrix_imag_);
      //boundary part for complex impedance condition
      typedef LocalOperator::Codim1BoundaryIntegral< LocalEvaluation::ProductTangential< ConstantFct > > BdryOperatorType;
      const BdryOperatorType bdry_operator(wavenumber_fct);
      const LocalAssembler::Codim1BoundaryMatrix< BdryOperatorType > bdry_assembler(bdry_operator);
      grid_walker.add(bdry_assembler, system_matrix_imag_, new Stuff::Grid::ApplyOn::NeumannIntersections< GridViewType >(boundary_info_));

      grid_walker.assemble();

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

  void solve(VectorTypeComplex& solution,
             const Dune::Stuff::Common::Configuration& options = Dune::Stuff::LA::Solver< MatrixTypeComplex >::options("bicgstab.diagonal")) const
  {
    if(!is_assembled_)
      assemble();
    // solve
    Dune::Stuff::LA::Solver< MatrixTypeComplex > solver(system_matrix_total_);
    solver.apply(rhs_vector_total_, solution, options);
  } //solve()

  void visualize(const VectorTypeComplex& vector, const std::string filename, const std::string name) const
  {
    VectorType realvector(vector.size());
    VectorType imagvector(vector.size());
    realvector.backend() = vector.backend().real();
    imagvector.backend() = vector.backend().imag();
    ConstDiscreteFunctionType functionreal(space_, realvector, name+"real");
    ConstDiscreteFunctionType functionimag(space_, imagvector, name+"imag");
    functionreal.visualize(filename+"real");
    functionimag.visualize(filename+"imag");
  }


private:
  const SpaceType           space_;
  const BoundaryInfoType&   boundary_info_;
  const CurlParameterType&  mu_real_;
  const CurlParameterType&  mu_imag_;
  const double              wavenumber_;
  const IdParameterType&    kappa_real_;
  const IdParameterType&    kappa_imag_;
  const Vectorfct&          bdryterm_real_;
  const Vectorfct&          bdryterm_imag_;
  mutable bool              is_assembled_;
  mutable MatrixType        system_matrix_real_;
  mutable MatrixType        system_matrix_imag_;
  mutable MatrixTypeComplex system_matrix_total_;
  mutable VectorType        rhs_vector_real_;
  mutable VectorType        rhs_vector_imag_;
  mutable VectorTypeComplex rhs_vector_total_;
}; //class ScatteringDiscretization

#endif // DUNE_GDT_TEST_CURLCURLDISCRETIZATION_HH
