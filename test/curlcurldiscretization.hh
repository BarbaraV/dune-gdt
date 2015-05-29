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

template< class GridViewType,
          int polynomialOrder//,
          /*class MatrixImp = Dune::Stuff::LA::Container< double >::MatrixType,
          class VectorImp = Dune::Stuff::LA::Container< double >::VectorType*/ >
class DiscretizationReal{
public:
    static const size_t dimDomain = GridViewType::dimension;
    typedef typename GridViewType::ctype DomainFieldType;
    static const size_t dimRange = dimDomain;
    static const unsigned int polOrder = polynomialOrder;

    typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewType::Intersection > BoundaryInfoType;
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimRange > Vectorfct;
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, 1 > ScalarFct;

    /*typedef MatrixImp MatrixType;
    typedef VectorImp VectorType;*/

    typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
    typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;
    typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixTypeComplex;
    typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorTypeComplex;

    typedef Dune::GDT::Spaces::Nedelec::PdelabBased< GridViewType, polOrder, double, dimRange > SpaceType;

    typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType > DiscreteFunctionType;
    typedef Dune::GDT::ConstDiscreteFunction < SpaceType, VectorType > ConstDiscreteFunctionType;


    DiscretizationReal(const GridViewType& gp,
                       const BoundaryInfoType& info,
                       const ScalarFct& mu,
                       const ScalarFct& kappareal,
                       const ScalarFct& kappaimag,
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

    VectorType create_vector() const
    {
        return VectorType(space_.mapper().size());
    }

    void assemble() const
    {
      using namespace Dune;
      using namespace Dune::GDT;
      if (!is_assembled_) {
        //prepare
        Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_face_and_volume_pattern();
        system_matrix_real_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
        system_matrix_imag_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
        //system_matrix_total_ = MatrixType(2*space_.mapper().size(), 2*space_.mapper().size());
        system_matrix_total_ = MatrixTypeComplex(space_.mapper().size(), space_.mapper().size());
        rhs_vector_real_ = VectorType(space_.mapper().size());
        rhs_vector_imag_ = VectorType(space_.mapper().size());
        //rhs_vector_total_ = VectorType(2*space_.mapper().size());
        rhs_vector_total_ = VectorTypeComplex(space_.mapper().size());
        SystemAssembler< SpaceType > grid_walker(space_);

        //rhs
        auto source_functional_real = Dune::GDT::Functionals::make_l2_volume(sourceterm_real_, rhs_vector_real_, space_);
        grid_walker.add(*source_functional_real);
        auto source_functional_imag = Dune::GDT::Functionals::make_l2_volume(sourceterm_imag_, rhs_vector_imag_, space_);
        grid_walker.add(*source_functional_imag);

        //lhs
        typedef GDT::Operators::CurlCurl< ScalarFct, MatrixType, SpaceType > CurlOperatorType;
        CurlOperatorType curlcurl_operator(mu_, system_matrix_real_, space_);
        grid_walker.add(curlcurl_operator);
        typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ScalarFct > > IdOperatorType;
        const IdOperatorType identity_operator1(kappa_real_);
        const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler1(identity_operator1);
        grid_walker.add(idMatrixAssembler1, system_matrix_real_);
        typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ScalarFct > > IdOperatorType;
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

    void solve(VectorTypeComplex& solution) const
    {
      if(!is_assembled_)
          assemble();

      // instantiate solver and options
      typedef Dune::Stuff::LA::Solver< MatrixTypeComplex > SolverType;
      SolverType solver(system_matrix_total_);
      Dune::Stuff::Common::Configuration options = SolverType::options();
      options.set("check_for_inf_nan", "0", true);
      //solve
      solver.apply(rhs_vector_total_, solution, options);
    } //solve()


private:
    const SpaceType space_;
    const BoundaryInfoType& boundary_info_;
    const ScalarFct& mu_;
    const ScalarFct& kappa_real_;
    const ScalarFct& kappa_imag_;
    const Vectorfct& sourceterm_real_;
    const Vectorfct& sourceterm_imag_;
    mutable bool is_assembled_;
    mutable MatrixType system_matrix_real_;
    mutable MatrixType system_matrix_imag_;
    mutable MatrixTypeComplex system_matrix_total_;
    mutable VectorType rhs_vector_real_;
    mutable VectorType rhs_vector_imag_;
    mutable VectorTypeComplex rhs_vector_total_;
}; //class discretizationreal




#endif // DUNE_GDT_TEST_CURLCURLDISCRETIZATION_HH
