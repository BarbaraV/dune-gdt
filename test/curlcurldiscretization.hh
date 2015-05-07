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



/** \brief Class to discretize curl-curl problems
 *
 * Problems of the form curl(mu curl E)+kappa E= f are solved, which admit a unique solution if the complex parameter kappa has positive imaginary part.
 * The discretization uses Nedelec spaces of the first family in lowest order on simplicial grids. The functions and parameters can be complex-valued.
 * The parameters mu and kappa have to be computed out of the material parameters permability, permittivity and conductivity by the user.
 * Note that mu in most cases will be the inverse (!) permability.
 * \note As boundary conditions, only a homogenous Dirchlet condition is supported at the moment.
 *
 * \tparam GridViewType  Type of grid
 * \tparam polynomialOrder Polynomial order of the function space to be used, only polynomialOrder=1 is possible at the moment
 * \tparam MatrixImp Type of the system matrix
 * \tparam VectorImp Type of the vectors for the right hand side and the solution
 */

template< class GridViewType,
          int polynomialOrder,
          class MatrixImp = Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::istl_sparse >::MatrixType,  //Matrix-interface atm inable to handle complex!
          class VectorImp = Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::istl_sparse >::VectorType >
class Discretization{
public:
    static const size_t dimDomain = GridViewType::dimension;
    typedef typename GridViewType::ctype DomainFieldType;
    static const size_t dimRange = dimDomain;
    static const unsigned int polOrder = polynomialOrder;

    typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewType::Intersection > BoundaryInfoType;
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, std::complex< double >, dimRange > FunctionType1;
    typedef Dune::Stuff::LocalizableFunctionInterface< typename GridViewType::template Codim< 0 >::Entity, DomainFieldType, dimDomain, double, dimRange > FunctionType2;

    typedef MatrixImp MatrixType;
    typedef VectorImp VectorType;

    typedef Dune::GDT::Spaces::Nedelec::PdelabBased< GridViewType, polOrder, double, dimRange > SpaceType;

    typedef Dune::GDT::DiscreteFunction< SpaceType, VectorType > DiscreteFunctionType;
    typedef Dune::GDT::ConstDiscreteFunction < SpaceType, VectorType > ConstDiscreteFunctionType;


    Discretization(const GridViewType& gp,
                   const BoundaryInfoType& info,
                   const FunctionType2& mu,
                   const FunctionType1& kappa,
                   const FunctionType1& src)
        : space_(gp)
        , boundary_info_(info)
        , mu_(mu)
        , kappa_(kappa)
        , sourceterm_(src)
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
      if (!is_assembled_) {
        //prepare
        Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_face_and_volume_pattern();
        system_matrix_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
        rhs_vector_ = VectorType(space_.mapper().size());
        SystemAssembler< SpaceType > grid_walker(space_);

        //rhs
        typedef GDT::Functionals::L2Volume< FunctionType1, VectorType, SpaceType > L2FunctionalType;
        L2FunctionalType source_functional(sourceterm_, rhs_vector_, space_);
        grid_walker.add(source_functional);

        //lhs
        typedef GDT::Operators::CurlCurl< FunctionType2, MatrixType, SpaceType > CurlOperatorType;
        CurlOperatorType curlcurl_operator(mu_, system_matrix_, space_);
        grid_walker.add(curlcurl_operator);
        typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< FunctionType1 > > IdOperatorType;
        const IdOperatorType identity_operator(kappa_);
        const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler(identity_operator);
        grid_walker.add(idMatrixAssembler, system_matrix_);

        //for non-homogeneous dirichlet boundary values you have to implement an appropriate DirichletProjection!
        //afterwards, the same procedure as in elliptic-cg-discretization can be used


        //apply the dirichlet constraints, atm only for homogenous dirichlet constraints!
        Spaces::DirichletConstraints< typename GridViewType::Intersection >
                dirichlet_constraints(boundary_info_, space_.mapper().maxNumDofs());
        grid_walker.add(dirichlet_constraints);
        grid_walker.assemble();
        dirichlet_constraints.apply(system_matrix_, rhs_vector_);
        is_assembled_ = true;
      }
    } //assemble()


    bool assembled() const
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

    void solve(VectorType& solution) const
    {
      if(!is_assembled_)
          assemble();

      //solve
      Dune::Stuff::LA::Solver< MatrixType >(system_matrix_).apply(rhs_vector_, solution);
    } //solve()


private:
    const SpaceType space_;
    const BoundaryInfoType& boundary_info_;
    const FunctionType2& mu_;
    const FunctionType1& kappa_;
    const FunctionType1& sourceterm_;
    mutable bool is_assembled_;
    mutable MatrixType system_matrix_;
    mutable VectorType rhs_vector_;
}; //class discretization




#endif // DUNE_GDT_TEST_CURLCURLDISCRETIZATION_HH
