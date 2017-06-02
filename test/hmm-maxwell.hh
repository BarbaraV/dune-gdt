// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_TEST_HMM_MAXWELL_HH
#define DUNE_GDT_TEST_HMM_MAXWELL_HH

#if !HAVE_DUNE_PDELAB
# error "This one requires dune-pdelab!"
#endif

#if !HAVE_DUNE_FEM
# error "This one requires dune-fem!"
#endif

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>
#include <limits>
#include <algorithm>

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

#include <dune/gdt/spaces/nedelec/pdelab.hh>
#include <dune/gdt/spaces/cg/fem.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/localevaluation/hmm.hh>
#include <dune/gdt/localevaluation/product.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/operators/cellreconstruction.hh>
#include <dune/gdt/operators/curlcurl.hh>
#include <dune/gdt/assembler/local/codim0.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/spaces/constraints.hh>

//for error computation
#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/hcurl.hh>
#include <dune/gdt/discretefunction/corrector.hh>


namespace Dune {
namespace GDT {
namespace Operators {

// for reference of a_eff only
template< class CoarseSpaceType, class CellGridType >
class CurlEllipticCellReconstruction
  : public CellReconstruction< CoarseSpaceType, CellGridType,
                               typename Spaces::CG::FemBased< typename Fem::PeriodicLeafGridPart< CellGridType >, 1, double, 1 >, false >
{
public:
  typedef typename Spaces::CG::FemBased< typename Fem::PeriodicLeafGridPart< CellGridType >, 1, double, 1 > CellSpaceType;
private:
  typedef CurlEllipticCellReconstruction< CoarseSpaceType, CellGridType >                                   ThisType;
  typedef CellReconstruction< CoarseSpaceType, CellGridType, CellSpaceType, false >                         BaseType;
public:
  using typename BaseType::CoarseEntityType;
  using typename BaseType::CoarseDomainType;
  using typename BaseType::CoarseDomainFieldType;
  using typename BaseType::CellSolutionStorageType;
  using typename BaseType::CellDiscreteFunctionType;
  using typename BaseType::DomainFieldType;
  using typename BaseType::PeriodicGridPartType;
  using typename BaseType::PeriodicViewType;
  using typename BaseType::PeriodicEntityType;
  using typename BaseType::MatrixType;
  using typename BaseType::VectorType;
  using typename BaseType::RangeFieldType;
  using BaseType::dimDomain;
  using BaseType::dimRange;

  typedef Dune::Stuff::LocalizableFunctionInterface< PeriodicEntityType, DomainFieldType, dimDomain, RangeFieldType, 1 >     ScalarFct;
  typedef std::function< bool(const PeriodicViewType&, const PeriodicEntityType&) >                                          FilterType;

  typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > >  EllipticOperator;
  typedef LocalAssembler::Codim0Matrix< EllipticOperator>    			   EllipticAssembler;

  using BaseType::coarse_space_;
  using BaseType::grid_part_;
  using BaseType::cell_space_;
  using BaseType::system_matrix_;
  using BaseType::system_assembler_;

  CurlEllipticCellReconstruction(const CoarseSpaceType& coarse_space, CellGridType& cell_grid, const ScalarFct& mu,
			         FilterType filter)
    : BaseType(coarse_space, cell_grid, false)
    , mu_(mu)
    , elliptic_op_(mu_)
    , elliptic_assembler_(elliptic_op_)
    , filter_(filter)
  {
    assert(filter_);  //not empty
    system_assembler_.add(elliptic_assembler_, system_matrix_, new Stuff::Grid::ApplyOn::FilteredEntities< PeriodicViewType >(filter_));
  }

  void assemble_all_local_rhs(const CoarseEntityType& coarse_entity, CellSolutionStorageType& cell_solutions, const CoarseDomainType& xx) const override final
  { }

  void assemble_cell_solutions_rhs(CellSolutionStorageType& cell_solutions) const override final { }

  /**
   * @brief compute_cell_solutions Computes the cell corrections
   * @param cell_solutions Vector of pointers to discrete functions to store the results in
   */
  void compute_cell_solutions(CellSolutionStorageType& cell_solutions) const
  {
    assert(cell_solutions.size() > 0 && "You have to pre-allocate space");
    //rhs
    typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType RealVectorType;
    std::vector< RealVectorType > rhs_vectors(dimDomain, RealVectorType(cell_space_.mapper().size()));
    system_assembler_.assemble();
    //constraints
    for (const auto& entity : DSC::entityRange(coarse_space_.grid_view()) ) {
      if (!filter_(cell_space_.grid_view(), entity)) {  //work only where filter_=false
        //set unit_row in the system_matrix
        Dune::DynamicVector< size_t > global_indices;
        cell_space_.mapper().globalIndices(entity, global_indices);
        for (size_t ii = 0; ii < global_indices.size(); ++ii)
          system_matrix_.unit_row(global_indices[ii]);
        std::vector< Dune::FieldVector< DomainFieldType, dimDomain > > vertices;
        const auto geometry = entity.geometry();
        for (auto cc : DSC::valueRange(geometry.corners()))
          vertices.emplace_back(geometry.local(geometry.corner(cc)));
        // find the corresponding basis functions
        const auto basis = cell_space_.base_function_set(entity);
        typedef typename Dune::FieldVector< RangeFieldType, dimRange > RangeType;
        std::vector< RangeType > tmp_basis_values(basis.size(), RangeType(0));
        for (size_t cc = 0; cc < vertices.size(); ++cc) {
          // find the basis function that evaluates to one here (has to be only one!)
          basis.evaluate(vertices[cc], tmp_basis_values);
          size_t ones = 0;
          size_t zeros = 0;
          size_t failures = 0;
          for (size_t jj = 0; jj < basis.size(); ++jj) {
            if (std::abs(tmp_basis_values[jj][0] - RangeFieldType(1)) < 1e-12) {
              auto globaljj = cell_space_.mapper().mapToGlobal(entity, jj);
              for (size_t dd = 0; dd < dimDomain; ++dd )
                rhs_vectors[dd][globaljj] = -1.0 * geometry.corner(cc)[dd];  //-x_dd as value
              ++ones;
            } else if (std::abs(tmp_basis_values[jj][0]) < 1e-12)
                ++zeros;
              else
                ++failures;
          }
          assert(ones == 1 && zeros == (basis.size() - 1) && failures == 0 && "This must not happen for polOrder 1!");
        } //loop over vertices
      } //only apply where filter_=false 
    } //loop over entities
    //solve
    //clear return argument
    for (auto& localSol : cell_solutions) {
      assert(localSol->size() > 0);
      localSol->operator[](0).vector() *= 0;
    }
    //actual solve
    for (auto ii : DSC::valueRange(cell_solutions.size())) {
      auto& current_rhs = rhs_vectors[ii];
      auto& current_solution = *cell_solutions[ii];
      apply(current_rhs, current_solution);
    }
  } //compute_cell_solutions

  void apply(const VectorType& current_rhs, CellDiscreteFunctionType& current_solution) const override final
  {
    assert(current_solution.size() > 0 && "This has to be a pre-allocated vector");
    BaseType::apply(current_rhs, current_solution[0].vector());
  } //apply(Vector, DiscreteFct)

  void apply(const CellDiscreteFunctionType& current_rhs, CellDiscreteFunctionType& current_solution) const override final
  {
    assert(current_rhs.size() > 0 && "This has to be a pre-allocated vector");
    apply(current_rhs[0].vector(), current_solution);
  } //apply(DiscreteFct, DiscreteFct)

  /**
   * @brief effective_matrix Computes the effective matrix belonging to this cell problem
   * @return effective matrix
   */
  FieldMatrix< RangeFieldType, dimDomain, dimDomain > effective_matrix() const
  {
    CellSolutionStorageType cell_solutions(dimDomain);
    for (auto& it : cell_solutions) {
      std::vector<DiscreteFunction< CellSpaceType, VectorType > > it1(1, DiscreteFunction< CellSpaceType, VectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    compute_cell_solutions(cell_solutions);
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    FieldMatrix< RangeFieldType, dimDomain, dimDomain > ret;
    typedef typename Stuff::LocalfunctionSetInterface
        < PeriodicEntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >::JacobianRangeType JacobianRangeType;
    //compute matrix
    for (const auto& entity : DSC::entityRange(cell_space_.grid_view()) ) {
      //if (filter_(cell_space_.grid_view(), entity)) {
        const auto localparam = mu_.local_function(entity);
        const size_t int_order = localparam->order();
        //get quadrature rule
        typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
        typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
        const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(int_order));
        // evaluate the jacobians of all local solutions in all quadrature points
        std::vector<std::vector<JacobianRangeType>> allLocalSolutionEvaluations(
           cell_solutions.size(), std::vector<JacobianRangeType>(volumeQuadrature.size(), JacobianRangeType(0.0)));
        for (auto lsNum : DSC::valueRange(cell_solutions.size())) {
          const auto local_cell_function = cell_solutions[lsNum]->operator[](0).local_function(entity);
          local_cell_function->jacobian(volumeQuadrature, allLocalSolutionEvaluations[lsNum]);
        }
        //loop over all quadrature points
        const auto quadPointEndIt = volumeQuadrature.end();
        size_t kk= 0;
        for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++kk) {
          const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
          //integration factors
          const double integration_factor = entity.geometry().integrationElement(x);
          const double quadrature_weight = quadPointIt->weight();
          //evaluate
          auto evaluation_result = localparam->evaluate(x);
          evaluation_result *= (quadrature_weight * integration_factor);
          for (size_t ii = 0; ii < dimDomain; ++ii) {
            auto& retRow = ret[ii];
            for (size_t jj = 0; jj < dimDomain; ++jj) {
              auto tmp_result = (unit_mat[ii][jj] + allLocalSolutionEvaluations[ii][kk][0][jj]
                                    + allLocalSolutionEvaluations[jj][kk][0][ii]
                                    + (allLocalSolutionEvaluations[ii][kk][0] * allLocalSolutionEvaluations[jj][kk][0]) );
              tmp_result *= evaluation_result;
              retRow[jj] += tmp_result;
            }
          }
        } //loop over quadrature points
      //} //only appply on entities with filter_=true
    } //loop over entities
    ret.invert();
    return ret;
  } //effective_matrix


private:
  const ScalarFct&                 mu_;
  mutable EllipticOperator         elliptic_op_;
  mutable EllipticAssembler        elliptic_assembler_;
  const FilterType		   filter_;
}; //CurlEllipticReconstruction //for reference/comparison only


} //namespcae Operators


template< class GridViewImp >
class MaxwellInclusionCell
{
public:
  typedef GridViewImp                                       PeriodicViewType;
  typedef typename GridViewImp::template Codim< 0 >::Entity EntityType;
  typedef typename GridViewImp::ctype                       DomainFieldType;
  static const size_t                                       dimDomain = GridViewImp::dimension;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename GridViewImp::Intersection > BdryInfoType;

  typedef std::complex< double > complextype;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType      RealMatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType      RealVectorType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< GridViewImp, 1, double, 3 > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, RealVectorType >             DiscreteFunctionType;
  typedef std::vector< DiscreteFunctionType >                                  CellDiscreteFunctionType;
  typedef std::vector< std::shared_ptr< CellDiscreteFunctionType > >           CellSolutionStorageType;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, double, 1 > ScalarFct;
  typedef Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, double, 1 >          ConstantFct;

  typedef LocalOperator::Codim0Integral< LocalEvaluation::CurlCurl< ScalarFct > >  CurlOperator;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ConstantFct > > IdOperator;

  typedef LocalAssembler::Codim0Matrix< CurlOperator >     CurlAssembler;
  typedef LocalAssembler::Codim0Matrix< IdOperator >       IdAssembler;


  MaxwellInclusionCell(const GridViewImp& cell_gridview, const ScalarFct& a_real, const ScalarFct& a_imag,
                       const ConstantFct& wavenumber_squared_neg, const BdryInfoType& info)
    : cell_space_(cell_gridview)
    , boundary_info_(info)
    , a_real_(a_real)
    , a_imag_(a_imag)
    , wavenumber_squared_neg_(wavenumber_squared_neg)
    , system_matrix_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_assembler_(cell_space_)
    , system_matrix_real_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_matrix_imag_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , curl_operator_real_(a_real_)
    , curl_operator_imag_(a_imag_)
    , id_operator_(wavenumber_squared_neg_)
    , curl_assembler_real_(curl_operator_real_)
    , curl_assembler_imag_(curl_operator_imag_)
    , id_assembler_(id_operator_)
  {
    system_assembler_.add(curl_assembler_real_, system_matrix_real_);
    system_assembler_.add(curl_assembler_imag_, system_matrix_imag_);
    system_assembler_.add(id_assembler_, system_matrix_real_);
  }

  const SpaceType& cell_space() const
  {
    return cell_space_;
  }

  void assemble_cell_solutions_rhs(CellSolutionStorageType& cell_rhs) const
  {
    assert(cell_rhs.size() > 0 && "You have to pre-allocate space");
    ConstantFct one(1.0);
    typedef FieldMatrix< double, dimDomain, dimDomain > VectorofVectorsType;
    VectorofVectorsType unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    typedef GDT::Functionals::L2Volume< ScalarFct, RealVectorType, SpaceType, PeriodicViewType,
                                        LocalEvaluation::VectorL2< ScalarFct, VectorofVectorsType > > RhsFunctionalType;
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals_real(dimDomain);
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      assert(cell_rhs[ii]);
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2< ScalarFct, VectorofVectorsType> >
              local_rhs_functional_real(one, unit_mat, ii);
      auto& rhs_vector_real = cell_rhs[ii]->operator[](0).vector();
      rhs_functionals_real[ii] = DSC::make_unique<RhsFunctionalType>(one, rhs_vector_real, cell_space_, local_rhs_functional_real);
      system_assembler_.add(*rhs_functionals_real[ii]);
    }
    //add Dirichlet constraints, assemble and apply
    Spaces::DirichletConstraints< typename GridViewImp::Intersection >
           dirichlet_constraints(boundary_info_, cell_space_.mapper().size());
    system_assembler_.add(dirichlet_constraints);
    system_assembler_.assemble(); 
    dirichlet_constraints.apply(system_matrix_real_);
    dirichlet_constraints.apply(system_matrix_imag_);
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      dirichlet_constraints.apply(cell_rhs[ii]->operator[](0).vector());
    }
    //make complex matrix
    complextype im(0.0, 1.0);
    system_matrix_.backend() = system_matrix_imag_.backend().template cast< complextype >();
    system_matrix_.scal(im);
    system_matrix_.backend() += system_matrix_real_.backend().template cast< complextype >();
  } //assemble_cell_solutions_rhs

  void apply(const CellDiscreteFunctionType& current_rhs, CellDiscreteFunctionType& current_solution) const
  {
    VectorType tmp_rhs(current_rhs[0].vector().size());
    tmp_rhs.backend() = current_rhs[0].vector().backend().template cast< complextype >();
    VectorType tmp_solution(current_rhs[0].vector().size());
    //actual solve
    if(!tmp_rhs.valid())
      DUNE_THROW(Dune::InvalidStateException, "RHS vector invalid!");
    typedef Stuff::LA::Solver< MatrixType > SolverType;
    Stuff::Common::Configuration options = SolverType::options("bicgstab.diagonal");
    options.set("max_iter", "50000", true);
    options.set("precision", "1e-6", true);
    SolverType solver(system_matrix_);
    solver.apply(tmp_rhs, tmp_solution, options);
    if(!tmp_solution.valid())
      DUNE_THROW(Dune::InvalidStateException, "Solution vector invalid!");
    //make discrete functions
    assert(current_solution.size() > 1);
    current_solution[0].vector().backend() = tmp_solution.backend().real();
    current_solution[1].vector().backend() = tmp_solution.backend().imag();
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
    //prepare rhs
    CellSolutionStorageType all_cell_rhs(cell_solutions.size());
    for (auto& it : all_cell_rhs) {
      std::vector<DiscreteFunction< SpaceType, RealVectorType > > it1(1, DiscreteFunction< SpaceType, RealVectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    assemble_cell_solutions_rhs(all_cell_rhs);
    //solve
    assert(cell_solutions.size() == dimDomain);
    for (auto ii : DSC::valueRange(cell_solutions.size())) {
      //extract vectors
      auto& current_rhs = *all_cell_rhs[ii];
      auto& current_solution = *cell_solutions[ii];
      apply(current_rhs, current_solution);
    }
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

  std::vector< FieldMatrix< double, dimDomain, dimDomain > > effective_matrix() const
  {
    CellSolutionStorageType cell_solutions(dimDomain);
    for (auto& it : cell_solutions) {
      std::vector<DiscreteFunction< SpaceType, RealVectorType > > it1(2, DiscreteFunction< SpaceType, RealVectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    compute_cell_solutions(cell_solutions);
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    typedef Dune::Stuff::Functions::Product< ConstantFct, DiscreteFunctionType > ProductFct;
    std::vector< FieldMatrix< double, dimDomain, dimDomain > > ret(2);
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow_real = ret[0][ii];
      auto& retRow_imag = ret[1][ii];
      ProductFct real_integrand(wavenumber_squared_neg_, cell_solutions[ii]->operator[](0));
      ProductFct imag_integrand(wavenumber_squared_neg_, cell_solutions[ii]->operator[](1));
      Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, double, dimDomain > unit_mat_col(unit_mat[ii]);
      auto average_vol = average(unit_mat_col);
      auto real_result = average(real_integrand);
      auto imag_result = average(imag_integrand);
      real_result *= -1.0;  //bc wavenumber_squared_neg
      imag_result *= -1.0;
      retRow_real = average_vol + real_result;
      retRow_imag = imag_result;
    }
    //ret[0].transpose();
    //ret[1].transpose();
    return ret;
  } //effective_matrix

private:
  const SpaceType                      cell_space_;
  const BdryInfoType& 		       boundary_info_;
  const ScalarFct&                     a_real_;
  const ScalarFct&                     a_imag_;
  const ConstantFct&                   wavenumber_squared_neg_;
  mutable MatrixType                   system_matrix_;
  mutable SystemAssembler< SpaceType > system_assembler_;
  mutable RealMatrixType               system_matrix_real_;
  mutable RealMatrixType               system_matrix_imag_;
  mutable CurlOperator                 curl_operator_real_;
  mutable CurlOperator                 curl_operator_imag_;
  mutable IdOperator                   id_operator_;
  mutable CurlAssembler                curl_assembler_real_;
  mutable CurlAssembler                curl_assembler_imag_;
  mutable IdAssembler                  id_assembler_;
}; //MaxwellInclusionCell


} //namespace GDT
} //namespcae Dune


double distance_to_cube(Dune::FieldVector< double, 3 > xx, double radius, double d_left, double d_right) {
//distance to corners
double a1 = std::min(std::sqrt(std::pow(xx[0]-d_left, 2)+std::pow(xx[1]-d_left, 2)+std::pow(xx[2]-d_left, 2)), radius)/radius;
double a2 = std::min(std::sqrt(std::pow(xx[0]-d_left, 2)+std::pow(xx[1]-d_left, 2)+std::pow(xx[2]-d_right, 2)), radius)/radius;
double a3 = std::min(std::sqrt(std::pow(xx[0]-d_left, 2)+std::pow(xx[1]-d_right, 2)+std::pow(xx[2]-d_left, 2)), radius)/radius;
double a4 = std::min(std::sqrt(std::pow(xx[0]-d_left, 2)+std::pow(xx[1]-d_right, 2)+std::pow(xx[2]-d_right, 2)), radius)/radius;
double a5 = std::min(std::sqrt(std::pow(xx[0]-d_right, 2)+std::pow(xx[1]-d_left, 2)+std::pow(xx[2]-d_left, 2)), radius)/radius;
double a6 = std::min(std::sqrt(std::pow(xx[0]-d_right, 2)+std::pow(xx[1]-d_left, 2)+std::pow(xx[2]-d_right, 2)), radius)/radius;
double a7 = std::min(std::sqrt(std::pow(xx[0]-d_right, 2)+std::pow(xx[1]-d_right, 2)+std::pow(xx[2]-d_left, 2)), radius)/radius;
double a8 = std::min(std::sqrt(std::pow(xx[0]-d_right, 2)+std::pow(xx[1]-d_right, 2)+std::pow(xx[2]-d_right, 2)), radius)/radius;
double a_corner = a1 * a2 * a3 * a4 * a5 * a6 * a7 * a8;
//distance to edges
double b1 = 1.0;
double b2 = 1.0;
double b3 = 1.0;
double b4 = 1.0;
double b5 = 1.0;
double b6 = 1.0;
double b7 = 1.0;
double b8 = 1.0;
double b9 = 1.0;
double b10 = 1.0;
double b11 = 1.0;
double b12 = 1.0;
if (xx[0]>=d_left && xx[0] <= d_right ) {
  b1 = std::min(std::sqrt(std::pow(xx[1]-d_left, 2) + std::pow(xx[2]-d_left, 2)), radius)/radius;
  b1 /= (a1 * a5);
  b2 = std::min(std::sqrt(std::pow(xx[1]-d_left, 2) + std::pow(xx[2]-d_right, 2)), radius)/radius;
  b2 /= (a2 * a6);
  b3 = std::min(std::sqrt(std::pow(xx[1]-d_right, 2) + std::pow(xx[2]-d_left, 2)), radius)/radius;
  b3 /= (a3 * a7);
  b4 = std::min(std::sqrt(std::pow(xx[1]-d_right, 2) + std::pow(xx[2]-d_right, 2)), radius)/radius;
  b4 /= (a4 * a8);
}
if (xx[1]>=d_left && xx[1] <= d_right ) {
  b5 = std::min(std::sqrt(std::pow(xx[0]-d_left, 2) + std::pow(xx[2]-d_left, 2)), radius)/radius;
  b5 /= (a1 * a3);
  b6 = std::min(std::sqrt(std::pow(xx[0]-d_left, 2) + std::pow(xx[2]-d_right, 2)), radius)/radius;
  b6 /= (a2 * a4);
  b7 = std::min(std::sqrt(std::pow(xx[0]-d_right, 2) + std::pow(xx[2]-d_left, 2)), radius)/radius;
  b7 /= (a5 * a7);
  b8 = std::min(std::sqrt(std::pow(xx[0]-d_right, 2) + std::pow(xx[2]-d_right, 2)), radius)/radius;
  b8 /= (a6 * a8);
}
if (xx[2]>=d_left && xx[2] <= d_right ) {
  b9 = std::min(std::sqrt(std::pow(xx[1]-d_left, 2) + std::pow(xx[0]-d_left, 2)), radius)/radius;
  b9 /= (a1 * a2);
  b10 = std::min(std::sqrt(std::pow(xx[1]-d_left, 2) + std::pow(xx[0]-d_right, 2)), radius)/radius;
  b10 /= (a5 * a6);
  b11 = std::min(std::sqrt(std::pow(xx[1]-d_right, 2) + std::pow(xx[0]-d_left, 2)), radius)/radius;
  b11 /= (a3 * a4);
  b12 = std::min(std::sqrt(std::pow(xx[1]-d_right, 2) + std::pow(xx[0]-d_right, 2)), radius)/radius;
  b12 /= (a7 * a8);
}
double b_edges = b1 * b2 * b3 * b4 * b5 * b6 * b7 * b8 * b9 * b10 * b11 * b12;
return a_corner * b_edges;
}


template< class MacroGridViewType, class CellGridType, class InclusionGridViewType, int polynomialOrder >
class HMMMaxwellDiscretization {
public:
  typedef typename MacroGridViewType::ctype                                   MacroDomainFieldType;
  typedef typename MacroGridViewType::template Codim<0>::Entity               MacroEntityType;
  typedef double                                                              RangeFieldType;
  static const size_t       dimDomain = MacroGridViewType::dimension;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename MacroGridViewType::Intersection >                            BoundaryInfoType;
  typedef Dune::Stuff::Functions::Constant< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >                 MacroConstFct;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, 1 >        MacroScalarFct;
  typedef Dune::Stuff::LocalizableFunctionInterface< MacroEntityType, MacroDomainFieldType, dimDomain, double, dimRange > MacroVectorFct;
  typedef std::function< bool(const MacroGridViewType&, const MacroEntityType&) >                                         MacroFilterType;

  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType                 RealMatrixType;
  typedef Dune::Stuff::LA::Container< double, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType                 RealVectorType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< std::complex< double >, Dune::Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;

  typedef Dune::GDT::Spaces::Nedelec::PdelabBased< MacroGridViewType, polOrder, double, dimRange > SpaceType;
  typedef Dune::GDT::DiscreteFunction< SpaceType, RealVectorType >                                 DiscreteFunctionType;

  typedef Dune::GDT::Operators::CurlCellReconstruction< SpaceType, CellGridType >                       CurlCellProblem;
  typedef Dune::GDT::Operators::IdCellReconstruction< SpaceType, CellGridType >                         IdCellProblem;
  typedef Dune::GDT::MaxwellInclusionCell< InclusionGridViewType >                                      InclusionCellProblem;
  typedef Dune::Stuff::Grid::BoundaryInfoInterface< typename InclusionGridViewType::Intersection >      InclusionBoundaryInfoType;
  typedef typename CurlCellProblem::PeriodicEntityType                                                  CellEntityType;
  typedef typename CurlCellProblem::DomainFieldType                                                     CellDomainFieldType;
  typedef typename CurlCellProblem::ScalarFct                                                           CellScalarFct;
  typedef typename CurlCellProblem::FilterType                                                          CellFilterType;
  typedef Dune::Stuff::Functions::Constant< CellEntityType, CellDomainFieldType, dimDomain, double, 1 > CellConstFct;
  typedef typename InclusionCellProblem::ConstantFct                                                    InclusionConstantFct;

  typedef Dune::GDT::DiscreteFunction< typename CurlCellProblem::CellSpaceType, RealVectorType >          CurlCellDiscreteFctType;
  typedef Dune::GDT::DiscreteFunction< typename IdCellProblem::CellSpaceType, RealVectorType >            IdCellDiscreteFctType;
  typedef Dune::GDT::DiscreteFunction< typename InclusionCellProblem::SpaceType, RealVectorType >         InclusionCellDiscreteFctType;

  typedef typename CurlCellProblem::CellSolutionStorageType       AllCurlSolutionsStorageType;
  typedef typename IdCellProblem::CellSolutionStorageType         AllIdSolutionsStorageType;
  typedef typename InclusionCellProblem::CellSolutionStorageType  AllInclusionSolutionsStorageType;

  HMMMaxwellDiscretization(const MacroGridViewType& macrogridview,
                           CellGridType& cellgrid,
                           const InclusionGridViewType& inclusion_gridview,
                           const BoundaryInfoType& info,
                           const InclusionBoundaryInfoType& incl_info,
                           const CellScalarFct& mu_diel,
                           const CellScalarFct& mu_incl_real,
                           const CellScalarFct& mu_incl_imag,
                           const double& wavenumber,
                           const MacroVectorFct& bdry_real,
                           const MacroVectorFct& bdry_imag,
                           MacroFilterType filter_scatterer,
                           MacroFilterType filter_outside,
                           CellFilterType filter_inclusion,
                           const CellScalarFct& divparam,
                           const CellScalarFct& stabil,
                           const MacroScalarFct& mu_diel_macro,
                           const MacroScalarFct& mu_incl_real_macro,
                           const MacroScalarFct& mu_incl_imag_macro)
    : coarse_space_(macrogridview)
    , bdry_info_(info)
    , incl_bdry_info_(incl_info)
    , macro_mu_diel_(mu_diel_macro)
    , macro_mu_incl_real_(mu_incl_real_macro)
    , macro_mu_incl_imag_(mu_incl_imag_macro)
    , bdry_real_(bdry_real)
    , bdry_imag_(bdry_imag)
    , periodic_mu_diel_(mu_diel)
    , periodic_mu_incl_real_(mu_incl_real)
    , periodic_mu_incl_imag_(mu_incl_imag)
    , div_param_(divparam)
    , stabil_param_(stabil)
    , wavenumber_(wavenumber)
    , k_squared_(wavenumber_ * wavenumber_)
    , k_squared_neg_(-1*wavenumber_*wavenumber_)
    , filter_scatterer_(filter_scatterer)
    , filter_outside_(filter_outside)
    , filter_inclusion_(filter_inclusion)
    , curl_cell_(coarse_space_, cellgrid, periodic_mu_diel_, div_param_, stabil_param_, filter_inclusion_)
    , id_cell_(coarse_space_, cellgrid, k_squared_, stabil_param_, filter_inclusion_)
    , inclusion_cell_(inclusion_gridview, periodic_mu_incl_real_, periodic_mu_incl_imag_, k_squared_neg_, incl_bdry_info_)
    , is_assembled_(false)
    , system_matrix_real_(0,0)
    , system_matrix_imag_(0,0)
    , system_matrix_(0,0)
    , rhs_vector_real_(0)
    , rhs_vector_imag_(0)
    , rhs_vector_(0)
  {}

  const SpaceType& space() const
  {
    return coarse_space_;
  }

  const typename CurlCellProblem::CellSpaceType& curl_cell_space() const
  {
    return curl_cell_.cell_space();
  }

  const typename IdCellProblem::CellSpaceType& id_cell_space() const
  {
    return id_cell_.cell_space();
  }

  const typename InclusionCellProblem::SpaceType& inclusion_cell_space() const
  {
    return inclusion_cell_.cell_space();
  }


  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = coarse_space_.compute_volume_pattern();
      system_matrix_real_ = RealMatrixType(coarse_space_.mapper().size(), coarse_space_.mapper().size(), sparsity_pattern);
      system_matrix_imag_ = RealMatrixType(coarse_space_.mapper().size(), coarse_space_.mapper().size(), sparsity_pattern);
      system_matrix_ = MatrixType(coarse_space_.mapper().size(), coarse_space_.mapper().size());
      rhs_vector_real_ = RealVectorType(coarse_space_.mapper().size());
      rhs_vector_imag_ = RealVectorType(coarse_space_.mapper().size());
      rhs_vector_ = VectorType(coarse_space_.mapper().size());
      SystemAssembler< SpaceType > walker(coarse_space_);

      //rhs
      typedef LocalFunctional::Codim1Integral< LocalEvaluation::ProductTangential < MacroVectorFct > > LocalFunctionalType;
      LocalFunctionalType bdry_fctnal_real(bdry_real_);
      LocalFunctionalType bdry_fctnal_imag(bdry_imag_);
      const LocalAssembler::Codim1Vector< LocalFunctionalType > bdry_vector_real(bdry_fctnal_real);
      const LocalAssembler::Codim1Vector< LocalFunctionalType > bdry_vector_imag(bdry_fctnal_imag);
      walker.add(bdry_vector_real, rhs_vector_real_, new Stuff::Grid::ApplyOn::NeumannIntersections< MacroGridViewType >(bdry_info_));
      walker.add(bdry_vector_imag, rhs_vector_imag_, new Stuff::Grid::ApplyOn::NeumannIntersections< MacroGridViewType >(bdry_info_));


      //solve cell problems
      AllCurlSolutionsStorageType curl_cell_solutions(dimDomain);
      for (auto& it : curl_cell_solutions) {
        std::vector<DiscreteFunction< typename CurlCellProblem::CellSpaceType, RealVectorType > >
                  it1(1, DiscreteFunction< typename CurlCellProblem::CellSpaceType, RealVectorType >(curl_cell_.cell_space()));
        it = DSC::make_unique< typename CurlCellProblem::CellDiscreteFunctionType >(it1);
      }
      std::cout<< "computing curl cell problems"<< std::endl;
      curl_cell_.compute_cell_solutions(curl_cell_solutions);
      AllIdSolutionsStorageType id_cell_solutions(dimDomain);
      for (auto& it : id_cell_solutions) {
        std::vector<DiscreteFunction< typename IdCellProblem::CellSpaceType, RealVectorType > >
                  it1(1, DiscreteFunction< typename IdCellProblem::CellSpaceType, RealVectorType >(id_cell_.cell_space()));
        it = DSC::make_unique< typename IdCellProblem::CellDiscreteFunctionType >(it1);
      }
      std::cout<< "computing id cell problems"<< std::endl;
      id_cell_.compute_cell_solutions(id_cell_solutions);
      AllInclusionSolutionsStorageType inclusion_cell_solutions(dimDomain);
      for (auto& it : inclusion_cell_solutions) {
        std::vector<DiscreteFunction< typename InclusionCellProblem::SpaceType, RealVectorType > >
                  it1(2, DiscreteFunction< typename InclusionCellProblem::SpaceType, RealVectorType >(inclusion_cell_.cell_space()));
        it = DSC::make_unique< typename InclusionCellProblem::CellDiscreteFunctionType >(it1);
      }
      std::cout<< "computing inclusion cell problems"<< std::endl;
      inclusion_cell_.compute_cell_solutions(inclusion_cell_solutions);

      //lhs in scatterer
      typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMCurlcurlPeriodic< CellScalarFct, CurlCellProblem > > HMMCurlOperator;
      typedef LocalOperator::Codim0Integral< LocalEvaluation::HMMMaxwellIdPeriodic< CellScalarFct, IdCellProblem, InclusionCellProblem > > HMMIdOperator;
      HMMCurlOperator hmmcurl(curl_cell_, periodic_mu_diel_, div_param_, macro_mu_diel_, curl_cell_solutions, filter_inclusion_);
      HMMIdOperator hmmid_real(id_cell_, inclusion_cell_, periodic_mu_incl_real_, periodic_mu_incl_imag_, wavenumber_, true, macro_mu_incl_real_, macro_mu_incl_imag_, id_cell_solutions,
                               inclusion_cell_solutions, filter_inclusion_);
      HMMIdOperator hmmid_imag(id_cell_, inclusion_cell_, periodic_mu_incl_real_, periodic_mu_incl_imag_, wavenumber_, false, macro_mu_incl_real_, macro_mu_incl_imag_, id_cell_solutions, 
                               inclusion_cell_solutions, filter_inclusion_);
      LocalAssembler::Codim0Matrix< HMMCurlOperator > hmm_curl_assembler(hmmcurl);
      LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_real_assembler(hmmid_real);
      LocalAssembler::Codim0Matrix< HMMIdOperator > hmm_id_imag_assembler(hmmid_imag);
      assert(filter_scatterer_);
      walker.add(hmm_curl_assembler, system_matrix_real_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_scatterer_));
      walker.add(hmm_id_real_assembler, system_matrix_real_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_scatterer_));
      walker.add(hmm_id_imag_assembler, system_matrix_imag_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_scatterer_));

      //lhs outside scatterer
      MacroConstFct one(1.0);
      //std::cout<< "watch out: diffusion_param outside scatterer is not 1.0!!!!"<<std::endl;
      assert(filter_outside_);
      typedef GDT::Operators::CurlCurl< MacroConstFct, RealMatrixType, SpaceType > CurlOperatorType;
      CurlOperatorType curl_operator_real(one, system_matrix_real_, coarse_space_);
      walker.add(curl_operator_real, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_outside_));
      //identity part
      MacroConstFct wavenumber_fct(-1.0*wavenumber_);
      MacroConstFct wavenumber_fct_squared(-1.0*wavenumber_ * wavenumber_);
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< MacroConstFct > > IdOperatorType;
      const IdOperatorType identity_operator_real(wavenumber_fct_squared);
      const LocalAssembler::Codim0Matrix< IdOperatorType > idMatrixAssembler_real(identity_operator_real);
      walker.add(idMatrixAssembler_real, system_matrix_real_, new Stuff::Grid::ApplyOn::FilteredEntities< MacroGridViewType >(filter_outside_));

      //boundary part for complex Robin-type condition
      typedef LocalOperator::Codim1BoundaryIntegral< LocalEvaluation::ProductTangential< MacroConstFct > > BdryOperatorType;
      const BdryOperatorType bdry_operator(wavenumber_fct);
      const LocalAssembler::Codim1BoundaryMatrix< BdryOperatorType > bdry_assembler(bdry_operator);
      walker.add(bdry_assembler, system_matrix_imag_, new Stuff::Grid::ApplyOn::NeumannIntersections< MacroGridViewType >(bdry_info_));

      //assemble
      std::cout<< "macro assembly" <<std::endl;
      walker.assemble();

      //assembly of total (complex) matrix and vector
      std::complex< double > im(0.0, 1.0);
      system_matrix_.backend() = system_matrix_imag_.backend().template cast< std::complex< double > >();
      system_matrix_.scal(im);
      system_matrix_.backend() += system_matrix_real_.backend().template cast< std::complex< double > >();
      rhs_vector_.backend() = rhs_vector_imag_.backend().template cast< std::complex< double > >();
      rhs_vector_.scal(im);
      rhs_vector_.backend() += rhs_vector_real_.backend().template cast< std::complex< double > > ();

      is_assembled_ = true;
    }
  }  //assemble

  bool assembled() const
  {
    return is_assembled_;
  }

  Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > effective_curl() const
  {
    return curl_cell_.effective_matrix();
  }

  std::vector< Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > > effective_id() const
  {
    auto inclusion_matrix = inclusion_cell_.effective_matrix();
    auto real_matrix = id_cell_.effective_matrix() + inclusion_matrix[0];
    return std::vector< Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > >({real_matrix, inclusion_matrix[1]});
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
    if (!is_assembled_)
      assemble();
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "bicgstab.diagonal");
  }

  /**
   * @brief solve_and_correct solves the HMM problem and directly computes the (discrete) correctors
   * @param macro_solution a vector (2 items, first for real, second for imaginary part) of DiscreteFunction (the space has to be given),
   *  where the macro_solution of the HMM will be saved to
   * @return a pair of PeriodicCorrector objects for the curl and identity corrector function
   */
/*  std::pair< Dune::GDT::PeriodicCorrector< DiscreteFunctionType, EllipticCellDiscreteFctType >, Dune::GDT::PeriodicCorrector< DiscreteFunctionType, InclusionCellDiscreteFctType > >
    solve_and_correct(std::vector< DiscreteFunctionType >& macro_solution)
  {
    if (!is_assembled_)
      assemble();
    VectorType solution(coarse_space_.mapper().size());
    Dune::Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, solution, "bicgstab.diagonal");
    //get real and imaginary part and make discrete functions
    RealVectorType solution_real(coarse_space_.mapper().size());
    RealVectorType solution_imag(coarse_space_.mapper().size());
    solution_real.backend() = solution.backend().real();
    solution_imag.backend() = solution.backend().imag();
    assert(macro_solution.size() > 1);
    macro_solution[0].vector() = solution_real;
    macro_solution[1].vector() = solution_imag;
    //compute cell problems
    AllEllipticSolutionsStorageType elliptic_cell_solutions(dimDomain);
    for (auto& it : elliptic_cell_solutions) {
      std::vector< EllipticCellDiscreteFctType >it1(1, EllipticCellDiscreteFctType(elliptic_cell_.cell_space()));
      it = DSC::make_unique< typename EllipticCellProblem::CellDiscreteFunctionType >(it1);
    }
    std::cout<< "computing elliptic cell problems"<< std::endl;
    elliptic_cell_.compute_cell_solutions(elliptic_cell_solutions);
    AllInclusionSolutionsStorageType inclusion_cell_solutions(1);
    for (auto& it : inclusion_cell_solutions) {
      std::vector< InclusionCellDiscreteFctType>it1(2, InclusionCellDiscreteFctType(inclusion_cell_.cell_space()));
      it = DSC::make_unique< typename InclusionCellProblem::CellDiscreteFunctionType >(it1);
    }
    std::cout<< "computing inclusion cell problems"<< std::endl;
    inclusion_cell_.compute_cell_solutions(inclusion_cell_solutions);

    std::vector< std::vector< EllipticCellDiscreteFctType > > elliptic_cell_functions(dimDomain,
                                                                                      std::vector< EllipticCellDiscreteFctType >(1, EllipticCellDiscreteFctType(elliptic_cell_.cell_space())));
    std::vector< std::vector< InclusionCellDiscreteFctType > > incl_cell_functions(1,
                                                                                   std::vector< InclusionCellDiscreteFctType >(2, InclusionCellDiscreteFctType(inclusion_cell_.cell_space())));
    for (size_t ii = 0; ii < dimDomain; ++ii){
      elliptic_cell_functions[ii][0].vector() = elliptic_cell_solutions[ii]->operator[](0).vector();
    }
    incl_cell_functions[0][0].vector() = inclusion_cell_solutions[0]->operator[](0).vector();
    incl_cell_functions[0][1].vector() = inclusion_cell_solutions[0]->operator[](1).vector();
    //build correctors
    return std::make_pair(Dune::GDT::PeriodicCorrector< DiscreteFunctionType, EllipticCellDiscreteFctType >(macro_solution, elliptic_cell_functions, "elliptic"),
                          Dune::GDT::PeriodicCorrector< DiscreteFunctionType, InclusionCellDiscreteFctType >(macro_solution, incl_cell_functions, "id_incl"));
  } //solve and correct
*/
  /** \brief computes the error between a reference solution (to the heterogeneous problem, on a fine grid) to the HMM approximation
   *
   * the HMM approximation is turned into a \ref DeltaCorrectorHelmholtz and the L2 or H1 seminorms can be requested
   * \tparam ReferenceFunctionImp Type for the discrete reference solution
   * \tparam CoarseFunctionImp Type for the macroscopic part of the HMM approximation
   * \tparam EllipticCellFunctionImp Type for the solutions to the cell problem for the gradient
   * \tparam InclusionCellFunctionImp Type for the solutions to the cell problem for the identity part in the inclusions
   */
/*  template< class ReferenceFunctionImp, class CoarseFunctionImp, class EllipticCellFunctionImp, class InclusionCellFunctionImp >
  RangeFieldType reference_error(std::vector< ReferenceFunctionImp >& reference_sol,
                                 Dune::GDT::PeriodicCorrector< CoarseFunctionImp, EllipticCellFunctionImp >& elliptic_corrector,
                                 Dune::GDT::PeriodicCorrector< CoarseFunctionImp, InclusionCellFunctionImp >& inclusion_corrector,
                                 double delta, std::string type)
  {
    //build DeltaCorrectorHelmholtz
    typedef Dune::GDT::DeltaCorrectorHelmholtz< CoarseFunctionImp, EllipticCellFunctionImp, InclusionCellFunctionImp > DeltaCorrectorType;
    DeltaCorrectorType corrector_real(elliptic_corrector.macro_function(), elliptic_corrector.cell_solutions(), inclusion_corrector.cell_solutions(),
                                      filter_scatterer_, filter_inclusion_, wavenumber_, delta, "real");
    DeltaCorrectorType corrector_imag(elliptic_corrector.macro_function(), elliptic_corrector.cell_solutions(), inclusion_corrector.cell_solutions(),
                                      filter_scatterer_, filter_inclusion_, wavenumber_, delta, "imag");
    //build errors
    typedef Dune::Stuff::Functions::Difference< ReferenceFunctionImp, DeltaCorrectorType > DifferenceFunctionType;
    DifferenceFunctionType error_real(reference_sol[0], corrector_real);
    DifferenceFunctionType error_imag(reference_sol[1], corrector_imag);
    //compute errors depending on type
    RangeFieldType result = 0;
    if(type == "l2") {  //maybe weigh it with k^2
      Dune::GDT::Products::L2< typename ReferenceFunctionImp::SpaceType::GridViewType > l2_product(reference_sol[0].space().grid_view());
      result += l2_product.apply2(error_real, error_real);
      result += l2_product.apply2(error_imag, error_imag);
      return std::sqrt(result);
    }
    else if(type == "h1semi") {
      Dune::GDT::Products::H1Semi< typename ReferenceFunctionImp::SpaceType::GridViewType > h1_product(reference_sol[0].space().grid_view());
      result += h1_product.apply2(error_real, error_real);
      result += h1_product.apply2(error_imag, error_imag);
      return std::sqrt(result);
    }
    else
      DUNE_THROW(Dune::NotImplemented, "This type of norm is not implemented");
  } //reference error
*/
private:
  const SpaceType                     coarse_space_;
  const BoundaryInfoType&             bdry_info_;
  const InclusionBoundaryInfoType&    incl_bdry_info_;
  const MacroScalarFct&               macro_mu_diel_;
  const MacroScalarFct&               macro_mu_incl_real_;
  const MacroScalarFct&               macro_mu_incl_imag_;
  const MacroVectorFct&               bdry_real_;
  const MacroVectorFct&               bdry_imag_;
  const CellScalarFct&                periodic_mu_diel_;
  const CellScalarFct&                periodic_mu_incl_real_;
  const CellScalarFct&                periodic_mu_incl_imag_;
  const CellScalarFct&                div_param_;
  const CellScalarFct&                stabil_param_;
  const double                        wavenumber_;
  CellConstFct                        k_squared_;
  InclusionConstantFct                k_squared_neg_;
  const MacroFilterType               filter_scatterer_;
  const MacroFilterType               filter_outside_;
  const CellFilterType                filter_inclusion_;
  const CurlCellProblem               curl_cell_;
  const IdCellProblem                 id_cell_;
  const InclusionCellProblem          inclusion_cell_;
  mutable bool                        is_assembled_;
  mutable RealMatrixType              system_matrix_real_;
  mutable RealMatrixType              system_matrix_imag_;
  mutable MatrixType                  system_matrix_;
  mutable RealVectorType              rhs_vector_real_;
  mutable RealVectorType              rhs_vector_imag_;
  mutable VectorType                  rhs_vector_;
}; //class HMMHelmholtzDiscretization

#endif // DUNE_GDT_TEST_HMM_MAXWELL_HH
