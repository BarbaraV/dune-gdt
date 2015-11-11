// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH
#define DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH

#if !HAVE_DUNE_FEM
# error "This one requires dune-fem!"
#endif

#include <memory>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>

#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/common/ranges.hh>
#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container.hh>
#include <dune/stuff/la/solver.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/constant.hh>

#include <dune/gdt/spaces/cg/fem.hh>
#include <dune/gdt/localevaluation/elliptic.hh>
#include <dune/gdt/localevaluation/curlcurl.hh>
#include <dune/gdt/localevaluation/divdiv.hh>
#include <dune/gdt/localevaluation/product-l2deriv.hh>
#include <dune/gdt/localoperator/codim0.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/discretefunction/default.hh>

namespace Dune {
namespace GDT {
namespace Operators {


/**
 * \brief Class for an elliptic cell problem of the form \int param*(vector+\nabla ansatzfct)*\nabla testfct = 0
 * \tparam GridPartImp GridPartType for the grid partition of the unit (hyper)cube
 * \tparam polynomialorder polynomial order of the lagrange finite element space to use
 */
template< class GridPartImp, int polynomialOrder >
class FemEllipticCell
{
public:
  typedef GridPartImp                                            GridPartType;
  typedef typename GridPartType::GridViewType                    GridViewType;
  typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;
  typedef typename GridPartType::ctype                           DomainFieldType;

  typedef double                         RangeFieldType;
  typedef std::complex< RangeFieldType > complextype;

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType    ComplexMatrixType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType    ComplexVectorType;

  static const size_t       dimDomain = GridViewType::dimension;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::FemBased< GridPartImp, 1, RangeFieldType, 1 >                                   SpaceType;

  FemEllipticCell(const GridPartImp& gridpart, const ScalarFct& kappa_real, const ScalarFct& kappa_imag)
    : space_(gridpart)
    , kappa_real_(kappa_real)
    , kappa_imag_(kappa_imag)
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

  /**
   * @brief assemble assembles the system matrix of the problem
   * @note the right hand side is assmebled in reconstruct as in can change while the cell problem stays the same
   */
  void assemble() const
  {
    using namespace Dune;
    using namespace Dune::GDT;
    if(!is_assembled_) {
      //prepare
      Stuff::LA::SparsityPatternDefault sparsity_pattern = space_.compute_volume_pattern();
      system_matrix_real_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_imag_ = MatrixType(space_.mapper().size(), space_.mapper().size(), sparsity_pattern);
      system_matrix_total_ = ComplexMatrixType(space_.mapper().size(), space_.mapper().size());
      rhs_vector_real_ = VectorType(space_.mapper().size());
      rhs_vector_imag_ = VectorType(space_.mapper().size());
      rhs_vector_total_ = ComplexVectorType(space_.mapper().size());
      SystemAssembler< SpaceType > walker(space_);

      //lhs
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > > EllipticOp;
      EllipticOp ellipticop1(kappa_real_);
      EllipticOp ellipticop2(kappa_imag_);
      LocalAssembler::Codim0Matrix< EllipticOp > matrixassembler1(ellipticop1);
      LocalAssembler::Codim0Matrix< EllipticOp > matrixassembler2(ellipticop2);
      walker.add(matrixassembler1, system_matrix_real_);
      walker.add(matrixassembler2, system_matrix_imag_);
      //maybe add an identity term for stabilization

      walker.assemble();

      std::complex< RangeFieldType > im(0.0, 1.0);
      system_matrix_total_.backend() = system_matrix_imag_.backend().template cast< std::complex< RangeFieldType > >();
      system_matrix_total_.scal(im);
      system_matrix_total_.backend() += system_matrix_real_.backend().template cast< std::complex< RangeFieldType > >();

      is_assembled_ = true;
    }
  } //assemble

  bool is_assembled() const
  {
    return is_assembled_;
  }

  const ComplexMatrixType& system_matrix() const
  {
    return system_matrix_total_;
  }

  const ComplexVectorType& rhs_vector() const
  {
    return rhs_vector_total_;
  }

  /**
   * @brief computes the solution of the cell problem for a given vector
   * @note the given vector mostly is an evaluation of macroscopic (shape) functions and
   * the solution vector gives the coefficients of the thus reconstructed discrete function
   * \tparam RhsVectorType the type of the vector to be given
   */
  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, ComplexVectorType& cell_sol) const
  {
    if(!is_assembled_)
      assemble();
    //clear rhs for case that a reconstruction has been computed before
    rhs_vector_real_.scal(0.0);
    rhs_vector_imag_.scal(0.0);
    // set up rhs
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, typename RhsVectorType::field_type, dimDomain > ConstFct;
    auto externfctvalue1 = externfctvalue;
    externfctvalue1 *= -1.0;
    const ConstFct constrhs(externfctvalue1);
    typedef Stuff::Functions::Product< ScalarFct, ConstFct > RhsFuncType;
    const RhsFuncType rhsfunc_real(kappa_real_, constrhs);
    const RhsFuncType rhsfunc_imag(kappa_imag_, constrhs);
    typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2grad< RhsFuncType > > L2gradOp;
    L2gradOp l2gradop1(rhsfunc_real);
    L2gradOp l2gradop2(rhsfunc_imag);
    LocalAssembler::Codim0Vector< L2gradOp > vectorassembler1(l2gradop1);
    LocalAssembler::Codim0Vector< L2gradOp > vectorassembler2(l2gradop2);

    //assemble rhs
    SystemAssembler< SpaceType > walker(space_);
    walker.add(vectorassembler1, rhs_vector_real_);
    walker.add(vectorassembler2, rhs_vector_imag_);
    walker.assemble();

    std::complex< RangeFieldType > im(0.0, 1.0);
    rhs_vector_total_.backend() = rhs_vector_imag_.backend().template cast< std::complex< RangeFieldType > >();
    rhs_vector_total_.scal(im);
    rhs_vector_total_.backend() += rhs_vector_real_.backend().template cast< std::complex< RangeFieldType > >();

    //solve
    Stuff::LA::Solver< ComplexMatrixType > solver(system_matrix_total_);
    solver.apply(rhs_vector_total_, cell_sol, "bicgstab.diagonal");

    //compute average over cube
    VectorType cell_sol_real(space_.mapper().size());
    cell_sol_real.backend() = cell_sol.backend().real();
    VectorType cell_sol_imag(space_.mapper().size());
    cell_sol_imag.backend() = cell_sol.backend().imag();
    Dune::GDT::ConstDiscreteFunction< SpaceType, VectorType > cell_sol_discr_real(space_, cell_sol_real);
    Dune::GDT::ConstDiscreteFunction< SpaceType, VectorType > cell_sol_discr_imag(space_, cell_sol_imag);
    auto cell_average = average(cell_sol_discr_real, cell_sol_discr_imag);
    ComplexVectorType cell_average_vector(space_.mapper().size(), cell_average);
    cell_sol -= cell_average_vector;
  } //reconstruct

  /**
   * @brief effective_matrix computes the effective (or homogenized) matrix corresponding to the microscopic parameter kappa_
   * @return a 2-vector of matrices which gives the real and imaginary part of the effective matrix
   */
  std::vector< Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > > effective_matrix() const
  {
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    std::vector< Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > > ret(2);
    if(!is_assembled_)
      assemble();
    const auto averageparam = averageparameter();
    //prepare temporary storage
    ComplexVectorType tmp_vector(space_.mapper().size());
    std::vector< ComplexVectorType > reconstr(dimDomain, tmp_vector);
    std::vector< ComplexVectorType > tmp_rhs;
    //compute solutions of cell problems
    for (size_t ii =0; ii < dimDomain; ++ii) {
      reconstruct(unit_mat[ii], reconstr[ii]);
      tmp_rhs.emplace_back(rhs_vector());
      tmp_rhs[ii].scal(std::complex< double >(-1.0, 0.0)); //necessary because rhs was -kappa*e_i and we want kappa*e_i
    }
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow_real = ret[0][ii];
      auto& retRow_imag = ret[1][ii];
      Dune::FieldVector< std::complex< RangeFieldType >, dimDomain > retRow;
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += tmp_rhs[ii].dot(reconstr[jj]);
        retRow_real[jj] += retRow[jj].real();
        retRow_imag[jj] += retRow[jj].imag();
      }
    }
    return ret;
  } //effective_matrix()

  /**
   * @brief averageparameter averages the paramter function over the unit cube
   * @return  the average of kappa_
   */
  const std::complex< typename ScalarFct::RangeFieldType > averageparameter() const
  {
    std::complex< typename ScalarFct::RangeFieldType > result(0.0);
    result = average(kappa_real_, kappa_imag_);
    return result;
  } //averageparameter

  /**
   * @brief average averages a scalar complex function over the unit cube
   * @param function_real real part of the function
   * @param function_imag imaginary part of the function
   * @return  the average of function_real+i*function_imag
   */
  template< class FunctionType >
  const std::complex< RangeFieldType > average(FunctionType& function_real, FunctionType& function_imag) const
  {
    std::complex< RangeFieldType > result(0.0);
    const auto entity_it_end = space_.grid_view().template end<0>();
    //integrate
    for (auto entity_it = space_.grid_view().template begin<0>(); entity_it != entity_it_end; ++entity_it) {
      const auto& entity = *entity_it;
      const auto localparam_real = function_real.local_function(entity);
      const auto localparam_imag = function_imag.local_function(entity);
      const size_t int_order = localparam_real->order();  //we assume the real and imaginary part to have the same order atm
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
        const auto evaluation_result_real = localparam_real->evaluate(x);
        const auto evaluation_result_imag = localparam_imag->evaluate(x);
        const auto resultreal = evaluation_result_real[0] * quadrature_weight * integration_factor;
        const auto resultimag = evaluation_result_imag[0] * quadrature_weight * integration_factor;
        result += std::complex< double >(resultreal, resultimag);
      } //loop over quadrature points
    } //loop over entities
    return result;
  } //average

private:
  const SpaceType           space_;
  const ScalarFct&          kappa_real_;
  const ScalarFct&          kappa_imag_;
  mutable bool              is_assembled_;
  mutable MatrixType        system_matrix_real_;
  mutable MatrixType        system_matrix_imag_;
  mutable ComplexMatrixType system_matrix_total_;
  mutable VectorType        rhs_vector_real_;
  mutable VectorType        rhs_vector_imag_;
  mutable ComplexVectorType rhs_vector_total_;
}; //class FemEllipticCell


/**
 * \brief Class for a curlcurl cell problem of the form \int param*(vector+\curl ansatzfct)*\curl testfct +\div ansatzfct * \div testfct = 0
 * \tparam GridPartImp GridPartType for the grid partition of the unit (hyper)cube
 * \tparam polynomialorder polynomial order of the lagrange finite element space to use
 */
template< class GridPartImp, int polynomialOrder >
class FemCurlCell
{
public:
  typedef GridPartImp                                            GridPartType;
  typedef typename GridPartType::GridViewType                    GridViewType;
  typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;
  typedef typename GridPartType::ctype                           DomainFieldType;
  typedef double                                                 RangeFieldType;

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;

  static const size_t       dimDomain = GridViewType::dimension;
  static const size_t dimRange = dimDomain;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 >     ScalarFct;
  typedef Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 >              ConstantFct;
  typedef Dune::GDT::Spaces::CG::FemBased< GridPartImp, 1, RangeFieldType, dimRange >                                SpaceType;

  FemCurlCell(const GridPartImp& gridpart, const ScalarFct& mu,
              const ScalarFct& divparam = Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1>(0.01))
    : space_(gridpart)
    , mu_(mu)
    , div_param_(divparam)
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

  /**
   * @brief assemble assembles the system matrix of the problem
   * @note the right hand side is assembled in reconstruct as in can change while the cell problem stays the same
   */
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
      DivOp divop(div_param_);
      LocalAssembler::Codim0Matrix< DivOp > matrixassembler2(divop);
      walker.add(matrixassembler2, system_matrix_);

      //(maybe) add an identity term for stabilization
      typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ScalarFct > > IdOp;
      IdOp idop(ConstantFct(0.0001));
      LocalAssembler::Codim0Matrix< IdOp > matrixassembler3(idop);
      walker.add(matrixassembler3, system_matrix_);

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

  /**
   * @brief computes the solution of the cell problem for a given vector
   * @note the given vector mostly is an evaluation of macroscopic (shape) functions and
   * the solution vector gives the coefficients of the thus reconstructed discrete function
   * \tparam RhsVectorType the type of the vector to be given
   */
  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, VectorType& cell_sol) const
  {
    if(!is_assembled_)
      assemble();
    //clear rhs for case that a reconstruction has been computed before
    rhs_vector_.scal(0.0);
    // set up rhs
    typedef Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, typename RhsVectorType::field_type, dimDomain > ConstFct;
    auto externfctvalue1 = externfctvalue;
    externfctvalue1 *= -1.0;
    const ConstFct constrhs(externfctvalue1);
    typedef Stuff::Functions::Product< ScalarFct, ConstFct > RhsFuncType;
    const RhsFuncType rhsfunc(mu_, constrhs);
    typedef LocalFunctional::Codim0Integral< LocalEvaluation::L2curl< RhsFuncType > > L2curlOp;
    L2curlOp l2curlop(rhsfunc);
    LocalAssembler::Codim0Vector< L2curlOp > vectorassembler(l2curlop);

    //assemble rhs
    SystemAssembler< SpaceType > walker(space_);
    walker.add(vectorassembler, rhs_vector_);
    walker.assemble();

    //solve
    Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(rhs_vector_, cell_sol, "bicgstab.diagonal");

    //compute average over cube
    Dune::GDT::ConstDiscreteFunction< SpaceType, VectorType > cell_sol_discr(space_, cell_sol);
    auto cell_average = average(cell_sol_discr);
    for (size_t ii = 0; ii < cell_sol.size(); ++ii){
      if (ii % dimRange == 0)
        cell_sol.add_to_entry(ii, -cell_average[0]);
      else if (ii % dimRange == 1)
        cell_sol.add_to_entry(ii, -cell_average[1]);
      else
        cell_sol.add_to_entry(ii, -cell_average[2]);
    }

  } //reconstruct

  /**
   * @brief effective_matrix computes the effective (or homogenized) matrix corresponding to the microscopic parameter kappa_
   * @return a 2-vector of matrices which gives the real and imaginary part of the effective matrix
   */
  Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > effective_matrix() const
  {
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain >  ret;
    if(!is_assembled_)
      assemble();
    const auto averageparam = averageparameter();
    //prepare temporary storage
    VectorType tmp_vector(space_.mapper().size());
    std::vector< VectorType > reconstr(dimDomain, tmp_vector);
    std::vector< VectorType > tmp_rhs;
    //compute solutions of cell problems
    for (size_t ii =0; ii < dimDomain; ++ii) {
      reconstruct(unit_mat[ii], reconstr[ii]);
      tmp_rhs.emplace_back(rhs_vector());
      tmp_rhs[ii].scal(-1.0); //necessary because rhs was -mu*e_i and we want mu*e_i
    }
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += tmp_rhs[ii].dot(reconstr[jj]);
      }
    }
    return ret;
  } //effective_matrix()

  /**
   * @brief averageparameter averages the parameter function over the unit cube
   * @return  the average of mu_
   */
  const typename ScalarFct::RangeFieldType averageparameter() const
  {
    typename ScalarFct::RangeFieldType result(0.0);
    result = average(mu_);
    return result;
  } //averageparameter

  /**
   * @brief average averages a real-valued function over the unit cube
   * @param function_real real part of the function
   * @return  the average of function
   */
  template< class FunctionType >
  typename FunctionType::RangeType average(FunctionType& function) const
  {
    typename FunctionType::RangeType result(0.0);
    const auto entity_it_end = space_.grid_view().template end<0>();
    //integrate
    for (auto entity_it = space_.grid_view().template begin<0>(); entity_it != entity_it_end; ++entity_it) {
      const auto& entity = *entity_it;
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

private:
  const SpaceType           space_;
  const ScalarFct&          mu_;
  const ScalarFct&          div_param_;
  mutable bool              is_assembled_;
  mutable MatrixType        system_matrix_;
  mutable VectorType        rhs_vector_;
}; //class FemCurlCell


/** \brief Abstract base class for cell reconstructions / cell problems
 * \tparam CoarseSpaceType Type of space the reconstructions are computed from
 * \tparam CellGridType Type of grid for the unit cube
 * \tparam CellSpaceType Type of space the corrections lie in
 * \tparam iscomplex Boolean whether the parameters are complex
 */
template< class CoarseSpaceType, class CellGridType, class CellSpaceType, bool iscomplex >
class CellReconstruction
{
public:
  typedef typename CoarseSpaceType::GridViewType::template Codim<0>::Entity CoarseEntityType;
  typedef typename CoarseSpaceType::GridViewType::ctype                     CoarseDomainFieldType;
  static const size_t                                                       coarsedimDomain = CoarseSpaceType::GridViewType::dimension;
  typedef FieldVector< CoarseDomainFieldType, coarsedimDomain >             CoarseDomainType;

  typedef typename Dune::Fem::PeriodicLeafGridPart< CellGridType>                PeriodicGridPartType;
  typedef typename Dune::Fem::PeriodicLeafGridPart< CellGridType >::GridViewType PeriodicViewType;
  typedef typename PeriodicViewType::template Codim< 0 >::Entity                 PeriodicEntityType;
  typedef typename PeriodicGridPartType::ctype                                   DomainFieldType;
  static const size_t                                                            dimDomain = PeriodicGridPartType::dimension;
  static const size_t                                                            dimRange = CellSpaceType::dimRange;

  typedef double                         RangeFieldType;
  typedef std::complex< RangeFieldType > ComplexType;
  static const bool                      complex = iscomplex;

private:
  template< bool is_complex, bool anything = true >
  struct Helper {
    typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType RealMatrixType;
    typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
    typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType RealVectorType;
    typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;
  };

  template< bool anything >
  struct Helper< true, anything > {
    typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType RealMatrixType;
    typedef Dune::Stuff::LA::Container< ComplexType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType    MatrixType;
    typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType RealVectorType;
    typedef Dune::Stuff::LA::Container< ComplexType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType    VectorType;
  };

public:
  typedef typename Helper< complex >::RealMatrixType RealMatrixType;
  typedef typename Helper< complex >::MatrixType     MatrixType;
  typedef typename Helper< complex >::RealVectorType RealVectorType;
  typedef typename Helper< complex >::VectorType     VectorType;

  typedef std::vector< DiscreteFunction< CellSpaceType, RealVectorType > > CellDiscreteFunctionType;
  typedef std::vector< std::shared_ptr< CellDiscreteFunctionType > >       CellSolutionStorageType;

  CellReconstruction(const CoarseSpaceType& coarse_space, CellGridType& cell_grid, const bool complex_problem)
    : coarse_space_(coarse_space)
    , grid_part_(cell_grid)
    , cell_space_(grid_part_)
    , system_matrix_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_assembler_(cell_space_)
    , complex_(complex_problem)
  {}

  virtual ~CellReconstruction() {}

  /**
    * \defgroup haveto ``These methods have to be implemented.''
    * @{
    **/
  virtual void assemble_all_local_rhs(const CoarseEntityType& /*coarse_entity*/, CellSolutionStorageType& /*local_rhs*/, const CoarseDomainType& /*xx*/) const = 0;

  virtual void assemble_cell_solutions_rhs(CellSolutionStorageType& /*cell_solutions*/) const = 0;

  virtual void apply(const VectorType& /*current_rhs*/, CellDiscreteFunctionType& /*current_solution*/) const = 0;

  virtual void apply(const CellDiscreteFunctionType& /*current_rhs*/, CellDiscreteFunctionType& /*current_solution*/) const = 0;
  /*@}*/

public:
  /**
    * \defgroup provided ``These methods are provided by the interface.''
    * @{
    **/
  void assemble_all_local_rhs(const CoarseEntityType& coarse_entity, CellSolutionStorageType& local_rhs) const
  {
    const auto xx = coarse_entity.geometry().center();
    assemble_all_local_rhs(coarse_entity, local_rhs, xx);
  }

  virtual bool is_complex() const
  {
    return complex_;
  }

  virtual const CellSpaceType& cell_space() const
  {
    return cell_space_;
  }

  void apply(const VectorType& current_rhs, VectorType& current_solution_vector) const
  {
    if(!current_rhs.valid())
      DUNE_THROW(Dune::InvalidStateException, "RHS vector invalid!");
    Stuff::LA::Solver< MatrixType > solver(system_matrix_);
    solver.apply(current_rhs, current_solution_vector, "bicgstab.diagonal");
    if(!current_solution_vector.valid())
      DUNE_THROW(Dune::InvalidStateException, "Solution vector invalid!");
  }

  /** \brief averages a function over the unit cube
   *
   */
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

  /**
   * @brief compute_cell_solutions Computes the cwell solutions, i.e. the corrections for the unit vectors
   * @param cell_solutions Vector of pointers to discrete functions to store the results in
   */
  void compute_cell_solutions(CellSolutionStorageType& cell_solutions) const
  {
    assert(cell_solutions.size() > 0);
    //clear return argument
    if(!complex_) {
      for (auto& localSol : cell_solutions) {
        assert(localSol->size() > 0);
        localSol->operator[](0).vector() *= 0;
      }
    }
    else {
      for (auto& localSol : cell_solutions) {
        assert(localSol->size() > 1);
        localSol->operator[](0).vector() *= 0;
        localSol->operator[](1).vector() *= 0;
      }
    }
    CellSolutionStorageType all_cell_rhs(cell_solutions.size());
    for (auto& it : all_cell_rhs) {
      if (!complex_) {
        std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(1, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
        it = DSC::make_unique< CellDiscreteFunctionType >(it1);
      }
      else {
        std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(2, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
        it = DSC::make_unique< CellDiscreteFunctionType >(it1);
      }
    }
    assemble_cell_solutions_rhs(all_cell_rhs);
    for (auto ii : DSC::valueRange(cell_solutions.size())) {
      auto& current_rhs = *all_cell_rhs[ii];
      auto& current_solution = *cell_solutions[ii];
      apply(current_rhs, current_solution);
    }
  } //compute_cell_solutions

  /**
   * @brief solve_all_at_single_point Computes corrections of all base functions at given local point
   * @param coarse_entity Entity of the macroscopic grid the corrections are computed for
   * @param all_cell_solutions Vector of pointers to discrete functions to store the results in
   * @param xx Local (macroscopic) point the base functions are evaluated in
   */
  void solve_all_at_single_point(const CoarseEntityType& coarse_entity, CellSolutionStorageType& all_cell_solutions,
                                 const CoarseDomainType& xx) const
  {
    assert(all_cell_solutions.size() > 0);
    //clear return argument
    if(!complex_) {
      for (auto& localSol : all_cell_solutions) {
        assert(localSol->size() > 0);
        localSol->operator[](0).vector() *= 0;
      }
    }
    else {
      for (auto& localSol : all_cell_solutions) {
        assert(localSol->size() > 1);
        localSol->operator[](0).vector() *= 0;
        localSol->operator[](1).vector() *= 0;
      }
    }
    CellSolutionStorageType allLocalRHS(all_cell_solutions.size());
    for (auto& it : allLocalRHS) {
      if (!complex_){
        std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(1, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
        it = DSC::make_unique< CellDiscreteFunctionType >(it1);
      }
      else {
        std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(2, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
        it = DSC::make_unique< CellDiscreteFunctionType >(it1);
      }
    }
    assemble_all_local_rhs(coarse_entity, allLocalRHS, xx);
    for (auto i : DSC::valueRange(all_cell_solutions.size())) {
      auto& current_rhs = *allLocalRHS[i];
      auto& current_solution = *all_cell_solutions[i];
      apply(current_rhs, current_solution);
    }
  } //solve_all_at_single_point

  void solve_all_at_single_point(CoarseEntityType& coarse_entity, CellSolutionStorageType& all_cell_solutions) const
  {
    const auto xx = coarse_entity.geometry().center();
    solve_all_at_single_point(coarse_entity, all_cell_solutions, xx);
  }

  /**
   * @brief solve_for_all_quad_points Computes the corrections of all base functions at all quadrature points of the macroscopic grid
   * @param order Order of the quadrature rule to use
   * @param solutions_storage Map of (entity index, no. quadrature point) to the local corrections (stored in a vector of pointers to discrete functions)
   */
  void solve_for_all_quad_points(const size_t order, std::map< std::pair< size_t, size_t >, CellSolutionStorageType >& solutions_storage) const
  {
    for (const auto& entity : DSC::entityRange(coarse_space_.grid_view()) ) {
      //prepare
      CellSolutionStorageType local_solutions(coarse_space_.mapper().maxNumDofs());
      for (auto& it : local_solutions) {
        if (!complex_){
          std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(1, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
          it = DSC::make_unique< CellDiscreteFunctionType >(it1);
        }
        else {
          std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(2, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
          it = DSC::make_unique< CellDiscreteFunctionType >(it1);
        }
      }
      auto index = coarse_space_.grid_view().indexSet().index(entity);
      //get quadrature rule
      typedef Dune::QuadratureRules< DomainFieldType, dimDomain > VolumeQuadratureRules;
      typedef Dune::QuadratureRule< DomainFieldType, dimDomain > VolumeQuadratureType;
      const VolumeQuadratureType& volumeQuadrature = VolumeQuadratureRules::rule(entity.type(), boost::numeric_cast< int >(order));
      //loop over all quadrature points
      const auto quadPointEndIt = volumeQuadrature.end();
      size_t ii = 0;
      for (auto quadPointIt = volumeQuadrature.begin(); quadPointIt != quadPointEndIt; ++quadPointIt, ++ii) {
        const Dune::FieldVector< DomainFieldType, dimDomain > x = quadPointIt->position();
        solve_all_at_single_point(entity, local_solutions, x);
        auto key = std::make_pair(index, ii);
        solutions_storage.insert({key, local_solutions});
      } //loop over quadrature points
    } //walk the grid
  } //solve_for_all_quad_points
  /*@}*/

protected:
  const CoarseSpaceType                    coarse_space_;
  const PeriodicGridPartType               grid_part_;
  const CellSpaceType                      cell_space_;
  mutable MatrixType                       system_matrix_;
  mutable SystemAssembler< CellSpaceType > system_assembler_;
  const bool                               complex_;
}; //class CellReconstruction


/** \brief Class for the correction of the curl of the macroscopic solution
 * \tparam CoarseSpaceType Type of space the corrections are computed from
 * \tparam CellGridType Type of grid to use for the unit cube
 */
template< class CoarseSpaceType, class CellGridType >
class CurlCellReconstruction
  : public CellReconstruction< CoarseSpaceType, CellGridType,
                               typename Spaces::CG::FemBased< typename Fem::PeriodicLeafGridPart< CellGridType >, 1, double, 3 >, false >
{
public:
  typedef typename Spaces::CG::FemBased< typename Fem::PeriodicLeafGridPart< CellGridType >, 1, double, 3 > CellSpaceType;
private:
  typedef CurlCellReconstruction< CoarseSpaceType, CellGridType >                                           ThisType;
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

  typedef LocalOperator::Codim0Integral< LocalEvaluation::CurlCurl< ScalarFct > > CurlOperator;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::Divdiv< ScalarFct > >   DivOperator;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ScalarFct > >  IdOperator;

  typedef LocalAssembler::Codim0Matrix< CurlOperator > CurlAssembler;
  typedef LocalAssembler::Codim0Matrix< DivOperator >  DivAssembler;
  typedef LocalAssembler::Codim0Matrix< IdOperator>    IdAssembler;

  using BaseType::coarse_space_;
  using BaseType::grid_part_;
  using BaseType::cell_space_;
  using BaseType::system_matrix_;
  using BaseType::system_assembler_;

  CurlCellReconstruction(const CoarseSpaceType& coarse_space, CellGridType& cell_grid, const ScalarFct& mu,
                         const ScalarFct& divparam = Stuff::Functions::Constant< PeriodicEntityType, DomainFieldType, dimDomain, RangeFieldType, 1>(0.01),
                         const ScalarFct& idparam = Stuff::Functions::Constant< PeriodicEntityType, DomainFieldType, dimDomain, RangeFieldType, 1>(0.0001))
    : BaseType(coarse_space, cell_grid, false)
    , mu_(mu)
    , div_param_(divparam)
    , id_param_(idparam)
    , curl_op_(mu_)
    , div_op_(div_param_)
    , id_op_(id_param_)
    , curl_assembler_(curl_op_)
    , div_assembler_(div_op_)
    , id_assembler_(id_op_)
  {
    system_assembler_.add(curl_assembler_, system_matrix_);
    system_assembler_.add(div_assembler_, system_matrix_);
    //(maybe) add an identity term for stabilization
    system_assembler_.add(id_assembler_, system_matrix_);
  }

  /**
   * @brief assemble_all_local_rhs Assembles the rhs of the cell problems for all base functions on the entity
   * @param coarse_entity Entity of the macroscopic grid we want to compute the corrections for
   * @param cell_solutions Vector of pointers to discrete functions to store the results in
   * @param xx Local (macroscopic) point the base functions are evaluated at
   */
  void assemble_all_local_rhs(const CoarseEntityType& coarse_entity, CellSolutionStorageType& cell_solutions, const CoarseDomainType& xx) const override final
  {
    assert(cell_solutions.size() > 0 && "You have to pre-allocate space");
    const auto coarse_basefunction_set = coarse_space_.base_function_set(coarse_entity);
    std::vector< typename CoarseSpaceType::BaseFunctionSetType::JacobianRangeType > tmp_jacobs(coarse_basefunction_set.size());
    coarse_basefunction_set.jacobian(xx, tmp_jacobs);
    typedef std::vector< FieldVector< RangeFieldType, dimDomain > > VectorofVectors;
    VectorofVectors tmp_curls(coarse_basefunction_set.size());
    //compute curls
    for (size_t ii = 0; ii < coarse_basefunction_set.size(); ++ii) {
      tmp_curls[ii][0] = tmp_jacobs[ii][2][1] - tmp_jacobs[ii][1][2];
      tmp_curls[ii][1] = tmp_jacobs[ii][0][2] - tmp_jacobs[ii][2][0];
      tmp_curls[ii][2] = tmp_jacobs[ii][1][0] - tmp_jacobs[ii][0][1];
    }
    typedef GDT::Functionals::L2Volume< ScalarFct, VectorType, CellSpaceType, PeriodicViewType,
                                        LocalEvaluation::VectorL2curl< ScalarFct, VectorofVectors > > RhsFunctionalType;
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals(coarse_basefunction_set.size());
    for (size_t num_coarsebase = 0; num_coarsebase < coarse_basefunction_set.size(); ++num_coarsebase) {
      assert(cell_solutions[num_coarsebase]);
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2curl< ScalarFct, VectorofVectors> >
              local_rhs_functional(mu_, tmp_curls, num_coarsebase);
      auto& rhs_vector = cell_solutions[num_coarsebase]->operator[](0).vector();
      rhs_functionals[num_coarsebase] = DSC::make_unique<RhsFunctionalType>(mu_, rhs_vector, cell_space_, local_rhs_functional);
      system_assembler_.add(*rhs_functionals[num_coarsebase]);
    }
   system_assembler_.assemble();
  }

  /**
   * @brief assemble_cell_solutions_rhs Assembles rhs for computation of cell corrections
   * @param cell_solutions Vector of pointers to discrete functions to store the results in
   */
  void assemble_cell_solutions_rhs(CellSolutionStorageType& cell_solutions) const override final
  {
    assert(cell_solutions.size() > 0 && "You have to pre-allocate space");
    typedef FieldMatrix< RangeFieldType, dimDomain, dimDomain > VectorofVectors;
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    typedef GDT::Functionals::L2Volume< ScalarFct, VectorType, CellSpaceType, PeriodicViewType,
                                        LocalEvaluation::VectorL2curl< ScalarFct, VectorofVectors > > RhsFunctionalType;
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals(dimDomain);
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      assert(cell_solutions[ii]);
      assert(cell_solutions[ii]->size() > 0 && "This has to be a vector");
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2curl< ScalarFct, VectorofVectors> >
              local_rhs_functional(mu_, unit_mat, ii);
      auto& rhs_vector = cell_solutions[ii]->operator[](0).vector();
      rhs_functionals[ii] = DSC::make_unique<RhsFunctionalType>(mu_, rhs_vector, cell_space_, local_rhs_functional);
      system_assembler_.add(*rhs_functionals[ii]);
    }
    system_assembler_.assemble();
  } //assemble_cell_solutions_rhs

  void apply(const VectorType& current_rhs, CellDiscreteFunctionType& current_solution) const override final
  {
    assert(current_solution.size() > 0 && "This has to be a pre-allocated vector");
    BaseType::apply(current_rhs, current_solution[0].vector());
    //substract average
    auto cell_average = this->average(current_solution[0]);
    for (size_t ii = 0; ii < dimRange; ++ii) {
      for (size_t kk = ii; kk < current_solution[0].vector().size(); ) {
        current_solution[0].vector().add_to_entry(kk, -cell_average[ii]);
        kk += dimRange;
      }
    }
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
    CellSolutionStorageType cell_rhs(dimDomain);
    for (auto& it : cell_rhs) {
      std::vector<DiscreteFunction< CellSpaceType, VectorType > > it1(1, DiscreteFunction< CellSpaceType, VectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    assemble_cell_solutions_rhs(cell_rhs);
    CellSolutionStorageType cell_solutions(dimDomain);
    for (auto& it : cell_solutions) {
      std::vector<DiscreteFunction< CellSpaceType, VectorType > > it1(1, DiscreteFunction< CellSpaceType, VectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    for (size_t ii =0; ii < dimDomain; ++ii) {
      auto& current_rhs = *cell_rhs[ii];
      auto& current_solution = *cell_solutions[ii];
      apply(current_rhs, current_solution);
    }
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    const auto averageparam = this->average(mu_);
    FieldMatrix< RangeFieldType, dimDomain, dimDomain > ret;
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow = ret[ii];
      cell_rhs[ii]->operator[](0).vector().scal(-1.0);  //necessary because rhs was -mu*e_i and we want mu*e_i
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += (cell_rhs[ii]->operator[](0).vector()).dot(cell_solutions[jj]->operator[](0).vector());
      }
    }
    return ret;
  } //effective_matrix

private:
  const ScalarFct&                 mu_;
  const ScalarFct&                 div_param_;
  const ScalarFct&                 id_param_;
  mutable CurlOperator             curl_op_;
  mutable DivOperator              div_op_;
  mutable IdOperator               id_op_;
  mutable CurlAssembler            curl_assembler_;
  mutable DivAssembler             div_assembler_;
  mutable IdAssembler              id_assembler_;
}; //class CurlCellReconstruction


/** Class for the correction of the macroscopic solution itself
 * \tparam CoarseSpaceType Type of space the corrections are computed from
 * \tparam CellGridType Type of grid for the unit cube
 */
template< class CoarseSpaceType, class CellGridType >
class IdEllipticCellReconstruction
  : public CellReconstruction< CoarseSpaceType, CellGridType,
                               typename Spaces::CG::FemBased< typename Fem::PeriodicLeafGridPart< CellGridType >, 1, double, 1 >, true >
{
public:
  typedef typename Spaces::CG::FemBased< typename Fem::PeriodicLeafGridPart< CellGridType >, 1, double, 1 > CellSpaceType;
private:
  typedef IdEllipticCellReconstruction< CoarseSpaceType, CellGridType >                                     ThisType;
  typedef CellReconstruction< CoarseSpaceType, CellGridType, CellSpaceType, true >                          BaseType;
public:
  using typename BaseType::CoarseEntityType;
  using typename BaseType::CoarseDomainType;
  using typename BaseType::CellSolutionStorageType;
  using typename BaseType::CellDiscreteFunctionType;
  using typename BaseType::DomainFieldType;
  using typename BaseType::PeriodicGridPartType;
  using typename BaseType::PeriodicViewType;
  using typename BaseType::PeriodicEntityType;
  using typename BaseType::RealMatrixType;
  using typename BaseType::MatrixType;
  using typename BaseType::RealVectorType;
  using typename BaseType::VectorType;
  using typename BaseType::RangeFieldType;
  using BaseType::dimDomain;  

  typedef Dune::Stuff::LocalizableFunctionInterface< PeriodicEntityType, DomainFieldType, dimDomain, RangeFieldType, 1 >     ScalarFct;

  typedef LocalOperator::Codim0Integral< LocalEvaluation::Elliptic< ScalarFct > > EllipticOperator;
  typedef LocalAssembler::Codim0Matrix< EllipticOperator >                        LocalAssemblerType;
  typedef LocalOperator::Codim0Integral< LocalEvaluation::Product< ScalarFct > >  IdOperator;
  typedef LocalAssembler::Codim0Matrix< IdOperator >                              IdAssemblerType;

  using BaseType::coarse_space_;
  using BaseType::grid_part_;
  using BaseType::cell_space_;
  using BaseType::system_matrix_;
  using BaseType::system_assembler_;

  IdEllipticCellReconstruction(const CoarseSpaceType& coarse_space, CellGridType& cell_grid, const ScalarFct& kappa_real, const ScalarFct& kappa_imag,
                               const ScalarFct& idparam = Stuff::Functions::Constant< PeriodicEntityType, DomainFieldType, dimDomain, RangeFieldType, 1>(0.0001))
    : BaseType(coarse_space, cell_grid, true)
    , kappa_real_(kappa_real)
    , kappa_imag_(kappa_imag)
    , id_param_(idparam)
    , system_matrix_real_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , system_matrix_imag_(cell_space_.mapper().size(), cell_space_.mapper().size(), cell_space_.compute_volume_pattern())
    , elliptic_operator_real_(kappa_real_)
    , elliptic_operator_imag_(kappa_imag_)
    , id_operator_(id_param_)
    , local_assembler_real_(elliptic_operator_real_)
    , local_assembler_imag_(elliptic_operator_imag_)
    , id_assembler_(id_operator_)
  {
    system_assembler_.add(local_assembler_real_, system_matrix_real_);
    system_assembler_.add(local_assembler_imag_, system_matrix_imag_);
    system_assembler_.add(id_assembler_, system_matrix_real_);
  }

  /**
   * @brief assemble_all_local_rhs Assembles the rhs of the cell problems for all base functions on the entity
   * @param coarse_entity Entity of the macroscopic grid we want to compute the corrections for
   * @param cell_solutions Vector of pointers to discrete functions to store the results in
   * @param xx Local (macroscopic) point the base functions are evaluated at
   */
  void assemble_all_local_rhs(const CoarseEntityType& coarse_entity, CellSolutionStorageType& cell_solutions, const CoarseDomainType& xx) const override final
  {
    assert(cell_solutions.size() > 0 && "You have to pre-allocate space");
    const auto coarse_basefunction_set = coarse_space_.base_function_set(coarse_entity);
    typedef std::vector< typename CoarseSpaceType::BaseFunctionSetType::RangeType > VectorofVectors;
    VectorofVectors tmp_values(coarse_basefunction_set.size());
    coarse_basefunction_set.evaluate(xx, tmp_values);
    typedef GDT::Functionals::L2Volume< ScalarFct, RealVectorType, CellSpaceType, PeriodicViewType,
                                        LocalEvaluation::VectorL2grad< ScalarFct, VectorofVectors > > RhsFunctionalType;
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals_real(coarse_basefunction_set.size());
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals_imag(coarse_basefunction_set.size());
    for (size_t num_coarsebase = 0; num_coarsebase < coarse_basefunction_set.size(); ++num_coarsebase) {
      assert(cell_solutions[num_coarsebase]);
      assert(cell_solutions[num_coarsebase]->size() > 1);
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2grad< ScalarFct, VectorofVectors> >
              local_rhs_functional_real(kappa_real_, tmp_values, num_coarsebase);
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2grad< ScalarFct, VectorofVectors> >
              local_rhs_functional_imag(kappa_imag_, tmp_values, num_coarsebase);
      auto& rhs_vector_real = cell_solutions[num_coarsebase]->operator[](0).vector();
      auto& rhs_vector_imag = cell_solutions[num_coarsebase]->operator[](1).vector();
      rhs_functionals_real[num_coarsebase] = DSC::make_unique<RhsFunctionalType>(kappa_real_, rhs_vector_real, cell_space_, local_rhs_functional_real);
      rhs_functionals_imag[num_coarsebase] = DSC::make_unique<RhsFunctionalType>(kappa_imag_, rhs_vector_imag, cell_space_, local_rhs_functional_imag);
      system_assembler_.add(*rhs_functionals_real[num_coarsebase]);
      system_assembler_.add(*rhs_functionals_imag[num_coarsebase]);
    }
   system_assembler_.assemble();

   std::complex< RangeFieldType > im(0.0, 1.0);
   system_matrix_.backend() = system_matrix_imag_.backend().template cast< std::complex< RangeFieldType > >();
   system_matrix_.scal(im);
   system_matrix_.backend() += system_matrix_real_.backend().template cast< std::complex< RangeFieldType > >();
  }

  /**
   * @brief assemble_cell_solutions_rhs Assembles rhs for computation of cell corrections
   * @param cell_solutions Vector of pointers to discrete functions to store the results in
   */
  void assemble_cell_solutions_rhs(CellSolutionStorageType& cell_rhs) const override final
  {
    assert(cell_rhs.size() > 0 && "You have to pre-allocate space");
    typedef FieldMatrix< RangeFieldType, dimDomain, dimDomain > VectorofVectorsType;
    VectorofVectorsType unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    typedef GDT::Functionals::L2Volume< ScalarFct, RealVectorType, CellSpaceType, PeriodicViewType,
                                        LocalEvaluation::VectorL2grad< ScalarFct, VectorofVectorsType > > RhsFunctionalType;
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals_real(dimDomain);
    std::vector<std::unique_ptr< RhsFunctionalType > > rhs_functionals_imag(dimDomain);
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      assert(cell_rhs[ii]);
      assert(cell_rhs[ii]->size() > 1);
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2grad< ScalarFct, VectorofVectorsType> >
              local_rhs_functional_real(kappa_real_, unit_mat, ii);
      GDT::LocalFunctional::Codim0Integral<LocalEvaluation::VectorL2grad< ScalarFct, VectorofVectorsType> >
              local_rhs_functional_imag(kappa_imag_, unit_mat, ii);
      auto& rhs_vector_real = cell_rhs[ii]->operator[](0).vector();
      auto& rhs_vector_imag = cell_rhs[ii]->operator[](1).vector();
      rhs_functionals_real[ii] = DSC::make_unique<RhsFunctionalType>(kappa_real_, rhs_vector_real, cell_space_, local_rhs_functional_real);
      rhs_functionals_imag[ii] = DSC::make_unique<RhsFunctionalType>(kappa_imag_, rhs_vector_imag, cell_space_, local_rhs_functional_imag);
      system_assembler_.add(*rhs_functionals_real[ii]);
      system_assembler_.add(*rhs_functionals_imag[ii]);
    }
    system_assembler_.assemble();

    std::complex< RangeFieldType > im(0.0, 1.0);
    system_matrix_.backend() = system_matrix_imag_.backend().template cast< std::complex< RangeFieldType > >();
    system_matrix_.scal(im);
    system_matrix_.backend() += system_matrix_real_.backend().template cast< std::complex< RangeFieldType > >();
  } //assemble_cell_solutions_rhs

  void apply(const VectorType& current_rhs, CellDiscreteFunctionType& current_solution) const override final
  {
    VectorType tmp_solution(current_rhs.size());
    BaseType::apply(current_rhs, tmp_solution);
    assert(current_solution.size() > 1);
    current_solution[0].vector().backend() = tmp_solution.backend().real();
    current_solution[1].vector().backend() = tmp_solution.backend().imag();
    //substract average
    auto cell_average_real = this->average(current_solution[0]);
    auto cell_average_imag = this->average(current_solution[1]);
    current_solution[0].vector() -= RealVectorType(cell_space_.mapper().size(), cell_average_real);
    current_solution[1].vector() -= RealVectorType(cell_space_.mapper().size(), cell_average_imag);
  }

  void apply(const CellDiscreteFunctionType& current_rhs, CellDiscreteFunctionType& current_solution) const override final
  {
    VectorType tmp_rhs(current_rhs[0].vector().size());
    std::complex< RangeFieldType > im(0.0, 1.0);
    tmp_rhs.backend() = current_rhs[1].vector().backend().template cast< std::complex< RangeFieldType > >();
    tmp_rhs.scal(im);
    tmp_rhs.backend() += current_rhs[0].vector().backend().template cast< std::complex< RangeFieldType > >();
    apply(tmp_rhs, current_solution);
  }

  /**
   * @brief effective_matrix Computes the effective matrix belonging to this cell problem
   * @return Vector with real and imaginary part of effective matrix
   */
  std::vector< FieldMatrix< RangeFieldType, dimDomain, dimDomain > > effective_matrix() const
  {
    CellSolutionStorageType cell_rhs(dimDomain);
    for (auto& it : cell_rhs){
      std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(2, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    assemble_cell_solutions_rhs(cell_rhs);
    CellSolutionStorageType cell_solutions(dimDomain);
    for (auto& it : cell_solutions) {
      std::vector<DiscreteFunction< CellSpaceType, RealVectorType > > it1(2, DiscreteFunction< CellSpaceType, RealVectorType >(cell_space_));
      it = DSC::make_unique< CellDiscreteFunctionType >(it1);
    }
    for (size_t ii =0; ii < dimDomain; ++ii) {
      auto& current_rhs = *cell_rhs[ii];
      auto& current_solution = *cell_solutions[ii];
      apply(current_rhs, current_solution);
    }
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    const auto averageparam_real = BaseType::average(kappa_real_);
    const auto averageparam_imag = BaseType::average(kappa_imag_);
    std::vector< FieldMatrix< RangeFieldType, dimDomain, dimDomain > > ret(2);
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow_real = ret[0][ii];
      auto& retRow_imag = ret[1][ii];
      cell_rhs[ii]->operator[](0).vector().scal(-1.0);  //necessary because rhs was -kappa*e_i and we want kappa*e_i
      cell_rhs[ii]->operator[](1).vector().scal(-1.0);
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow_real[jj] += averageparam_real * unit_mat[ii][jj];
        retRow_imag[jj] += averageparam_imag * unit_mat[ii][jj];
        retRow_real[jj] += (cell_rhs[ii]->operator[](0).vector().dot(cell_solutions[jj]->operator[](0).vector())
                            - cell_rhs[ii]->operator[](1).vector().dot(cell_solutions[jj]->operator[](1).vector()));
        retRow_imag[jj] += (cell_rhs[ii]->operator[](1).vector().dot(cell_solutions[jj]->operator[](0).vector())
                            + cell_rhs[ii]->operator[](0).vector().dot(cell_solutions[jj]->operator[](1).vector()));
      }
    }
    return ret;
  } //effective_matrix

private:
  const ScalarFct&                 kappa_real_;
  const ScalarFct&                 kappa_imag_;
  const ScalarFct&                 id_param_;
  mutable RealMatrixType           system_matrix_real_;
  mutable RealMatrixType           system_matrix_imag_;
  mutable EllipticOperator         elliptic_operator_real_;
  mutable EllipticOperator         elliptic_operator_imag_;
  mutable IdOperator               id_operator_;
  mutable LocalAssemblerType       local_assembler_real_;
  mutable LocalAssemblerType       local_assembler_imag_;
  mutable IdAssemblerType          id_assembler_;
};


} //namespace Operators
} //namespace GDT
} //namespace Dune

#endif // DUNE_GDT_OPERATORS_CELLRECONSTRUCTION_HH
