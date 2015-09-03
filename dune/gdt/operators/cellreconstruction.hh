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


enum class ChooseCellProblem {Elliptic, CurlcurlDivreg};

//forward
template< class GridViewImp, int polynomialOrder, ChooseCellProblem >
class Cell;

namespace internal {

/**
 * \brief Traits for cell problems
 */
template< class GridViewImp, int polynomialOrder >
class CellTraits
{
public:
  typedef GridViewImp                                          GridViewType;
  typedef typename GridViewType::template Codim< 0 >::Entity   EntityType;
  typedef typename GridViewType::ctype                         DomainFieldType;
  static const size_t                                          dimDomain = GridViewType::dimension;

  static const unsigned int              polOrder = polynomialOrder;
  typedef double                         RangeFieldType;
  typedef std::complex< RangeFieldType > complextype;

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType    ComplexMatrixType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType    ComplexVectorType;
};


} //namespace internal

/**
 * \brief Class for an elliptic cell problem of the form \int param*(vector+\nabla ansatzfct)*\nabla testfct = 0
 * \tparam GridViewImp GridViewType for the grid of the unit (hyper)cube
 * \tparam polynomialorder polynomial order of the lagrange finite element space to use
 */
template< class GridViewImp, int polynomialOrder >
class Cell< GridViewImp, polynomialOrder, ChooseCellProblem::Elliptic >
{
public:
  typedef internal::CellTraits< GridViewImp, polynomialOrder> Traits;
  typedef typename Traits::EntityType                         EntityType;
  typedef typename Traits::DomainFieldType                    DomainFieldType;
  typedef typename Traits::RangeFieldType                     RangeFieldType;
  typedef typename Traits::MatrixType                         MatrixType;
  typedef typename Traits::VectorType                         VectorType;
  typedef typename Traits::ComplexMatrixType                  ComplexMatrixType;
  typedef typename Traits::ComplexVectorType                  ComplexVectorType;

  static const size_t       dimDomain = Traits::dimDomain;
  static const unsigned int polOrder = Traits::polOrder;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, RangeFieldType, 1 >                                SpaceType;

  Cell(const GridViewImp& gridview, const ScalarFct& kappa_real, const ScalarFct& kappa_imag)
    : space_(gridview)
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
    solver.apply(rhs_vector_total_, cell_sol, "lu.sparse");
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
      //clear rhs_vector
      rhs_vector_total_.scal(std::complex< double >(0.0, 0.0));
      reconstruct(unit_mat[ii], reconstr[ii]);
      tmp_rhs.emplace_back(rhs_vector_total_);
      tmp_rhs[ii].scal(std::complex< double >(-1.0, 0.0)); //necessary because rhs was -kappa*e_i and we want kappa*e_i
    }
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow_real = ret[0][ii];
      auto& retRow_imag = ret[1][ii];
      Dune::FieldVector< std::complex< RangeFieldType >, dimDomain > retRow;
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += tmp_rhs[jj] * reconstr[ii];
        retRow[jj] += reconstr[jj] * tmp_rhs[ii]; //for complex, this has to be conjugated!
        system_matrix_total_.mv(reconstr[ii], tmp_vector);
        retRow[jj] += reconstr[jj] * tmp_vector;
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
    const auto entity_it_end = space_.grid_view().template end<0>();
    //integrate
    for (auto entity_it = space_.grid_view().template begin<0>(); entity_it != entity_it_end; ++entity_it) {
      const auto& entity = *entity_it;
      const auto localparam_real = kappa_real_.local_function(entity);
      const auto localparam_imag = kappa_imag_.local_function(entity);
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
  } //averageparameter

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
}; //class Cell<... ChoosecellProblem::Elliptic >



/**
 * \brief Class for an elliptic cell problem of the form \int param*(vector+\nabla ansatzfct)*\nabla testfct = 0
 * \tparam GridPartImp GridPartType for the grid partition of the unit (hyper)cube
 * \tparam polynomialorder polynomial order of the lagrange finite element space to use
 */
template< class GridPartImp, int polynomialOrder >
class FemEllipticCell
{
public:
  typedef GridPartImp GridPartType;
  typedef typename GridPartType::GridViewType GridViewType;
  typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;
  typedef typename GridPartType::ctype DomainFieldType;
  typedef double                   RangeFieldType;
  typedef std::complex< RangeFieldType > complextype;

  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType MatrixType;
  typedef Dune::Stuff::LA::Container< RangeFieldType, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType VectorType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::MatrixType    ComplexMatrixType;
  typedef Dune::Stuff::LA::Container< complextype, Dune::Stuff::LA::ChooseBackend::eigen_sparse >::VectorType    ComplexVectorType;

  static const size_t       dimDomain = GridViewType::dimension;
  static const unsigned int polOrder = polynomialOrder;

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::FemBased< GridPartImp, 1, RangeFieldType, 1 >                                SpaceType;

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
    solver.apply(rhs_vector_total_, cell_sol, "lu.sparse");

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
 * \brief Class for a curlcurl cellproblem with divergence regularization
 * the problem has the type \int param*(vector+curl\ansatz)*\curl test + \div ansatz * \div test = 0
 * \sa Cell< GridViewImp, polynomialOrder, ChooseCellProblem::Elliptic > for explanation of template parameters
 */
template< class GridViewImp, int polynomialOrder >
class Cell< GridViewImp, polynomialOrder, ChooseCellProblem::CurlcurlDivreg >
{
public:
  typedef internal::CellTraits< GridViewImp, polynomialOrder> Traits;
  typedef typename Traits::EntityType                         EntityType;
  typedef typename Traits::DomainFieldType                    DomainFieldType;
  typedef typename Traits::RangeFieldType                     RangeFieldType;
  typedef typename Traits::MatrixType                         MatrixType;
  typedef typename Traits::VectorType                         VectorType;

  static const size_t       dimDomain = Traits::dimDomain;
  static const size_t       dimRange = dimDomain;
  static const unsigned int polOrder = Traits::polOrder;

  static_assert(dimDomain == 3, "This cell problem is only defined in 3d");

  typedef Dune::Stuff::LocalizableFunctionInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1 > ScalarFct;
  typedef Dune::GDT::Spaces::CG::PdelabBased< GridViewImp, 1, RangeFieldType, dimRange, 1 >                      SpaceType;  //will need FemSpace here as well

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

  /**
   * @brief assemble assmebles the system matrix of this problem
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

  /**
   * \brief computes the solution of the cell problem for given VectorType
   * \tparam RhsVectorType the type of vector to be given
   */
  template< class RhsVectorType >
  void reconstruct(RhsVectorType& externfctvalue, VectorType& cell_sol) const
  {
    if(!is_assembled_)
      assemble();
    // set up rhs, todo:clear rhs first!
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
    solver.apply(rhs_vector_, cell_sol, "lu.sparse");
  } //reconstruct

  /**
   * @brief effective_matrix computes the effective (homogenized) matrix
   * @return  the effective_matrix corresponding to the parameter mu_
   */
  Dune::FieldMatrix< DomainFieldType, dimDomain, dimDomain > effective_matrix() const
  {
    auto unit_mat = Dune::Stuff::Functions::internal::unit_matrix< double, dimDomain >();
    Dune::FieldMatrix< DomainFieldType, dimDomain, dimDomain > ret;
    if(!is_assembled_)
      assemble();
    const auto averageparam = averageparameter();
    //prepare temporary storage
    VectorType tmp_vector(space_.mapper().size());
    std::vector< VectorType > reconstr(dimDomain, tmp_vector);
    std::vector< VectorType > tmp_rhs;
    //compute solutions of cell problems
    for (size_t ii =0; ii < dimDomain; ++ii) {
      //clear rhs vector
      rhs_vector_.scal(0.0);
      reconstruct(unit_mat[ii], reconstr[ii]);
      tmp_rhs.emplace_back(rhs_vector_);
      tmp_rhs[ii].scal(-1.0); //necessary because rhs was -mu*e_i and we want mu*e_i
    }
    //compute matrix
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      auto& retRow = ret[ii];
      for (size_t jj = 0; jj < dimDomain; ++jj) {
        retRow[jj] += averageparam * unit_mat[ii][jj];
        retRow[jj] += tmp_rhs[jj] * reconstr[ii];
        retRow[jj] += reconstr[jj] * tmp_rhs[ii]; //for complex, this has to be conjugated!
        system_matrix_.mv(reconstr[ii], tmp_vector);
        retRow[jj] += reconstr[jj] * tmp_vector;
      }
    }
    return ret;
  } //effective_matrix()

  /**
   * @brief averageparameter averages mu_ over the unit cube
   * @return average of mu_ over the unit cube
   */
  const typename ScalarFct::RangeFieldType averageparameter() const
  {
    typename ScalarFct::RangeFieldType result(0.0);
    const auto entity_it_end = space_.grid_view().template end<0>();
    //integrate
    for (auto entity_it = space_.grid_view().template begin<0>(); entity_it != entity_it_end; ++entity_it) {
      const auto& entity = *entity_it;
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
  } //averageparameter

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
