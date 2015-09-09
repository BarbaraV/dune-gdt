// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <vector>
#include <string>


#include <dune/common/parallel/mpihelper.hh>

#include <dune/grid/alugrid.hh>

#include <dune/stuff/functions/global.hh>
#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/periodicview.hh>

#include <dune/gdt/products/l2.hh>
#include <dune/gdt/products/hcurl.hh>

#include "curlcurldiscretization.hh"

// ------------------------------------------------------------------------
#include <dune/fem/gridpart/periodicgridpart/periodicgridpart.hh>
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/grid/yaspgrid.hh>
//#include <dune/gdt/localevaluation/hmm.hh>
#include "hmm-discretization.hh"
// ------------------------------------------------------------------------


using namespace Dune;


int main(int argc, char** argv) {
  //instantiate mpimanager
  Fem::MPIManager::initialize(argc, argv);
  auto world = MPIHelper::getCommunicator();

  // instantiate alugrid
  typedef Dune::ALUGrid< 3, 3, simplex, conforming > GridType;
  Stuff::Grid::Providers::Cube< GridType > grid_provider(0.0, 1.0, 6);
  auto& grid = grid_provider.grid();
  typedef GridType::LeafGridView LeafGridView;
  auto leafView = grid.leafGridView();
  typedef LeafGridView::Codim< 0 >::Entity EntityType;



  //test real discretization
  typedef Stuff::Functions::Expression< EntityType, double, 3, double, 3 > ExpressionFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 1 >  ConstantFct;
  typedef Stuff::Functions::Constant< EntityType, double, 3, double, 3> VectorFct;

  const ConstantFct scalar1(1.0);
  const ConstantFct scalar2(-1.0);
/*  const ExpressionFct expsol("x", {"sin(pi*x[1])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[2])", "sin(pi*x[0])*sin(pi*x[1])"}, 2, "expectedsolution",
                             {{"0", "pi*cos(pi*x[1])*sin(pi*x[2])", "pi*sin(pi*x[1])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[2])", "0", "pi*sin(pi*x[0])*cos(pi*x[2])"},
                              {"pi*cos(pi*x[0])*sin(pi*x[1])", "pi*sin(pi*x[0])*cos(pi*x[1])", "0"}});
  const double pi = 3.141592653589793;
  const ConstantFct realweight(2*pi*pi-1.0);   //2*pi^2+scalar2
  const ConstantFct imagweight(1.0);           //scalar1

  const Stuff::Functions::Product< ConstantFct, ExpressionFct > freal(realweight, expsol);
  const Stuff::Functions::Product< ConstantFct, ExpressionFct > fimag(imagweight, expsol);

  //expsol.visualize(leafView, "expectedsolution", false);

  DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
  Discretization< LeafGridView, 1> realdiscr(leafView, bdry_info, scalar1, scalar2, scalar1, freal, fimag);
  typedef std::complex< double > complextype;
  std::cout<< "assembling"<<std::endl;
  realdiscr.assemble();
  Dune::Stuff::LA::Container< complextype >::VectorType sol;
  std::cout<<"solving with bicgstab.diagonal"<<std::endl;
  realdiscr.solve(sol);
  std::cout<< (realdiscr.system_matrix()*sol-realdiscr.rhs_vector()).sup_norm()<< std::endl;
  //realdiscr.visualize(sol, "discrete_solution", "discrete_solution");

  typedef Stuff::LA::Container< double, Stuff::LA::ChooseBackend::eigen_sparse>::VectorType VectorType;
  typedef GDT::Spaces::Nedelec::PdelabBased< LeafGridView, 1, double, 3> SpaceType;
  typedef GDT::ConstDiscreteFunction< SpaceType, VectorType > ConstDiscreteFct;
  Stuff::LA::Container< double >::VectorType solreal(sol.size());
  solreal.backend() = sol.backend().cwiseAbs();
  ConstDiscreteFct solasfunc(realdiscr.space(), solreal, "solution_as_discrete_function");

  //error computation
  typedef Stuff::Functions::Difference< ExpressionFct, ConstDiscreteFct > DifferenceFct;
  DifferenceFct myerror(expsol, solasfunc);
  std::cout<< "error computation"<<std::endl;
  GDT::Products::L2< LeafGridView> l2_product_operator(leafView);
  GDT::Products::HcurlSemi< LeafGridView > hcurl_product_operator(leafView);
  const double abserror = std::sqrt(l2_product_operator.apply2(myerror, myerror) + hcurl_product_operator.apply2(myerror, myerror));
  const double relerror = abserror/(std::sqrt(l2_product_operator.apply2(expsol, expsol) + hcurl_product_operator.apply2(expsol, expsol)));
  std::cout<< "absolute error: "<< abserror << std::endl;
  std::cout<< "relative error: "<< relerror <<std::endl; */


  //------------------------------------------------------------------------------------------------------------------------------------------------
  //test hmm-discretization
  //periodic grid
  Stuff::Grid::Providers::Cube< GridType > cell_grid_provider(0.0, 1.0, 8);
  auto& cell_grid = cell_grid_provider.grid();

  //test FEM periodic grid
  typedef Fem::PeriodicLeafGridPart< GridType> PeriodicGridPartType;
  typedef Fem::PeriodicLeafGridPart< GridType >::GridViewType PeriodicViewType;
  typedef PeriodicViewType::Codim< 0 >::Entity PeriodicEntityType;
  PeriodicGridPartType periodic_grid_part(cell_grid);
  auto periodic_grid_view = Fem::PeriodicLeafGridPart< GridType >(cell_grid).gridView();


  FieldVector< double, 3 > wert(0.0);
  wert[0] = 1.0;
  //elliptic cell problem
  /*typedef GDT::Operators::Cell< PeriodicLeafView, 1, GDT::Operators::ChooseCellProblem::Elliptic > EllipticCellType;
  EllipticCellType ellipticell(periodic_leafview, scalar1);
  std::cout<< "assembling elliptic cell problem"<<std::endl;
  ellipticell.assemble();
  std::cout<< "solving elliptic cell problem with lu.sparse"<<std::endl;
  auto cellrec = ellipticell.reconstruct(wert);
  GDT::ConstDiscreteFunction< EllipticCellType::SpaceType, EllipticCellType::VectorType > cellreconstrfnct(ellipticell.space(), cellrec);
  const auto& entity_ptr = periodic_leafview.template begin<0>();
  const auto& localreconstr = cellreconstrfnct.local_function(*entity_ptr);
  FieldMatrix< double, 1, 3 > ret;
  localreconstr->jacobian(entity_ptr->geometry().center(), ret); */
  //cellrec.visualize("cell_reconstrcution");

  //curlcurl cell problem
 /* typedef GDT::Operators::Cell< PeriodicLeafView, 1, GDT::Operators::ChooseCellProblem::CurlcurlDivreg > CurlCellType;
  CurlCellType curlcell(periodic_leafview, scalar1);
  std::cout<< "assembling curl cell problem"<<std::endl;
  curlcell.assemble();                                              //assembly dauert recht lange, normal??????
  std::cout<< "solving curl cell problem with lu.sparse"<<std::endl;
  auto cellrec1 = curlcell.reconstruct(wert);
  GDT::ConstDiscreteFunction< CurlCellType::SpaceType, CurlCellType::VectorType > cellreconstrfnct(curlcell.space(), cellrec1);
  FieldMatrix< double, 3, 3 > ret;
  const auto entity_it_end = periodic_leafview.template end<0>();
  int iterations = 0;
  for (auto entity_it = periodic_leafview.template begin<0>(); entity_it != entity_it_end; ++entity_it) {
    const auto& entity = *entity_it;
    const auto localreconstr = cellreconstrfnct.local_function(entity);
    localreconstr->jacobian(entity.geometry().center(), ret);
    ret *= 0.0;
    ++iterations;
  } */


  /*DSG::BoundaryInfos::AllDirichlet< LeafGridView::Intersection > bdry_info;
  HMMDiscretization< LeafGridView, PeriodicLeafView, 1 > hmmdiscr(leafView, periodic_leafview, bdry_info, scalar1, scalar2, scalar1, VectorFct(0), VectorFct(0), scalar1);
  hmmdiscr.assemble();
  std::cout<< "hmm" <<std::endl;
  std::cout<< hmmdiscr.rhs_vector().sup_norm()<<std::endl;
  std::cout<< hmmdiscr.system_matrix().rows() * hmmdiscr.system_matrix().cols() <<std::endl;
  std::cout<< hmmdiscr.system_matrix().non_zeros()<<std::endl;
  Dune::Stuff::LA::Container< std::complex< double > >::VectorType sol;
  hmmdiscr.solve(sol);
  std::cout<< sol.sup_norm() <<std::endl; */

  /*Stuff::LA::Container< double >::VectorType cellsol;
  GDT::Operators::Cell< PeriodicLeafView, 1, GDT::Operators::ChooseCellProblem::CurlcurlDivreg > curlcell(periodic_leafview, scalar1);
  curlcell.reconstruct(wert, cellsol);
  std::cout<< "curl cell problem" <<std::endl;
  std::cout<< curlcell.rhs_vector().sup_norm()<< std::endl;
  std::cout<< cellsol.sup_norm()<<std::endl;
  std::cout<< curlcell.effective_matrix() <<std::endl; */


  typedef Stuff::Functions::Expression< PeriodicEntityType, double, 3, double, 1 > ScalarExpressionFct;
  typedef Stuff::Functions::Constant< PeriodicEntityType, double, 3, double, 1> PerConstFct;
  const ScalarExpressionFct hetep("x", "1/(2+cos(2*pi*x[0]))", 2, "periodic_permittivity");
  const ScalarExpressionFct hetkappa_real("x", "(2+cos(2*pi*x[0]))/(5+4*cos(2*pi*x[0]))", 2, "periodic_kappa_real");
  const ScalarExpressionFct hetkappa_imag("x", "-1*sin(2*pi*x[0])/(5+4*cos(2*pi*x[0]))", 2, "periodic_kappa_imag");
  const PerConstFct zero(0.0);
  typedef GDT::Operators::FemEllipticCell< PeriodicGridPartType, 1 > EllCellType;
  EllCellType::ComplexVectorType cellsol;
  EllCellType ellipticcell(periodic_grid_part, hetkappa_real, hetkappa_imag);
 /* ellipticcell.reconstruct(wert, cellsol);
  EllCellType::VectorType realvec(ellipticcell.space().mapper().size());
  realvec.backend() = cellsol.backend().real();
  typedef GDT::ConstDiscreteFunction< EllCellType::SpaceType, EllCellType::VectorType > ConstDiscreteFct;
  ConstDiscreteFct cellreconstr(ellipticcell.space(), realvec); */
 // cellreconstr.visualize("elliptic_reconstruction");
  //std::cout<< ellipticcell.effective_matrix()<<std::endl;


  ScalarExpressionFct expcell("x", "1/(4*pi)*sin(2*pi*x[0])", 4, "expectedcellsolution");  //fuer hetep
 // expcell.visualize(periodic_grid_view, "cell_solution", false);


  //error computation
  /*typedef Stuff::Functions::Difference< ScalarExpressionFct, ConstDiscreteFct > DifferenceFct;
  DifferenceFct myerror(expcell, cellreconstr);
  GDT::Products::L2< PeriodicViewType > l2_prod(periodic_grid_view);
  std::cout<< std::sqrt(l2_prod.apply2(myerror, myerror)) <<std::endl; */


  /*typedef GDT::Operators::FemCurlCell< PeriodicGridPartType, 1 > CurlCellType;
  CurlCellType::VectorType cellsol1;
  CurlCellType curlcell(periodic_grid_part, hetep);
  curlcell.reconstruct(wert, cellsol1);
  std::cout<< cellsol1.sup_norm() <<std::endl;*/


  return 0;
}

