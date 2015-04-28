// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#include "config.h"

#include <iostream>
#include <vector>
#include <string>


#include <dune/grid/yaspgrid.hh>

#include <dune/stuff/functions/combined.hh>
#include <dune/stuff/functions/expression.hh>

#include <dune/gdt/localevaluation/curlcurl.hh>
#include <dune/gdt/localevaluation/elliptic.hh>

#include <dune/gdt/spaces/nedelec/pdelab.hh>


int main() {
 //   std::cout << "Hallo" <<std::endl;
    //dune::Yaspgrid erzeugen
    Dune::array< int, 3> n;
    std::fill(n.begin(), n.end(), 5);
    Dune::FieldVector< double, 3> upper(1.0);
    typedef Dune::YaspGrid< 3 > GridType;
    GridType grid(upper, n);
    typedef GridType::LeafGridView LeafGridView;
    LeafGridView leafView = grid.leafGridView();
    typedef LeafGridView::template Codim< 0 >::Iterator ElementLeafIterator;
    ElementLeafIterator it = leafView.template begin< 0 >();
    typedef GridType::Codim< 0 >::Entity EntityType;

    //Fkt addieren und multiplizieren
    typedef Dune::Stuff::Functions::Expression< EntityType, double, 3, double, 1 > ExpressionFct;
    typedef Dune::Stuff::Functions::Constant< EntityType, double, 3, double , 1 > ConstFct;
    ExpressionFct myfunction("x", "x[0]+x[1]", 1);
   // ExpressionFctType a("x", "3.0", 0);
    ConstFct a(1.0);
    auto newfct = myfunction+a;
    Dune::Stuff::Functions::Product< ConstFct, ExpressionFct > newfct2(a, myfunction);
//    typedef Dune::Stuff::Functions::Expression< EntityType, double, 3, std::complex< double >, 1> ComplexExpressionFct;
//    ComplexExpressionFct mycfct("x", "x[0]+i*x[1]", 1);
    //Fazit: geht genau so, wie es hier steht. RangeFiled muss bei beiden Fkt natuerlich gleich sein. Geht aber nicht fuer komplexe Fkt, da Fktsauswertungen
    // bzw. Ueberpruefungen gegen unedlich etc. (ohne Betrag) ausgefuehrt werden
    // es sollten die Parameterfkt fertig an curlcurldiskretisierung ubergeben werden!
    // Problem: kann noch nicht einmal komplex-wertige Expressionfkt anlegen, da deren evaluate-Methode std::isnan und std::isinf benutzt!!!


    //local evaluation curl curl testen
    const ConstFct permeab(1.0);\
    typedef Dune::Stuff::Functions::Expression< EntityType, double, 3, double, 3 > ExpressionFctVector;
    const std::vector< std::string > expressions{"0", "0", "x[0]"};
    const std::vector< std::vector< std::string > > gradients{{"0", "0", "0"}, {"0", "0", "0"}, {"1", "0", "0"}};
    const ExpressionFctVector testfct("x", expressions, 1, "stuff.globalfunction.expression", gradients);
    const std::vector< std::string > expressions1{"0", "0", "-x[0]"};
    const std::vector< std::vector< std::string > > gradients1{{"0", "0", "0"}, {"0", "0", "0"}, {"-1", "0", "0"}};
    const ExpressionFctVector testfct1("x", expressions1, 1, "stuff.globalfunction.expression", gradients1);
    const Dune::FieldVector< double, 3 > localpoint(0.5);
    Dune::DynamicMatrix< double > ret(1, 1, 0.0);

    Dune::GDT::LocalEvaluation::CurlCurl< ConstFct > localeval(permeab);
    auto permeabtuple = localeval.localFunctions(*it);
    localeval.evaluate(permeabtuple, *(testfct.local_function(*it)), *(testfct1.local_function(*it)), localpoint, ret);
    std::cout << ret <<std::endl;
    //funktioniert!!! :-)


    return 0;
}

