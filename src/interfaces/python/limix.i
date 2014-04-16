%module core
%feature("autodoc", "3");
%include exception.i       

%{
#define SWIG_FILE_WITH_INIT
#define SWIG
#include "limix/types.h"
#include "limix/covar/covariance.h"
#include "limix/utils/cache.h"
#include "limix/covar/linear.h"
#include "limix/covar/freeform.h"
#include "limix/covar/se.h"
#include "limix/covar/combinators.h"	
#include "limix/likelihood/likelihood.h"
#include "limix/mean/ADataTerm.h"
#include "limix/mean/CData.h"
#include "limix/mean/CLinearMean.h"
#include "limix/mean/CSumLinear.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_kronecker.h"
#include "limix/gp/gp_kronSum.h"
#include "limix/gp/gp_Sum.h"
#include "limix/gp/gp_opt.h"
#include "limix/LMM/lmm.h"
#include "limix/LMM/kronecker_lmm.h"
#include "limix/modules/CVarianceDecomposition.h"
#include "limix/io/dataframe.h"
#include "limix/io/genotype.h"



using namespace limix;
//  removed namespace bindings (12.02.12)
%}

/* Get the numpy typemaps */
%include "numpy.i"
//support for eigen matrix stuff
%include "eigen.i"
//support for std libs

#define SWIG_SHARED_PTR_NAMESPACE std
//C11, no tr!
//#define SWIG_SHARED_PTR_SUBNAMESPACE tr1
%include "std_shared_ptr.i"

//removed boost
//%include <boost_shared_ptr.i>

%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"
%include "stdint.i"


%init %{
  import_array();
%}


%exception{
	try {
	$action
	} catch (limix::CGPMixException& e) {
	std::cout << "caught: " << e.what() << "\n";
	SWIG_exception(SWIG_ValueError, "LIMIX exception");
	return NULL;
	} catch (...) {
	std::cout << "caught: unknown "<< "\n";
	SWIG_exception(SWIG_RuntimeError,"Unknown exception");
	}
}


// Includ dedicated interface files
%include "./../types.i"
%include "./../covar.i"
%include "./../gp.i"
%include "./../lik.i"
%include "./../mean.i"
%include "./../lmm.i"
%include "./../modules.i"
%include "./../io.i"


//generated outodoc:
%include "limix/types.h"
%include "limix/covar/covariance.h"
%include "limix/utils/cache.h"
%include "limix/covar/linear.h"
%include "limix/covar/freeform.h"
%include "limix/covar/se.h"
%include "limix/covar/combinators.h"	
%include "limix/likelihood/likelihood.h"
%include "limix/mean/ADataTerm.h"
%include "limix/mean/CData.h"
%include "limix/mean/CLinearMean.h"
%include "limix/mean/CSumLinear.h"
%include "limix/mean/CKroneckerMean.h"
%include "limix/gp/gp_base.h"
%include "limix/gp/gp_kronecker.h"
%include "limix/gp/gp_kronSum.h"
%include "limix/gp/gp_Sum.h"
%include "limix/gp/gp_opt.h"
%include "limix/LMM/lmm.h"
%include "limix/LMM/kronecker_lmm.h"
%include "limix/modules/CVarianceDecomposition.h"
%include "limix/io/dataframe.h"
%include "limix/io/genotype.h"

 
