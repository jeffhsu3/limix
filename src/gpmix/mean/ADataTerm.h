/*
 * ADataTerm.h
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#ifndef ADATATERM_H_
#define ADATATERM_H_

#include <gpmix/types.h>

namespace gpmix {

//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore ADataTerm::evaluate();
%ignore ADataTerm::gradY();
%ignore ADataTerm::gradParamsRows();
%ignore ADataTerm::sumJacobianGradParams();
%ignore ADataTerm::sumLogJacobian();


//rename argout versions for python; this overwrites the C++ convenience functions
%rename(evaluate) ADataTerm::aEvaluate;
%rename(gradY) ADataTerm::aGradY;
%rename(gradParamsRows) ADataTerm::aGradParamsRows;
%rename(sumJacobianGradParams) ADataTerm::aSumJacobianGradParams;
%rename(sumLogJacobian) ADataTerm::aSumLogJacobian;
#endif


class ADataTerm {
protected:
	MatrixXd Y;
	bool insync;

public:
	ADataTerm();
	ADataTerm(MatrixXd& Y);
	virtual ~ADataTerm();
	virtual inline void setParams(MatrixXd& params){};

	virtual void aGetParams(MatrixXd* outParams){};
	virtual inline MatrixXd getParams(){ MatrixXd outParams = MatrixXd(); aGetParams(&outParams); return outParams;	};
	virtual inline void setY(const MatrixXd& Y)
	{
		checkDimensions(Y);
		this->insync = false;
		this->Y = Y;
	}

	virtual void aEvaluate(MatrixXd* Y);
	virtual void aGradY(MatrixXd* gradY);
	virtual void aGradParamsRows(MatrixXd* outGradParamsRows);
	virtual void aGradParamsCols(MatrixXd* outGradParamsCols);
	virtual void aSumJacobianGradParams(MatrixXd* sumJacobianGradParams);
	virtual void aSumLogJacobian(MatrixXd* sumJacobianGradParams);

	virtual inline MatrixXd getY(){return Y;}
	virtual inline MatrixXd evaluate() { MatrixXd ret = MatrixXd(); aEvaluate(&ret); return ret;};
	virtual inline MatrixXd gradY() { MatrixXd ret = MatrixXd(); aGradY(&ret); return ret;};
	virtual inline MatrixXd gradParamsRows(){ MatrixXd ret = MatrixXd(); aGradParamsRows(&ret); return ret;};
	virtual inline MatrixXd gradParamsCols(){ MatrixXd ret = MatrixXd(); aGradParamsCols(&ret); return ret;};
	virtual inline MatrixXd sumJacobianGradParams(){ MatrixXd ret = MatrixXd(); aSumJacobianGradParams(&ret); return ret;};
	virtual inline MatrixXd sumLogJacobian(){ MatrixXd ret = MatrixXd(); aSumLogJacobian(&ret); return ret;};
	virtual inline string getName() const {return "ADataTerm";};
	virtual inline void checkDimensions(const MatrixXd& Y){};
	bool isInSync() const;
	void makeSync();
};

} /* namespace gpmix */
#endif /* ADATATERM_H_ */