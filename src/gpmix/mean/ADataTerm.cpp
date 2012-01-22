/*
 * ADataTerm.cpp
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#include "ADataTerm.h"

namespace gpmix {
ADataTerm::ADataTerm()
{
	this->Y = MatrixXd();
}

ADataTerm::ADataTerm(MatrixXd& Y) {
	this->Y = Y;
}

ADataTerm::~ADataTerm()
{
}

bool ADataTerm::isInSync() const
{return insync;}

void ADataTerm::makeSync()
{ insync = true;}

void ADataTerm::aEvaluate(MatrixXd* outY)
{
	*outY = this->Y;
}

void ADataTerm::aGradY(MatrixXd* outGradY)
{
	*outGradY = MatrixXd::Ones(this->Y.rows(), this->Y.cols());
}

void ADataTerm::aGradParamsRows(MatrixXd* outGradParamsRows)
{
	*outGradParamsRows = MatrixXd();
}

void ADataTerm::aGradParamsCols(MatrixXd* outGradParamsCols)
{
	*outGradParamsCols = MatrixXd();
}

void ADataTerm::aSumJacobianGradParams(MatrixXd* outSumJacobianGradParams)
{
	*outSumJacobianGradParams = MatrixXd();
}

void ADataTerm::aSumLogJacobian(MatrixXd* outSumJacobianGradParams)
{
	*outSumJacobianGradParams = MatrixXd();
}



} /* namespace gpmix */