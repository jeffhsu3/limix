/*
 * ALmm.cpp
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#include "lmm.h"
#include "gpmix/utils/gamma.h"
#include "gpmix/utils/fisherf.h"


namespace gpmix {

//const double _PI = (double)(((2.0))) * std::acos((double)(((0.0))));
const double L2pi = 1.8378770664093453;

/*ALMM*/
ALMM::ALMM()
{
	//Default settings:
	num_intervals0 = 100;
	num_intervalsAlt = 0;
	ldeltamin0 = -5;
	ldeltamax0 = 5;
	UK_cached = false;
	Usnps_cached = false;
	Upheno_cached = false;
	Ucovs_cached = false;
}

ALMM::~ALMM()
{
}

mfloat_t ALMM::getLdeltamin0() const
{
	return ldeltamin0;
}

muint_t ALMM::getNumIntervalsAlt() const
{
	return num_intervalsAlt;
}

muint_t ALMM::getNumSamples() const
{
	return num_samples;
}

void ALMM::setLdeltamin0(mfloat_t ldeltamin0)
{
	this->ldeltamin0 = ldeltamin0;
}

void ALMM::setNumIntervalsAlt(muint_t num_intervalsAlt)
{
	this->num_intervalsAlt = num_intervalsAlt;
}

MatrixXd ALMM::getPheno() const
{
	return pheno;
}

MatrixXd ALMM::getPv() const
{
	return pv;
}

void ALMM::getPheno(MatrixXd *out) const
{
	(*out) = pheno;
}

void ALMM::getPv(MatrixXd *out) const
{
	(*out) = pv;
}

void ALMM::getSnps(MatrixXd *out) const
{
	(*out) = snps;
}

void ALMM::getCovs(MatrixXd *out) const
{
	(*out) = covs;
}

MatrixXd ALMM::getCovs() const
{
	return covs;
}

void ALMM::setCovs(const MatrixXd & covs)
{
	this->covs = covs;
}

void ALMM::setPheno(const MatrixXd & pheno)
{
	this->pheno = pheno;
}

void ALMM::setSNPs(const MatrixXd & snps)
{
	this->snps = snps;
}

mfloat_t ALMM::getLdeltaminAlt() const
{
	return ldeltaminAlt;
}

void ALMM::setLdeltaminAlt(mfloat_t ldeltaminAlt)
{
	this->ldeltaminAlt = ldeltaminAlt;
}

MatrixXd ALMM::getSnps() const
{
	return snps;
}

/*CLMM*/
CLMM::CLMM()
:ALMM()
{
}

CLMM::~CLMM()
{
	// TODO Auto-generated destructor stub
}

void CLMM::getK(MatrixXd *out) const
{
	(*out) = K;
}

MatrixXd CLMM::getK() const
{
	return K;
}

void CLMM::setK(const MatrixXd & K)
{
	this->K = K;
	this->UK_cached = false;
}

void CLMM::setK(const MatrixXd & K, const MatrixXd & U, const VectorXd & S)
{
	this->K = K;
	this->U = U;
	this->S = S;
	this->UK_cached = true;
}

/*CLMM*/
void CLMM::updateDecomposition()
{
	//check that dimensions match
	this->num_samples = snps.rows();
	this->num_snps = snps.cols();
	this->num_pheno = pheno.cols();
	this->num_covs = covs.cols();
	if(!num_samples == pheno.rows())
		throw new CGPMixException("phenotypes and SNP dimensions inconsistent");

	if(!num_samples == covs.rows())
		throw CGPMixException("covariates and SNP dimensions inconsistent");

	if(!(this->UK_cached)){
		//decomposition of K
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(K);
		U = eigensolver.eigenvectors();
		S = eigensolver.eigenvalues();
	}
	if(!Usnps_cached){
		Usnps = U.transpose() * snps;
		Usnps_cached = true;
	}
	if(!Upheno_cached){
		Upheno = U.transpose() * pheno;
		Upheno_cached = true;
	}
	if(!Ucovs_cached){
		Ucovs = U.transpose() * covs;
		Ucovs_cached = true;
	}
}

void CLMM::process()
{
	//get decomposition
	updateDecomposition();
	//resize phenotype output
	this->pv.resize(this->num_pheno, this->num_snps);
	MatrixXd pvF = MatrixXd(this->num_pheno, this->num_snps);

	//result matries: think about what to store in the end
	MatrixXd ldelta0(num_pheno, 1);
	MatrixXd ldelta(num_pheno, num_snps);
	MatrixXd nLL0(num_pheno, 1);
	MatrixXd nLL(num_pheno, num_snps);
	//reserve memory for snp-wise foreground model
	MatrixXd UX_(num_samples, num_covs + 1);
	//store covariates upfront
	UX_.block(0, 0, num_samples, num_covs) = Ucovs;
	MatrixXd f_tests;
	for(muint_t ip = 0;ip < num_pheno;ip++){
		//get UY columns
		MatrixXd UY_ = Upheno.block(0, ip, num_samples, 1);
		//fit delta on null model
		ldelta0(ip) = optdelta(UY_, Ucovs, S, num_intervals0, ldeltamin0, ldeltamax0);
		nLL0(ip) = nLLeval(&f_tests, ldelta0(ip), UY_, Ucovs, S);
		for(muint_t is = 0;is < num_snps;is++){
			//1. construct foreground testing SNP transformed
			UX_.block(0, num_covs, num_samples, 1) = Usnps.block(0, is, num_samples, 1);
			//2. fit delta
			if(num_intervalsAlt > 0)
				//fit delta on alt model also
				ldelta(ip, is) = optdelta(UY_, UX_, S, num_intervalsAlt, ldelta0(ip) + ldeltaminAlt, ldelta0(ip) + ldeltamaxAlt);

			else
				ldelta(ip, is) = ldelta0(ip);

			//3. evaluate
			MatrixXd f_tests;
			nLL(ip, is) = nLLeval(&f_tests, ldelta(ip, is), UY_, UX_, S);
			//4. calc p-value
			this->pv(ip, is) = Gamma::gammaQ(nLL0(ip, 0) - nLL(ip, is), (double)(((0.5))) * 1.0);
			//compare p-value of LRT with F-test:
		} //end for SNP
	} //end for phenotypes

}

CKroneckerLMM::CKroneckerLMM()
{
}

CKroneckerLMM::~CKroneckerLMM()
{
}

void CKroneckerLMM::updateDecomposition()
{
	//TODO: think about caching procedures:
}

mfloat_t CKroneckerLMM::optdelta(const MatrixXd & UX, const MatrixXd & UYU, const VectorXd & S_C, const VectorXd & S_R, const muint_t numintervals, const mfloat_t ldeltamin, const mfloat_t ldeltamax, const MatrixXd & WkronDiag, const MatrixXd & WkronBlock)
{
	//grid variable with the current likelihood evaluations
	MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
	MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
	//current delta
	mfloat_t ldelta = ldeltamin;
	mfloat_t ldeltaD = (ldeltamax - ldeltamin);
	ldeltaD /= ((mfloat_t)((((numintervals)))) - 1);
	mfloat_t nllmin = HUGE_VAL;
	mfloat_t ldeltaopt_glob = 0;
	MatrixXd f_tests;
	for(muint_t i = 0;i < numintervals;i++){
		nllgrid(i, 0) = CKroneckerLMM::nLLeval(&f_tests, ldeltagrid(i), WkronDiag, WkronBlock, UX, UYU, S_C, S_R);
		ldeltagrid(i, 0) = ldelta;
		//std::cout<< "nnl( " << ldelta << ") = " << nllgrid(i,0) <<  "VS" << nllmin << ")\n\n";
		if(nllgrid(i, 0) < nllmin){
			//		std::cout << "new min (" << nllmin << ") -> " <<  nllgrid(i,0) << "\n\n";
			nllmin = nllgrid(i, 0);
			ldeltaopt_glob = ldelta;
		}
		//move on delta
		ldelta += ldeltaD;
	} //end for all intervals

	//std::cout << "\n\n nLL_i:\n" << nllgrid;
	return ldeltaopt_glob;
}

mfloat_t CKroneckerLMM::nLLeval(MatrixXd *F_tests, mfloat_t ldelta, const MatrixXd & WkronDiag, const MatrixXd & WkronBlock, const MatrixXd & UX, const MatrixXd & UYU, const VectorXd & S_C, const VectorXd & S_R)
{
	muint_t n = UX.rows();
	muint_t d = UX.cols();
	muint_t p = UYU.cols();
	assert(UYU.cols() == S_C.rows());
	assert(UYU.rows() == S_R.rows());
	assert(UYU.rows() == UX.rows());
	assert((muint_t)WkronDiag.cols() == d);
	assert((muint_t)WkronBlock.cols() == d);
	assert((muint_t)WkronDiag.rows() * (muint_t)WkronBlock.rows() == p);
	mfloat_t delta = exp(ldelta);
	mfloat_t ldet = 0.0;
	(*F_tests).resize(d, WkronDiag.rows());
	MatrixXd beta = MatrixXd(d, (muint_t)(WkronDiag.rows()));
	MatrixXd Sd = MatrixXd(S_R.rows(), S_C.rows());
	for(muint_t col = 0;col < (muint_t)(S_C.rows());++col){
		for(muint_t row = 0;row < (muint_t)(S_R.rows());++row){
			Sd(row, col) = S_R(row) * S_C(col) + delta;
			ldet += std::log((mfloat_t)(Sd(row, col)));
		}
	}

	muint_t phen = 0;
	MatrixXd XSdi = MatrixXd(UX.rows(), UX.cols());
	mfloat_t res = (UYU.array() * UYU.array() / Sd.array()).sum();
	for(muint_t i_diag = 0;i_diag < (muint_t)(WkronDiag.rows());++i_diag){
		MatrixXd XSX = MatrixXd::Zero(d, d);
		MatrixXd XSY = MatrixXd::Zero(d, 1);
		for(muint_t i_block = 0;i_block < (muint_t)(WkronBlock.rows());++i_block){
			VectorXd Sd_p = Sd.block(0, phen, n, 1);
			for(muint_t dim = 0;dim < d;++dim){
				XSdi.block(0, dim, n, 1).array() = (UX.block(0, dim, n, 1).array() / Sd.block(0, phen, n, 1).array()) * (WkronDiag(i_diag, dim) * WkronBlock(i_block, dim));
			}
			XSX += XSdi.transpose() * UX;
			XSY += XSdi.transpose() * UYU.block(0, phen, n, 1);
			++phen;
		}

		//least sqaures solution of XSX*beta = XSY
		//decomposition of K
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XSX);
		MatrixXd U_X = eigensolver.eigenvectors();
		MatrixXd S_X = eigensolver.eigenvalues();
		beta.block(0, i_diag, d, 1) = U_X.transpose() * XSY;
		//MatrixXd S_i = MatrixXd::Zero(d,d);
		for(size_t dim = 0;dim < d;++dim){
			if(S_X(dim, 0) > 3E-8){
				beta(dim, i_diag) /= S_X(dim, 0);
				for(size_t dim2 = 0;dim2 < d;++dim2){
					(*F_tests)(dim2, i_diag) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
				}
				//S_i(dim,dim) = 1.0/S_X(dim,0);
			}
			else{
				beta(dim, i_diag) = 0.0;
			}
		}

		beta.block(0, i_diag, d, 1) = U_X * beta.block(0, i_diag, d, 1);
		res -= (XSY.array() * beta.block(0, i_diag, d, 1).array()).sum();
	}

	//sqared residuals
	mfloat_t sigg2 = res / (n * p);
	//compute the F-statistics
	(*F_tests).array() = beta.array() * beta.array() / (*F_tests).array();
	(*F_tests).array() /= sigg2;
	double nLL = 0.5 * (n * p * L2pi + ldet + n * p + n * p * log(sigg2));
	return nLL;
}

void CKroneckerLMM::process()
{
	this->Usnps = this->U_R.transpose() * this->snps;
	this->Upheno = this->U_R.transpose() * this->pheno * this->U_C;
	this->Ucovs = this->U_R.transpose() * this->covs;
	//resize phenotype output
	this->pv.resize(1, this->snps.cols());
	//this->pv.resize(pheno.cols(), this->snps.cols());
	//result matrices: think about what to store in the end
	MatrixXd ldelta0(1, 1);
	MatrixXd ldelta(1, snps.cols());
	MatrixXd nLL0(1, 1);
	MatrixXd nLL(1, snps.cols());
	//reserve memory for snp-wise foreground model
	MatrixXd UX(snps.rows(), covs.cols() + 1);
	//store covariates upfront
	UX.block(0, 0, snps.rows(), covs.cols()) = Ucovs;
	MatrixXd f_tests;
	//fit delta on null model
	ldelta0(0) = CKroneckerLMM::optdelta(Ucovs, Upheno, this->S_C, this->S_R, this->num_intervals0, this->ldeltamin0, this->ldeltamax0, WkronDiag0, WkronBlock0);
	nLL0(0) = this->nLLeval(&f_tests, ldelta0(0), WkronDiag0, WkronBlock0, Ucovs, Upheno, this->S_C, this->S_R);
	for(muint_t is = 0;is < (muint_t)(snps.cols());is++){
		//1. construct foreground testing SNP transformed
		UX.block(0, covs.cols(), snps.rows(), 1) = Usnps.block(0, is, snps.rows(), 1);
		//2. fit delta
		if(this->num_intervalsAlt > 0)
			//fit delta on alt model also
			ldelta(0, is) = CKroneckerLMM::optdelta(UX, Upheno, this->S_C, this->S_R, this->num_intervalsAlt, this->ldeltaminAlt, this->ldeltamaxAlt, WkronDiag, WkronBlock);

		else
			ldelta(0, is) = ldelta0(0);

		//3. evaluate
		MatrixXd f_tests;
		nLL(0, is) = CKroneckerLMM::nLLeval(&f_tests, ldelta(0, is), WkronDiag, WkronBlock, UX, Upheno, this->S_C, this->S_R);
		//				gpmix::nLLeval(&f_tests, ldelta(0, is), Upheno, UX, S);
		//TODO:
		//rotate F_tests to get per pheno P-values
		//4. calc lod score
		//TODO: calculate dofs for arbitrary WkronDiag and WkronBlock, currently we expect all ones...
		mfloat_t dof = ((mfloat_t)(Upheno.cols()) / (mfloat_t)(WkronBlock.rows()));
		this->pv(0, is) = Gamma::gammaQ(nLL0(0, 0) - nLL(0, is), (double)(((0.5) * dof)));
	} //end for phenotypes
}

void CKroneckerLMM::setKronStructure(const MatrixXd & WkronDiag0, const MatrixXd & WkronBlock0, const MatrixXd & WkronDiag, const MatrixXd & WkronBlock)
{
	this->WkronDiag0 = WkronDiag0;
	this->WkronBlock0 = WkronBlock0;
	this->WkronDiag = WkronDiag;
	this->WkronBlock = WkronBlock;
}

void CKroneckerLMM::getK_C(MatrixXd *out) const
{
	(*out) = C;
}

void CKroneckerLMM::setK_C(const MatrixXd & C)
{
	this->C = C;
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(C);
	this->U_C = eigensolver.eigenvectors();
	this->S_C = eigensolver.eigenvalues();
}

void CKroneckerLMM::setK_C(const MatrixXd & C, const MatrixXd & U_C, const VectorXd & S_C)
{
	this->C = C;
	this->U_C = U_C;
	this->S_C = S_C;
}

void CKroneckerLMM::getK_R(MatrixXd *out) const
{
	(*out) = R;
}

void CKroneckerLMM::setK_R(const MatrixXd & R)
{
	this->R = R;
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(R);
	this->U_R = eigensolver.eigenvectors();
	this->S_R = eigensolver.eigenvalues();
}

void CKroneckerLMM::setK_R(const MatrixXd & R, const MatrixXd & U_R, const VectorXd & S_R)
{
	this->R = R;
	this->U_R = U_R;
	this->S_R = S_R;
}

MatrixXd CKroneckerLMM::getK_R() const
{
	return R;
}

MatrixXd CKroneckerLMM::getK_C() const
{
	return C;
}

double optdelta(const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, int numintervals, double ldeltamin, double ldeltamax)
{
	//grid variable with the current likelihood evaluations
	MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
	MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
	//current delta
	double ldelta = ldeltamin;
	double ldeltaD = (ldeltamax - ldeltamin);
	ldeltaD /= ((double)((((numintervals)))) - 1);
	double nllmin = HUGE_VAL;
	double ldeltaopt_glob = 0;
	MatrixXd f_tests;
	for(int i = 0;i < (numintervals);i++){
		nllgrid(i, 0) = nLLeval(&f_tests, ldelta, UY, UX, S);
		ldeltagrid(i, 0) = ldelta;
		//std::cout<< "nnl( " << ldelta << ") = " << nllgrid(i,0) <<  "VS" << nllmin << ")\n\n";
		if(nllgrid(i, 0) < nllmin){
			//		std::cout << "new min (" << nllmin << ") -> " <<  nllgrid(i,0) << "\n\n";
			nllmin = nllgrid(i, 0);
			ldeltaopt_glob = ldelta;
		}
		//move on delta
		ldelta += ldeltaD;
	} //end for all intervals

	//std::cout << "\n\n nLL_i:\n" << nllgrid;
	return ldeltaopt_glob;
}

/* internal functions */
double nLLeval(MatrixXd *F_tests, double ldelta, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S)
{
	size_t n = UX.rows();
	size_t d = UX.cols();
	size_t n_pheno = UY.cols();
	assert(UY.cols() == S.cols());
	assert(UY.rows() == S.rows());
	assert(UY.rows() == UX.rows());
	double delta = exp(ldelta);
	MatrixXd Sdi = S.array() + delta;
	double ldet = 0.0;
	for(size_t ind = 0;ind < n_pheno * n;++ind){
		ldet += log(Sdi.data()[ind]);
	}
	Sdi = Sdi.array().inverse();
	(*F_tests).resize(d, n_pheno);
	MatrixXd beta = MatrixXd(d, n_pheno);
	//replice Sdi
	for(muint_t phen = 0;phen < n_pheno;++phen){
		VectorXd Sdi_p = Sdi.block(0, phen, n, 1);
		MatrixXd XSdi = (UX.array() * Sdi_p.replicate(1, d).array()).transpose();
		MatrixXd XSX = XSdi * UX;
		MatrixXd XSY = XSdi * UY.block(0, phen, n, 1);
		//least sqaures solution of XSX*beta = XSY
		//decomposition of K
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XSX);
		MatrixXd U_X = eigensolver.eigenvectors();
		MatrixXd S_X = eigensolver.eigenvalues();
		beta.block(0, phen, d, 1) = U_X.transpose() * XSY;
		//MatrixXd S_i = MatrixXd::Zero(d,d);
		for(size_t dim = 0;dim < d;++dim){
			if(S_X(dim, 0) > 3E-8){
				beta(dim, phen) /= S_X(dim, 0);
				for(size_t dim2 = 0;dim2 < d;++dim2){
					(*F_tests)(dim2, phen) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
				}
				//S_i(dim,dim) = 1.0/S_X(dim,0);
			}
			else{
				beta(dim, phen) = 0.0;
			}
		}

		beta.block(0, phen, d, 1) = U_X * beta.block(0, phen, d, 1);
	}

	MatrixXd res = UY - UX * beta;
	//sqared residuals
	res.array() *= res.array();
	res.array() *= Sdi.array();
	double sigg2 = res.array().sum() / (n * n_pheno);
	//compute the F-statistics
	(*F_tests).array() = beta.array() * beta.array() / (*F_tests).array();
	(*F_tests).array() /= sigg2;
	double nLL = 0.5 * (n * n_pheno * L2pi + ldet + n * n_pheno + n * n_pheno * log(sigg2));
	return nLL;
}

void optdeltaAllY(MatrixXd *out, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, const MatrixXd & ldeltagrid)
{
	size_t n_p = UY.cols();
	size_t numintervals = ldeltagrid.rows();
	//grid variable with the current likelihood evaluations
	(*out) = MatrixXd::Ones(numintervals,n_p).array()*HUGE_VAL;
	//current delta
	for(size_t i = 0;i < numintervals;i++){
		MatrixXd row;
		nLLevalAllY(&row, ldeltagrid(i, 0), UY, UX, S);
		(*out).row(i) = row;
		//ldeltagrid(0,i) = ldelta;
	} //end for all intervals
}

void train_associations_SingleSNP(MatrixXd *PV, MatrixXd *LL, MatrixXd *ldelta, const MatrixXd & X, const MatrixXd & Y, const MatrixXd & U, const MatrixXd & S, const MatrixXd & C, int numintervals, double ldeltamin, double ldeltamax)
{
	//get dimensions:
	//samples
	size_t nn = X.rows();
	//snps
	size_t ns = X.cols();
	assert( ns == 1 );
	//phenotypes
	size_t np = Y.cols();
	//covaraites
	size_t nc = C.cols();
	//make sure the size of N/Y is correct
	assert((int)nn==(int)Y.rows());
	assert((int)nn==(int)C.rows());
	assert((int)nn==(int)U.rows());
	assert((int)nn==(int)U.cols());
	//resize output variable if needed
	(*PV).resize(np, ns);
	(*LL).resize(np, ns);
	(*ldelta).resize(np, ns);
	//transform everything
	MatrixXd UX = U.transpose() * X;
	MatrixXd UY = U.transpose() * Y;
	MatrixXd Ucovariates = U.transpose() * C;
	//reserve memory for snp-wise foreground model
	MatrixXd UX_(nn, nc + 1);
	//store covariates upfront
	UX_.block(0, 0, nn, nc) = Ucovariates;
	UX_.block(0, nc, nn, 1) = UX;
	MatrixXd ldeltagrid(numintervals, 1);
	for(size_t interval = 0;interval < (size_t)((numintervals));++interval){
		ldeltagrid(interval, 0) = ldeltamin + interval * ((ldeltamax - ldeltamin) / (1.0 * (numintervals - 1)));
	}
	MatrixXd nllgrid;
	optdeltaAllY(&nllgrid, UY, UX_, S, ldeltagrid);
	//1. fit background covariances on phenotype and covariates alone
	for(size_t ip = 0;ip < np;ip++){
		(*ldelta)(ip) = ldeltamin;
		size_t i_min = 0;
		for(size_t interval = 1;interval < (size_t)((numintervals));++interval){
			if(nllgrid(interval, ip) < nllgrid(i_min, ip)){
				//printf("oldmin : %.4f, newmin : %.4f, newdelta : %.4f, interval : %i\n" ,nllgrid(i_min,ip) , nllgrid(interval,ip), ldeltagrid(interval,0),interval);
				(*ldelta)(ip, 0) = ldeltagrid(interval, 0);
				i_min = interval;
			}
		}

		//get UY columns
		MatrixXd UY_ = UY.block(0, ip, nn, 1);
		//fit delta on null model
		MatrixXd f_tests;
		(*LL)(ip, 0) = -1.0 * nLLeval(&f_tests, (*ldelta)(ip), UY_, UX_, S);
		(*PV)(ip, 0) = 1.0 - FisherF::Cdf(f_tests(nc, 0), 1.0, (double)(((nn - f_tests.rows()))));
	}

}

/* Internal C++ functions */
void nLLevalAllY(MatrixXd *out, double ldelta, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S)
{
	size_t n = UX.rows();
	size_t d = UX.cols();
	size_t p = UY.cols();
	/*
std::cout << UX<< "\n\n";
std::cout << UY<< "\n\n";
std::cout << S<< "\n\n";
	 */
	double delta = exp(ldelta);
	MatrixXd Sdi = S.array() + delta;
	double ldet = Sdi.array().log().sum();
	//std::cout << "ldet" << ldet << "\n\n";
	//elementwise inverse
	Sdi = Sdi.array().inverse();
	//std::cout << "Sdi" << Sdi << "\n\n";
	//replice Sdi
	MatrixXd XSdi = (UX.array() * Sdi.replicate(1, d).array()).transpose();
	MatrixXd XSX = XSdi * UX;
	MatrixXd XSY = XSdi * UY;
	//std::cout << "XSdi" << XSdi << "\n\n";
	//std::cout << "XSX" << XSX << "\n\n";
	//least squares solution of XSX*beta = XSY
	MatrixXd beta = XSX.colPivHouseholderQr().solve(XSY);
	MatrixXd res = UY - UX * beta;
	//squared residuals
	res.array() *= res.array();
	res.array() *= Sdi.replicate(1, p).array();
	//MatrixXd sigg2 = MatrixXd(1,p);
	(*out) = MatrixXd(1, p);
	for(size_t phen = 0;phen < p;++phen){
		double sigg2 = res.col(phen).array().sum() / n;
		(*out)(0, phen) = 0.5 * (n * L2pi + ldet + n + n * log(sigg2));
	}
}

CSimpleKroneckerLMM::CSimpleKroneckerLMM()
{
}

CSimpleKroneckerLMM::~CSimpleKroneckerLMM()
{
}

void CSimpleKroneckerLMM::kron_snps(MatrixXd* out,const MatrixXd& x,const MatrixXd& kron)
{
	//kronecker the SNPs with the internal Wkron matrix
	//1. determine size of output object and resize if needed
	muint_t nc = (muint_t)kron.cols();
	muint_t np = (muint_t)kron.rows();
	muint_t nn = (muint_t)x.rows();
	//int ns = x.cols();
	(*out).resize(nn*np,nc);
	//loop through and set things
	//for columns to compile:
	MatrixXd _xrot = MatrixXd::Zero(num_samples,num_pheno);
	for (muint_t ic=0;ic<nc;ic++)
	{
		MatrixXd _x = MatrixXd::Zero(num_samples,num_pheno);
		//for phenotypes:
		for(muint_t ip=0;ip<np;ip++)
		{
			if (kron(ip,ic)>0)
				_x.block(0,ip,num_samples,1) = kron(ip,ic)*x.block(0,ic,num_samples,1);
			//create effective genotype weight matrix for rotation
		}
		kron_rot(&_xrot,_x);
		_xrot.resize(nn*np,1);
		(*out).block(0,ic,nn*np,1) = _xrot;
	}
}

void CSimpleKroneckerLMM::kron_rot(MatrixXd* out,const MatrixXd& x)
{
	(*out).resize(x.rows(),x.cols());
	//use vec trick to rate with U_C and U_r
	(*out) = U_R*x*U_C.transpose();
}

void CSimpleKroneckerLMM::process()
{
	//0. update decomposition
	updateDecomposition();

	//1. check that everything is sober
	assert (pheno.cols()==Wkron.rows());
	assert (pheno.cols()==Wkron0.rows());
	assert (snps.rows()==pheno.rows());
	assert ((muint_t)this->S.rows()==(num_samples*num_pheno));

	//result matries: think about what to store in the end
	//number of weights depend on kroneckering
	muint_t nw  = Wkron.cols();
	muint_t nw0 = Wkron0.cols();

	MatrixXd ldelta0(1, 1);
	MatrixXd ldelta(1, num_snps);

	MatrixXd nLL0(1, 1);
	MatrixXd nLL(num_snps,1);

	this->pv.resize(this->num_snps,1);



	//1. rotate Y
	MatrixXd UY_;
	kron_rot(&UY_,this->pheno);
	UY_.resize(num_samples*num_pheno,1);

	//2. get rotatated covariates
	MatrixXd Uc;
	kron_snps(&Uc,covs,Wkron0);

	assert(((muint_t)Uc.rows())==num_pheno*num_samples);
	assert(((muint_t)Uc.cols())==nw0);

	//reserve memory for snp-wise foreground model
	MatrixXd UX_(num_samples*num_pheno, nw + nw0);
	//store covariates upfront
	UX_.block(0, 0, num_samples*num_pheno, nw0) = Ucovs;
	MatrixXd f_tests;

	//calc null model
	ldelta0(0) = optdelta(UY_, Ucovs, S, num_intervals0, ldeltamin0, ldeltamax0);
	nLL0(0) = nLLeval(&f_tests, ldelta0(1), UY_, Ucovs, S);
	MatrixXd Ux;

	assert ((muint_t)Ux.rows()==num_samples*num_pheno);
	assert ((muint_t)Ux.cols()==nw);

	//calc degrees of freedom
	muint_t n_fr = (nw);

	for (muint_t is=0;is<num_snps;is++)
	{
		//1. create rotated x for this SNP
		this->kron_snps(&Ux,this->snps.block(0,is,num_samples,1),this->Wkron);
		//2. fill in
		UX_.block(0, nw0, num_samples*num_pheno, nw) = Ux;
		if(num_intervalsAlt > 0)
			//fit delta on alt model also
			ldelta(is) = optdelta(UY_, UX_, S, num_intervalsAlt, ldelta0(0) + ldeltaminAlt, ldelta0(0) + ldeltamaxAlt);

		else
			ldelta(is) = ldelta0(0);
		//
		MatrixXd f_tests;
		nLL(is) = nLLeval(&f_tests, ldelta(is), UY_, UX_, S);
		//4. calc p-value
		this->pv(is) = Gamma::gammaQ(nLL0(0) - nLL(is), (double)(((0.5))) * n_fr);
	}

}

void CSimpleKroneckerLMM::updateDecomposition()
{
	//check that dimensions match
	this->num_samples = snps.rows();
	this->num_snps = snps.cols();
	this->num_pheno = pheno.cols();
	this->num_covs = covs.cols();
	if(!num_samples == pheno.rows())
		throw new CGPMixException("phenotypes and SNP dimensions inconsistent");

	if(!num_samples == covs.rows())
		throw CGPMixException("covariates and SNP dimensions inconsistent");

	//we just update the kronecker of S
	S.resize(num_samples*num_pheno);
	for (muint_t ip=0;ip<num_pheno;ip++)
	{
		S.block(ip*num_samples,0,num_samples,1) = this->S_C(ip)*S_R;
	}

}

void CSimpleKroneckerLMM::setK_C(const MatrixXd & C)
{
	this->C = C;
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(C);
	this->U_C = eigensolver.eigenvectors();
	this->S_C = eigensolver.eigenvalues();
}

void CSimpleKroneckerLMM::setK_C(const MatrixXd & C, const MatrixXd & U_C, const VectorXd & S_C)
{
	this->C = C;
	this->U_C = U_C;
	this->S_C = S_C;
}

void CSimpleKroneckerLMM::setK_R(const MatrixXd & R)
{
	this->R = R;
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(R);
	this->U_R = eigensolver.eigenvectors();
	this->S_R = eigensolver.eigenvalues();
}

void CSimpleKroneckerLMM::setK_R(const MatrixXd & R, const MatrixXd & U_R, const VectorXd & S_R)
{
	this->R = R;
	this->U_R = U_R;
	this->S_R = S_R;
}

void CSimpleKroneckerLMM::getK_R(MatrixXd *out) const
{
	(*out) = R;
}

void CSimpleKroneckerLMM::getK_C(MatrixXd *out) const
{
	(*out) = C;
}

MatrixXd CSimpleKroneckerLMM::getK_R() const
{
	return R;
}

MatrixXd CSimpleKroneckerLMM::getWkron() const
{
	return Wkron;
}

MatrixXd CSimpleKroneckerLMM::getWkron0() const
{
	return Wkron0;
}

void CSimpleKroneckerLMM::setWkron(const MatrixXd& Wkron)
{
	this->Wkron = Wkron;
}

void CSimpleKroneckerLMM::setWkron0(const MatrixXd& Wkron0)
{
	this->Wkron0 = Wkron0;
}



MatrixXd CSimpleKroneckerLMM::getK_C() const
{
	return C;
}




/* namespace gpmix */
}

