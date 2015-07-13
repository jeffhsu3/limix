import numpy as np
import numpy.linalg as la
import pandas as pd
from util import *
import time, os
import ipdb
from pysnptools.snpreader import Bed

import logging
import pysnptools.standardizer
import pysnptools
import pysnptools.util
import pysnptools.util.pheno
import time
import kron_gwas
from gwas import GWAS, create_dir


class MultivariateGWAS(GWAS):
	def __init__(self, snps_test, phenotype, K=None, snps_K=None, covariates=None, 
				 h2=None, interact_with_snp=None, nGridH2=10, standardizer=None, add_bias=True, normalize_K=True, blocksize=10000, A_snps=None, R2=None, C1=None, C2=None):
		if standardizer is None:
			standardizer = pysnptools.standardizer.Unit()
		self.standardizer = standardizer
		self.phenotype = GWAS.check_pheno_format(phenotype=phenotype)
		self.covariates = GWAS.check_pheno_format(phenotype=covariates)
		self.snps_test = snps_test
		self.snps_K = snps_K
		self.K = K
		if (self.K is None) and (self.snps_K is None):
			self.K = np.eye(self.sample_count)
		elif self.K is None:
			self.K = snps_K.kernel(standardizer=standardizer, blocksize=blocksize)
			if normalize_K:
				self.K /= self.K.diagonal().mean()
		elif snps_K is None:
			if normalize_K:
				self.K = self.K / self.K.diagonal().mean()
		else:
			raise NotImplementedError("either K or snps_K has to be None")
		if add_bias and (self.covariates is None):
			bias = pd.Series(np.ones((self.sample_count)), index=self.covariates.index)
			self.covariates = pd.DataFrame(data=bias, columns=["bias"])
		elif add_bias:
			bias = pd.Series(np.ones((self.covariates.shape[0])), index=self.covariates.index)
			self.covariates['bias'] = bias
		elif (add_bias==False) and self.covariates is None:
			raise NotImplementedError("currently a model with neither bias term, nor covariates is not supported.")

		if R2 is None:
			self.R2 = np.eye(self.sample_count)
		else:
			self.R2 = R2

		if C1 is None:
			self.C1 = np.eye(self.phenotype.shape[1])
		else:
			self.C1 = C1

		if C2 is None:
			self.C2 = np.eye(self.phenotype.shape[1])
		else:
			self.C2 = C2

		self.A_snps = A_snps
		self.interact_with_snp = interact_with_snp
		self.nGridH2 = nGridH2
		self.lmm = kron_gwas.KroneckerGWAS(Y=self.phenotype.values, R1=self.K, C1=self.C1, R2=self.R2, C2=self.C2, X=self.covariates.values, A=[None], h2=0.5, reml=True, X_snps=None, A_snps=None)
		self.lmm.find_h2()

	def compute_association(self, blocksize=10000, temp_dir=None):
		result = None

		if temp_dir is not None:
			if (not os.path.exists(temp_dir)) or (not os.path.isdir(temp_dir)):
				os.makedirs(temp_dir)
		self._generate_intervals(blocksize=blocksize, total_snps=self.snps_test.sid.shape[0])
		for stop in range(1,len(self.intervals)):
			res = self.snps_test_block(block_start=self.intervals[stop-1],block_end=self.intervals[stop])
			if temp_dir is not None:
				self.save_results_block(dir_name=temp_dir, res=res, intervals=self.intervals, idx_interval=stop)
			else:
				if result is None:
					result = res
				else:
					result = pd.concat([result,res])
		return result

	def save_results_block(self, dir_name, res, intervals=None, idx_interval=None):
		mydir = dir_name + "/"
		create_dir(mydir)
		if intervals is not None:
			fill = int(np.floor(np.log10(len(intervals))))+1
		else:
			fill = 0
		if idx_interval is None:
			stop = 1
		else:
			stop = idx_interval
		#block_format = '%0*d' % (fill, stop)
		myfile = "%sblock_%0*d.csv" % (mydir, fill, stop)
		res.to_csv(myfile)

	def snps_test_block(self, block_start, block_end):
		snps = self.snps_test[:,block_start:block_end].read().standardize(self.standardizer)
		lrts, p_values = self.lmm.run_gwas(snps=snps.val, A_snps=self.A_snps)
		result = self._format_gwas_results(lrts=lrts, p_values=p_values, snps=snps, h2=self.lmm.h2)
		return result

	def _format_gwas_results(self, lrts, p_values, snps, h2):
		items = [
			('SNP', snps.sid),
			('Chr', snps.pos[:,0]), 
			('GenDist', snps.pos[:,1]),
			('ChrPos', snps.pos[:,2]), 
			('PValue', p_values),
			('lrt', lrts),
			('Nullh2', np.zeros((snps.sid_count)) + h2)
		]
		return pd.DataFrame.from_items(items)

if __name__ == "__main__":

	bed_fn = "../../test/data/plinkdata/toydata"
	pheno_fn = bed_fn + ".phe6"#"../../test/data/plinkdata/toydata.phe"
	covariate_fn = 	bed_fn + ".phe"

	
	blocksize = 20000
	snp_reader = Bed(bed_fn)#[:,0:50000]
	#snp_reader.run_once()
	if 1:
		standardizer = pysnptools.standardizer.Unit()
	else:
		standardizer = pysnptools.standardizer.Beta(1,2)
	pheno = pysnptools.util.pheno.loadPhen(filename=pheno_fn,   missing ='-9')
	pheno = GWAS._pheno_fixup(pheno)
	covariates = GWAS._pheno_fixup(covariate_fn, iid_source_if_none=pheno)
	print "intersecting data"
	t00 = time.time()
	snp_intersect, pheno_intersect = pysnptools.util.intersect_apply([snp_reader, pheno], sort_by_dataset=True)
	t1 = time.time()
	print "done intersecting after %.4fs" % (t1-t00)

	print "building kernel"
	t0 = time.time()
	if 0: 
		#snp_data = snp_intersect.read().standardize()
		snp_data = snp_intersect.read().standardize(standardizer)

		G = snp_data.val

		K = G.dot(G.T)
		K/=K.diagonal().mean()
	else:
		K = snp_intersect.kernel(standardizer=standardizer,blocksize=blocksize)
		K /= K.diagonal().mean()
	t1 = time.time()
	print "done building kernel after %.4fs" % (t1-t0)	

	if 0:
		print "computing Eigenvalue decomposition of K"
		t0 = time.time()
		S,U = la.eigh(K)
		t1 = time.time()
		print "done computing eigenvalue decomposition of kernel after %.4fs" % (t1-t0)	
		
	if 1:
		print "running GWAS"
		t0 = time.time()
		if 1:#LMM with pre-built kernel K
			mygwas = MultivariateGWAS(K=K, snps_K=None, snps_test=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)
		elif 0:#LMM building kernel inside from snps_K
			mygwas = MultivariateGWAS(K=None, snps_K=snp_intersect, snps_test=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)

		else:#linear regression
			mygwas = MultivariateGWAS(K=None, snps_K=None, snps_test=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)
		if 1:
			result = mygwas.compute_association(blocksize=blocksize, temp_dir=None)#'./temp_dir_testdata/')
		else:
			result_block = mygwas.snps_test_block(block_start=0, block_end=blocksize)
			mygwas.save_results_block("./res_check/", result_block)
		t1 = time.time()
		print "done running GWAS after %.4fs" % (t1-t0)
		print "total: %.4fs" % (t1-t00)