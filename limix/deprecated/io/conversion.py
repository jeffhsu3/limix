
import h5py
import pandas
from optparse import OptionParser
import scipy as sp
import os
import pdb
import subprocess, sys, os
import limix.deprecated.io.plink as PLINK
from numpy import repeat

from numpy import uint32


class LIMIX_converter(object):
    '''
    A class to help with file conversion in LIMIX
    '''
    __slots__=["options","args","data", "result","infostring","hdf"]

    def __init__(self,infostring=None):
        '''
        nothing to initialize
        '''
        self.options=None
        self.args=None
        self.data=None
        self.result={}
        self.infostring=infostring
        if self.infostring is not None:
            self.result["infostring"]=self.infostring
        pass

    def parse_args(self):
        usage = "usage: %prog [options]"
        parser = OptionParser(usage=usage)
        parser.add_option("-O","--outfile", action="store", dest='outfile', 
                type=str, help='The output hdf5 file wiht the resulting data', 
                default="example_out")
        parser.add_option("-P","--plink",action="store", dest='plink', 
                help="Read genotype from binary plink file (filename)", 
                default=None)
        parser.add_option("-G","--g012",action="store", dest='g012', 
                help="Read genotype from 012 file generated using vcftools (filename)", 
                default=None)
        parser.add_option("-d","--g012_dosage",action="store", dest='g012_dosage', 
                help="Read genotype dosage file generated using vcftools (filename)", 
                default=None)
        parser.add_option("-c","--chrom",action="store", dest='chrom', 
                help="Select specific chromosome for conversion (default: all)", default=None)
        parser.add_option("-s","--start",action="store", dest='start', 
                help="Specify start position for conversion (default: all)", default=None)
        parser.add_option("-e","--end",action="store", dest='end', 
                help="Specify start position for conversion (default: all)", default=None)
        parser.add_option("-C","--csv",action="store", dest='csv', 
                help="Phenotype csv file", default=None)
        parser.add_option("-T","--csv_transpose",action="store_true", 
                dest='csv_transpose', help="transpose phenotpye csv file (standard is rows: samples)", default=False)
        parser.add_option("-D","--csv_sep",action="store", dest='csv_sep', 
                help="CSV delimiter (standard: auto)", default=None)
        parser.add_option("-S","--csv_num_samples",action="store", dest='csv_num_samples', help="Number of samples to parse (all)", default=None)
        parser.add_option("-Q","--csv_num_phenotypes",action="store", dest='csv_num_phenotypes', help="Number of phenotypes to parse (all)", default=None)
        parser.add_option("-R", "--meQTL_geno", action="store",
                dest="matrix_eQTL_geno", help="matrix_eQTL style genotypes")
        parser.add_option("-K", "--add", action="store_true",
                dest="add_existing", 
                help=("add to existing hdf5 file"
                    "the matrix_eQTL option"))
        parser.add_option("-A", "--snp_annot", action="store",
                dest="snp_annot", default=False, 
                help=("matrix_eQTL style snp annotation required for using"
                    "the matrix_eQTL option"))

        (self.options, self.args) = parser.parse_args()
        self.result["options"]=str(self.options)
        return (self.options,self.args)

    def run(self):
        """
        Run the conversion job
        """
        #open hdfstore
        self.hdf = h5py.File(self.options.outfile)

        #read genotype plink file?
        if self.options.csv is not None:
            self.convert_phenotype_csv(self.hdf,self.options.csv,
                    self.options.csv_sep,self.options.csv_transpose,
                    num_samples=self.options.csv_num_samples,
                    num_phenotypes=self.options.csv_num_phenotypes)
        if self.options.plink is not None:
            self.convert_plink(self.hdf,self.options.plink,self.options.chrom,
                    self.options.start,self.options.end)
        if self.options.g012 is not None:
            self.convert_g012(self.hdf,self.options.g012,self.options.chrom,
                    self.options.start,self.options.end)
        if self.options.g012_dosage is not None:
            self.convert_g012_dosage(self.hdf,self.options.g012_dosage,self.options.chrom,
                    self.options.start,self.options.end)
        if self.options.matrix_eQTL_geno is not None:
            self.convert_meQTL_dosage(self.hdf, self.options.matrix_eQTL_geno,
                    self.options.snp_annot,self.options.chrom, 
                    self.options.start, self.options.end, add=self.options.add_existing)
        pass

    def convert_phenotype_csv(self,hdf,csv_file,sep=None, 
            transpose=False,num_samples=None,
            num_phenotypes=None,*args,**kw_args):
        """Convert phenotype csv file to LIMIX hdf5

        Arguments
        ---------
        hdf:  
            handle for hdf5 file (target)
        csv_file: 
            filename of csv file with phenoyptes
        transpose: boolean 
            standard is rows - individuals unless transpose = True
        """
        C = pandas.io.parsers.read_csv(csv_file,sep=sep,header=None,index_col=False,
                *args,**kw_args)
        if transpose:
            C = C.T
        values = C.values
        if num_samples is not None:
            values = values[0:1+int(num_samples)]
        if num_phenotypes is not None:
            values = values[:,0:1+int(num_phenotypes)]
        matrix = sp.array(values[1::,1::],dtype='float')
        #TODO: check if sample_IDs are present, and if they may be 2 columns
        sample_IDs = sp.array(values[1::,0],dtype='str')
        #TODO: check if pheno_IDs are present
        pheno_IDs  = sp.array(values[0,1::],dtype='str')
        #store
        if 'phenotype' in hdf.keys():
            del(hdf['phenotype'])
        phenotype = hdf.create_group('phenotype')
        col_header = phenotype.create_group('col_header')
        row_header = phenotype.create_group('row_header')
        #write phenotype
        phenotype.create_dataset(name='matrix',data=matrix)
        #write row header
        row_header.create_dataset(name='sample_ID',data=sample_IDs)
        #write col header
        col_header.create_dataset(name='phenotype_ID',data=pheno_IDs)
        pass


    def convert_g012(self,hdf,g012_file,chrom,start,end):
        """convert g012 file to LIMIX hdf5
        hdf: handle for hdf5 file (target)
        g012_file: filename of g012 file
        chrom: select chromosome for conversion
        start: select start position for conversion
        end:  select end position for conversion
        """
        if ((start is not None) or (end is not None) or (chrom is not None)):
            print "cannot handle start/stop/chrom boundaries for g012 file"
            return
        #store
        if 'genotype' in hdf.keys():
            del(hdf['genotype'])
        genotype = hdf.create_group('genotype')
        col_header = genotype.create_group('col_header')
        row_header = genotype.create_group('row_header')
        #load position and meta information
        indv_file = g012_file + '.indv'
        pos_file  = g012_file + '.pos'
        sample_ID = sp.loadtxt(indv_file,dtype='str')
        pos  = sp.loadtxt(pos_file,dtype='str')
        chrom = pos[:,0]
        pos   = sp.array(pos[:,1],dtype='int')

        row_header.create_dataset(name='sample_ID',data=sample_ID)
        col_header.create_dataset(name='chrom',data=chrom)
        col_header.create_dataset(name='pos',data=pos)
        M = sp.loadtxt(g012_file,dtype='uint8')
        snps = M[:,1::]
        genotype.create_dataset(name='matrix',data=snps,chunks=(snps.shape[0],
            min(10000,snps.shape[1])),compression='gzip')
        pass

    def convert_g012_dosage(self,hdf,g012_file,chrom,start,end):
        """convert g012 file to LIMIX hdf5
        hdf: handle for hdf5 file (target)
        g012_file: filename of g012 file
        chrom: select chromosome for conversion
        start: select start position for conversion
        end:  select end position for conversion
        """
        if ((start is not None) or (end is not None) or (chrom is not None)):
            print "cannot handle start/stop/chrom boundaries for g012 file"
            return
        #store
        if 'genotype' in hdf.keys():
            del(hdf['genotype'])
        genotype = hdf.create_group('genotype')
        col_header = genotype.create_group('col_header')
        row_header = genotype.create_group('row_header')
        #load position and meta information
        indv_file = g012_file + '.indv'
        pos_file  = g012_file + '.pos'
        sample_ID = sp.loadtxt(indv_file,dtype='str')
        pos  = sp.loadtxt(pos_file,dtype='str')
        chrom = pos[:,0]
        pos   = sp.array(pos[:,1],dtype='int')
        # load marker info
        ids_file = g012_file + '.ids'
        ids = sp.loadtxt(ids_file,dtype='str')
        gdid = ids[:,2]
        ref = ids[:,3]
        alt = ids[:,4]
        impute = ids[:,5]
        row_header.create_dataset(name='sample_ID',data=sample_ID)
        col_header.create_dataset(name='chrom',data=chrom)
        col_header.create_dataset(name='pos',data=pos)
        col_header.create_dataset(name='gdid',data=gdid)
        col_header.create_dataset(name='ref',data=ref)
        col_header.create_dataset(name='alt',data=alt)
        col_header.create_dataset(name='IMP2',data=impute)
        M = sp.loadtxt(g012_file,dtype='float')
        snps = M[:,:-1]
        genotype.create_dataset(name='matrix',data=snps,
                chunks=(snps.shape[0],min(10000,snps.shape[1])),
                compression='gzip')
        pass

    def convert_meQTL_dosage(self, hdf, matrix_eQTL_dosage, 
            snp_annotation, chrom, start, end, add=False):
        """Convert matrix-eQTL input for limix
        Parameters
        ----------
        snp_annotation : pickle file
            a pickle file of snp annotations with 
        Return
        ------
        None
        """
        if add:
            genotype = hdf['genotype']
            col_header = hdf['genotype']['col_header']
            row_header = hdf['genotype']['row_header']
            prev = col_header[col_header.items()[0][0]].shape[0]
        else:
            if 'genotype' in hdf.keys() and not add:
                del(hdf['genotype'])
            genotype = hdf.create_group('genotype')
            col_header = genotype.create_group('col_header')
            row_header = genotype.create_group('row_header')
        snp_annot = pandas.read_pickle(snp_annotation)
        chrom = repeat(int(chrom), snp_annot.shape[0])
        gid = snp_annot.index.values.copy()
        gid = gid.astype('|S16')
        refa = snp_annot['a0'].values.astype('|S8')
        alta = snp_annot['a1'].values.astype('|S8')
        with open(matrix_eQTL_dosage) as f:
            header = f.readline().split(' ')
            ncol = len(header)
        M = sp.loadtxt(matrix_eQTL_dosage,  skiprows=1, 
                usecols=range(1,ncol), dtype='float').T
        assert( len(gid) == M.shape[1])
        ############## What to do with header?
        if add:
            for i in col_header.items():
                keyi = i[0]
                col_header[keyi].resize((len(chrom) + prev,))
                print(i)
            col_header['chrom'][prev:] = chrom
            col_header['pos'][prev:] = snp_annot['pos'].astype(uint32)
            col_header['gdid'][prev:] = gid
            col_header['ref'][prev:] = refa
            col_header['alt'][prev:] = alta
            genotype['matrix'].resize((M.shape[0], M.shape[1] + prev))
            genotype['matrix'][:,prev:] = M
        else:
            row_header.create_dataset(name='sample_ID', data=header, 
                    chunks=True)
            col_header.create_dataset(name='chrom', 
                    data=chrom, maxshape=(None,), chunks=True)
            col_header.create_dataset(name='pos',
                    data=snp_annot['pos'].astype(uint32), 
                    maxshape=(None,), chunks=True)
            col_header.create_dataset(name='gdid',
                    data=gid, maxshape=(None,), chunks=True)
            col_header.create_dataset(name='ref', 
                    data=refa, maxshape=(None,), chunks=True)
            col_header.create_dataset(name='alt', 
                    data=alta, maxshape=(None,), chunks=True)
            #assert(snp_annot.shape[0] == M.shape[0])
            genotype.create_dataset(name='matrix', data=M, chunks=(M.shape[0],
                min(10000, M.shape[1])), maxshape=(M.shape[0], None), compression='gzip')

    def convert_plink(self, hdf, bed_file, chrom, start, end):
        """Convert plink file to LIMIX hdf5

        Parameters
        ----------
        hdf : h5py.File
            handle for hdf5 file (target)
        bed_file : filename of bed file
        chrom : select chromosome for conversion
        start : select start position for conversion
        end :  select end position for conversion

        Returns
        -------
        None
        """
        #TODO: check that this is a valid plink file
        #start_pos?
        startpos = None
        endpos = None
        if (start is not None) or (end is not None):
            if chrom is None:
                raise Exception('start or end pos specification requires a target chromosome')
        if start is not None:
            startpos = [chrom,0,start]
        if end is not None:
            endpos = [chrom,0,end]
        #read
        data=PLINK.readBED(bed_file,startpos=startpos,endpos=endpos)
        #store
        if 'genotype' in hdf.keys():
            del(hdf['genotype'])
        genotype = hdf.create_group('genotype')
        col_header = genotype.create_group('col_header')
        row_header = genotype.create_group('row_header')
        #write genotype
        genotype.create_dataset(name='matrix',data=data['snps'],
                chunks=(data['snps'].shape[0],min(10000,data['snps'].shape[1])))
        #write row header
        row_header.create_dataset(name='sample_ID',data=data['iid'][:,0])
        #write col header
        col_header.create_dataset(name='chrom',data=data['pos'][:,0])
        col_header.create_dataset(name='pos',data=data['pos'][:,2])
        col_header.create_dataset(name='alleles',data=data['alleles'])
        pass
