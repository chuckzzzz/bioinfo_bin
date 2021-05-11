# Author: Ziyang Zhang
# Description: split a vcf file by chromosome and build index if specified
# Input: a vcf file, gzipped or non-zipped  
#        an output directory to store the files 
# Output: Output separate files for each of the chromosomes into specified directory 
# Example: python split.py -f /storage/czhang/projects/STR_expression/data/hg38.SNP.gz -o /storage/czhang/projects/STR_expression/data/hg38_SNP_genotypes/ -index -header

import argparse
import subprocess
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Split vcf by chromosome and build index')
parser.add_argument('-f', type=str, dest="f", help='input vcf file')
parser.add_argument('-o', type=str, default="./", dest="o", help='output directory')
parser.add_argument("-index", action="store_true",help='whether to build index')
parser.add_argument('-header', dest='header', default=False, action='store_true')

args = parser.parse_args()

# credit to stackoverflow
def is_gz_file(filepath):
    magic=b'\x1f\x8b'
    with open(filepath, 'rb') as test_f:
        a=test_f.read(2)
        return a == magic

if __name__=="__main__":
    vcf_file=args.f
    output_dir=args.o
    build_index=args.index
    store_header=args.header

    print("INFO: input file: {}\nINFO: output directory: {}\nINFO: build index {}\nINFO: include header {}".format(vcf_file,output_dir,str(build_index),str(store_header)))
    
    if is_gz_file(vcf_file)==False:
        print("INFO: vcf file not gzipped, gzipping")
        subprocess.check_call(["bgzip","-c",vcf_file,">",vcf_file+".gz"], shell=True)
        vcf_file=vcf_file+".gz"
    vcf_dir=os.path.dirname(vcf_file)
    if os.path.isfile(vcf_file+".tbi")==False:
        print("INFO: building vcf index...")
        subprocess.check_call(["tabix","-p","vcf",vcf_file])
    
    for i in tqdm(range(1,23)):
        cur_chrom="chr"+str(i)
        cur_output=output_dir+cur_chrom
        try:
            with open(cur_output,'wb') as f:
                if(store_header):
                    subprocess.check_call(["tabix","-h",vcf_file,cur_chrom], stdout=f)
                else:
                    subprocess.check_call(["tabix",vcf_file,cur_chrom], stdout=f)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        subprocess.check_call(["bgzip",cur_output])
        if build_index:
            subprocess.check_call(["tabix","-p","vcf",cur_output+".gz"])    
