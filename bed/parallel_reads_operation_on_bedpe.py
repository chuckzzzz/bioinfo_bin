from itertools import islice
import gzip
import argparse
from multiprocessing import Pool
import os
import time


parser = argparse.ArgumentParser(description='Script to calculate midpoints of reads')
parser.add_argument('-i', help='path to input bedpe file', required=True)
parser.add_argument('-n', help='number of parallel processes to run', required=True,type=int)
parser.add_argument('-t', help='temp directory', required=True)
parser.add_argument('-o', help='output file', required=True)

args = parser.parse_args()

input_bedpe=args.i
num_worker=args.n 
temp_dir=args.t
output_file=args.o

def file_len(fname,gzipped=False):
    if gzipped:
        with gzip.open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    else:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
def is_gz_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def operation(f_iterator,start,end,out_file,gzipped):

    if(gzipped):
        f_iterator=gzip.open(input_bedpe,'r')
    else:
        f_iterator=open(input_bedpe,'r')
    with open(out_file,"w+") as f_output:
        for line in islice(f_iterator,start,end):
            skip=False
            line=line.decode('utf-8')
            line=line.strip('\n').split('\t')
            chr_coord=line[0]
            if(line[-1]=='-'):
                R1=line[1],line[2]
                R2=line[4],line[5]

            else:
                R2=line[1],line[2]
                R1=line[4],line[5]

            # not properly paired 
            if(R2[1]<R1[0]):
                continue
            read_coord=R1[0],R2[1]
            length=int(read_coord[1])-int(read_coord[0])
            out_line='\t'.join([chr_coord,read_coord[0],read_coord[1],str(length)])
            if(length<0):
                # print(out_line)
                skip=True
            if(skip==False):
                f_output.write("%s\n" % out_line)
    f_iterator.close()

# for large txt files
def join_files(input_files, output_file,delete=False):
    with open(output_file, 'w+') as outfile:
        for fname in input_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    if(delete):
        for f in input_files:
            os.remove(f)

def main(input_bedpe,num_worker,temp_dir,out_file):
    gzipped=is_gz_file(input_bedpe)
    line_number=file_len(input_bedpe,gzipped)
    segment_size=line_number//num_worker
    segment_index=[segment_size*i for i in range(num_worker)]
    segment_index.append(line_number)
    arg_list=[]
    temp_files=[]
    for i in range(len(segment_index)-1):
        start,end=segment_index[i],segment_index[i+1]
        temp_file=temp_dir+"%s"%('_'.join([str(start),str(end)])+".bed")
        temp_files.append(temp_file)
        arg_list.append((input_bedpe,start,end,temp_file,gzipped))
    print("INFO: Multiprocess on %d cores"%num_worker)
    pool=Pool(processes=num_worker)
    pool.starmap(operation,arg_list)
    pool.close()
    pool.join()

    join_files(temp_files,out_file,delete=True)

if __name__=="__main__":
    start_time = time.time()
    main(input_bedpe,num_worker,temp_dir,output_file)
    print("INFO: Program finished in %s seconds!" % (time.time() - start_time))

# python 01.parallel_reads_operation_on_bedpe.py -i /mnt/tscc/yangli/projects/heroin/00.data/Heroin_2/Heroin_2.DNA.bedpe.gz -n 10 -t /mnt/silencer2/home/ziz361/projects/heroin/pipeline/test_files/ -o /mnt/silencer2/home/ziz361/projects/heroin/pipeline/test_files/heroin_2.bed

# pair end reads explanation: https://informatics.fas.harvard.edu/atac-seq-guidelines-old-version.html