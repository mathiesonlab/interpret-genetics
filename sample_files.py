import pickle
import vcf 

def main():

    with open("samples/AFR.txt", "w+") as f:
        reader = vcf.Reader(open("/home/smathieson/public/cs68/1000g/AFR_135-136Mb.chr2.vcf", 'r'))
        record = next(reader)
        for call in record.samples:
            f.write(call.sample + "\n")

main()
