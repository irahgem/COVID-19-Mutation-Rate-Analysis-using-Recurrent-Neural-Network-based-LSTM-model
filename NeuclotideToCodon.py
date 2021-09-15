import xml.etree.ElementTree as ET
import pandas as pd

df = pd.read_excel("reference.xlsx", usecols="A")
reference = df.iloc[0][0]

df1 = pd.read_excel("sequence.xlsx", usecols="A")
dataset = [df1.iloc[i][0] for i in range(0,82)]
df2 = pd.read_excel("sequence.xlsx", usecols="B")
date = [df2.iloc[i][0] for i in range(0,82)]

CodonIndex = {'TTT':1,'TTC':2,'TTA':3,'TTG':4,'TCT':5,'TCC':6,'TCA':7,'TCG':8,'TAT':9,'TAC':10,'TAA':11,'TAG':12,'TGT':13,'TGC':14,'TGA':15,'TGG':16,
              'CTT':17,'CTC':18,'CTA':19,'CTG':20,'CCT':21,'CCC':22,'CCA':23,'CCG':24,'CAT':25,'CAC':26,'CAA':27,'CAG':28,'CGT':29,'CGC':30,'CGA':31,'CGG':32,
              'ATT':33,'ATC':34,'ATA':35,'ATG':36,'ACT':37,'ACC':38,'ACA':39,'ACG':40,'AAT':41,'AAC':42,'AAA':43,'AAG':44,'AGT':45,'AGC':46,'AGA':47,'AGG':48,
              'GTT':49,'GTC':50,'GTA':51,'GTG':52,'GCT':53,'GCC':54,'GCA':55,'GCG':56,'GAT':57,'GAC':58,'GAA':59,'GAG':60,'GGT':61,'GGC':62,'GGA':63,'GGG':64}

codon=[]

for seq in dataset:
    seq = seq.upper()
    n = len(seq)

    cd = []
    for i in range(0,n-3+1,3):
        cd.append(str(CodonIndex[seq[i:i+3]]))
    cd = "+".join(cd)
    codon.append(cd)

df = pd.DataFrame({'Sequence':codon, 'Date':date})
writer = pd.ExcelWriter('sequenceCodon.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()

seq = reference.upper()
n = len(seq)
rd = []
rcodon=[]
for i in range(0,n-3+1,3):
    rd.append(str(CodonIndex[seq[i:i+3]]))
rd = "+".join(rd)
rcodon.append(rd)

df = pd.DataFrame({'Sequence':codon})
writer = pd.ExcelWriter('referenceCodon.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()
