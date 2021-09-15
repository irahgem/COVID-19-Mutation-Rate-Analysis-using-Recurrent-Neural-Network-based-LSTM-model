import numpy as np
import pandas as pd

df = pd.read_excel("referenceCodon.xlsx", usecols="A")
reference = df.iloc[0][0]
reference = reference.split("+")

df1 = pd.read_excel("sequenceCodon.xlsx", usecols="A")
dataset = [df1.iloc[i][0] for i in range(0,82)]
dataset = [codon.split("+") for codon in dataset]
mutation = np.zeros((64,64))

for i in range(0,len(dataset)):
    for j in range(0,min(len(reference),len(dataset[i]))):
        D1 = dataset[i][j]
        D2 = reference[j]
        if D1!=D2:
            mutation[int(D1)-1][int(D2)-1] += 1

MutationRate = (mutation/(len(dataset)*len(reference)))*100
print(MutationRate)

import matplotlib.pyplot as p

cmap = p.imshow(MutationRate,interpolation='None',cmap="Blues",origin='upper',vmin=0,vmax=0.13)
p.colorbar()
p.show()