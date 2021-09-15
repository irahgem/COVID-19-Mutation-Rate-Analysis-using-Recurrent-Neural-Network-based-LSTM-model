import numpy as np
import pandas as pd

df = pd.read_excel("reference.xlsx", usecols="A")
reference = df.iloc[0][0]

df1 = pd.read_excel("sequence.xlsx", usecols="A")
dataset = [df1.iloc[i][0] for i in range(0,82)]

mutation = np.zeros((4,4))
nuc = ["a", "c", "g", "t"]

for i in range(0,len(dataset)):
    for j in range(0,min(len(reference),len(dataset[i]))):
        D1 = dataset[i][j]
        D2 = reference[j]
        if D1!=D2:
            mutation[nuc.index(D1)][nuc.index(D2)] += 1

MutationRate = (mutation/(len(dataset)*len(reference)))*100
print(MutationRate)

import matplotlib.pyplot as p

cmap = p.imshow(MutationRate,interpolation='none',cmap="Blues",origin='upper')
p.colorbar()
p.show()