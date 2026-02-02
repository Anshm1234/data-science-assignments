import numpy as np
import pandas as pd


#step 1
df=pd.read_csv("data.csv", encoding="latin1")
x=df["no2"]
x=x.dropna()

r=102317159

ar=0.05*(r%7)

br=0.3*((r%5)+1)

z=x.apply(lambda val:val+ar*np.sin(br*val))


#step 2
m=np.mean(z)
st_dev=np.std(z)
var=st_dev*st_dev


lam=1/(2*var)
c=1/(np.sqrt(2*np.pi*var))

print(m)
print(lam)
print(c)
p_hat = c * np.exp(-lam * (z - m)**2)
