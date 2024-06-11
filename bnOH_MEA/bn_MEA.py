#this takes a series of spectra and plots them,
#then takes the average of the spectra in the range 1200-1210,
#to be used to create calibration curve

#peaks here are 3352, 3287,2922, 2857, 1595,1453,1355,1076,1030,941,866,738,699



#import libraries
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from scipy.signal import savgol_filter

#wavenumbers
#df1=pd.read_csv('80_20_bnOH-MEA.csv',usecols=[0])

#the arrays of the spectra

array1=np.genfromtxt('10_90_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array2=np.genfromtxt('20_80_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array3=np.genfromtxt('30_70_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array4=np.genfromtxt('40_60_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array5=np.genfromtxt('50_50_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array6=np.genfromtxt('60_40_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array7=np.genfromtxt('70_30_bnOH-MEA.CSV',delimiter=',',usecols=(1))
array8=np.genfromtxt('80_20_bnOH-MEA.CSV',delimiter=',',usecols=(1))


#the wavenumbers of the spectra

wavenumbers=np.genfromtxt('80_20_bnOH-MEA.CSV',delimiter=',',usecols=(0))


#put the arrays into a matrix

data=np.array([wavenumbers,array1,array2,array3,array4,array5,array6,array7,array8])

data2=np.array([array1,array2,array3,array4,array5,array6,array7,array8])

wavenumbers1=np.array([wavenumbers,wavenumbers,wavenumbers,wavenumbers,wavenumbers,wavenumbers,wavenumbers,wavenumbers])

#plot the spectra

#plt.plot(wavenumbers,array1)
#plt.plot(wavenumbers,array2)
#plt.plot(wavenumbers,array3)
#plt.plot(wavenumbers,array4)
#plt.plot(wavenumbers,array5)
#plt.plot(wavenumbers,array6)
#plt.plot(wavenumbers,array7)
#plt.plot(wavenumbers,array8)
#plt.show()

#first derivative of the spectra

deriv1 = savgol_filter(data2, window_length=3, polyorder=2, deriv=1)

#plot the first derivative of the spectra

#plt.plot(wavenumbers1,deriv1)
#plt.show()



#take the average of the spectra in the range 1200-1210, to be used to create calibration curve
#peaks here are 3352, 3287,2922, 2857, 1595,1453,1355,1076,1030,941,866,738,699
#how to shorten this code?

b=np.ma.masked_outside(wavenumbers,1208,1209).mask
c=np.ma.masked_outside(wavenumbers,3351,3353).mask
d=np.ma.masked_outside(wavenumbers,3286,3288).mask
e=np.ma.masked_outside(wavenumbers,2921,2923).mask
f=np.ma.masked_outside(wavenumbers,2856,2858).mask
g=np.ma.masked_outside(wavenumbers,1594,1596).mask
h=np.ma.masked_outside(wavenumbers,1452,1454).mask
i=np.ma.masked_outside(wavenumbers,1354,1356).mask
j=np.ma.masked_outside(wavenumbers,1075,1077).mask
k=np.ma.masked_outside(wavenumbers,1029,1031).mask
l=np.ma.masked_outside(wavenumbers,940,942).mask
m=np.ma.masked_outside(wavenumbers,865,867).mask
n=np.ma.masked_outside(wavenumbers,737,739).mask
o=np.ma.masked_outside(wavenumbers,698,700).mask

y2=wavenumbers[~b]
y3=wavenumbers[~c]
y4=wavenumbers[~d]
y4=wavenumbers[~e]
y4=wavenumbers[~f]
y5=wavenumbers[~g]
y6=wavenumbers[~h]
y7=wavenumbers[~i]
y8=wavenumbers[~j]
y9=wavenumbers[~k]
y10=wavenumbers[~l]
y11=wavenumbers[~m]
y12=wavenumbers[~n]
y13=wavenumbers[~o]


x1=array1[~b]
x2=array2[~b]
x3=array3[~b]
x4=array4[~b]
x5=array5[~b]
x6=array6[~b]
x7=array7[~b]
x8=array8[~b]

b1=array1[~b]
b2=array2[~b]
b3=array3[~b]
b4=array4[~b]
b5=array5[~b]
b6=array6[~b]
b7=array7[~b]
b8=array8[~b]

c1=array1[~c]
c2=array2[~c]
c3=array3[~c]
c4=array4[~c]
c5=array5[~c]
c6=array6[~c]
c7=array7[~c]
c8=array8[~c]

d1=array1[~d]
d2=array2[~d]
d3=array3[~d]
d4=array4[~d]
d5=array5[~d]
d6=array6[~d]
d7=array7[~d]
d8=array8[~d]

e1=array1[~e]
e2=array2[~e]
e3=array3[~e]
e4=array4[~e]
e5=array5[~e]
e6=array6[~e]
e7=array7[~e]
e8=array8[~e]

f1=array1[~f]
f2=array2[~f]
f3=array3[~f]
f4=array4[~f]
f5=array5[~f]
f6=array6[~f]
f7=array7[~f]
f8=array8[~f]

g1=array1[~g]
g2=array2[~g]
g3=array3[~g]
g4=array4[~g]
g5=array5[~g]
g6=array6[~g]
g7=array7[~g]
g8=array8[~g]

h1=array1[~h]
h2=array2[~h]
h3=array3[~h]
h4=array4[~h]
h5=array5[~h]
h6=array6[~h]
h7=array7[~h]
h8=array8[~h]

i1=array1[~i]
i2=array2[~i]
i3=array3[~i]
i4=array4[~i]
i5=array5[~i]
i6=array6[~i]
i7=array7[~i]
i8=array8[~i]

j1=array1[~j]
j2=array2[~j]
j3=array3[~j]
j4=array4[~j]
j5=array5[~j]
j6=array6[~j]
j7=array7[~j]
j8=array8[~j]

k1=array1[~k]
k2=array2[~k]
k3=array3[~k]
k4=array4[~k]
k5=array5[~k]
k6=array6[~k]
k7=array7[~k]
k8=array8[~k]

l1=array1[~l]
l2=array2[~l]
l3=array3[~l]
l4=array4[~l]
l5=array5[~l]
l6=array6[~l]
l7=array7[~l]
l8=array8[~l]

m1=array1[~m]
m2=array2[~m]
m3=array3[~m]
m4=array4[~m]
m5=array5[~m]
m6=array6[~m]
m7=array7[~m]
m8=array8[~m]

n1=array1[~n]
n2=array2[~n]
n3=array3[~n]
n4=array4[~n]
n5=array5[~n]
n6=array6[~n]
n7=array7[~n]
n8=array8[~n]

o1=array1[~o]
o2=array2[~o]
o3=array3[~o]
o4=array4[~o]
o5=array5[~o]
o6=array6[~o]
o7=array7[~o]
o8=array8[~o]




b1=np.average(b1)
b2=np.average(b2)
b3=np.average(b3)
b4=np.average(b4)
b5=np.average(b5)
b6=np.average(b6)
b7=np.average(b7)
b8=np.average(b8)

c1=np.average(c1)
c2=np.average(c2)
c3=np.average(c3)
c4=np.average(c4)
c5=np.average(c5)
c6=np.average(c6)
c7=np.average(c7)
c8=np.average(c8)

d1=np.average(d1)
d2=np.average(d2)
d3=np.average(d3)
d4=np.average(d4)
d5=np.average(d5)
d6=np.average(d6)
d7=np.average(d7)
d8=np.average(d8)

e1=np.average(e1)
e2=np.average(e2)
e3=np.average(e3)
e4=np.average(e4)
e5=np.average(e5)
e6=np.average(e6)
e7=np.average(e7)
e8=np.average(e8)

f1=np.average(f1)
f2=np.average(f2)
f3=np.average(f3)
f4=np.average(f4)
f5=np.average(f5)
f6=np.average(f6)
f7=np.average(f7)
f8=np.average(f8)

g1=np.average(g1)
g2=np.average(g2)
g3=np.average(g3)
g4=np.average(g4)
g5=np.average(g5)
g6=np.average(g6)
g7=np.average(g7)
g8=np.average(g8)

h1=np.average(h1)
h2=np.average(h2)
h3=np.average(h3)
h4=np.average(h4)
h5=np.average(h5)
h6=np.average(h6)
h7=np.average(h7)
h8=np.average(h8)

i1=np.average(i1)
i2=np.average(i2)
i3=np.average(i3)
i4=np.average(i4)
i5=np.average(i5)
i6=np.average(i6)
i7=np.average(i7)
i8=np.average(i8)

j1=np.average(j1)
j2=np.average(j2)
j3=np.average(j3)
j4=np.average(j4)
j5=np.average(j5)
j6=np.average(j6)
j7=np.average(j7)
j8=np.average(j8)

k1=np.average(k1)
k2=np.average(k2)
k3=np.average(k3)
k4=np.average(k4)
k5=np.average(k5)
k6=np.average(k6)
k7=np.average(k7)
k8=np.average(k8)

l1=np.average(l1)
l2=np.average(l2)
l3=np.average(l3)
l4=np.average(l4)
l5=np.average(l5)
l6=np.average(l6)
l7=np.average(l7)
l8=np.average(l8)

m1=np.average(m1)
m2=np.average(m2)
m3=np.average(m3)
m4=np.average(m4)
m5=np.average(m5)
m6=np.average(m6)
m7=np.average(m7)
m8=np.average(m8)

n1=np.average(n1)
n2=np.average(n2)
n3=np.average(n3)
n4=np.average(n4)
n5=np.average(n5)
n6=np.average(n6)
n7=np.average(n7)
n8=np.average(n8)

o1=np.average(o1)
o2=np.average(o2)
o3=np.average(o3)
o4=np.average(o4)
o5=np.average(o5)
o6=np.average(o6)
o7=np.average(o7)
o8=np.average(o8)



#x9=deriv1[~b]



#print(y2)

x1=np.average(x1)
x2=np.average(x2)
x3=np.average(x3)
x4=np.average(x4)
x5=np.average(x5)
x6=np.average(x6)
x7=np.average(x7)
x8=np.average(x8)

print("absorbance 10_90",(x1))
print("absorbance 20_80",(x2))
print("absorbance 30_70",(x3))
print("absorbance 40_60",(x4))
print("absorbance 50_50",(x5))
print("absorbance 60_40",(x6))
print("absorbance 70_30",(x7))
print("absorbance 80_20",(x8))


absorbance=np.array([x1,x2,x3,x4,x5,x6,x7,x8])

absorbance1=np.array([b1,b2,b3,b4,b5,b6,b7,b8])
absorbance2=np.array([c1,c2,c3,c4,c5,c6,c7,c8])
absorbance3=np.array([d1,d2,d3,d4,d5,d6,d7,d8])
absorbance4=np.array([e1,e2,e3,e4,e5,e6,e7,e8])
absorbance5=np.array([f1,f2,f3,f4,f5,f6,f7,f8])
absorbance6=np.array([g1,g2,g3,g4,g5,g6,g7,g8])
absorbance7=np.array([h1,h2,h3,h4,h5,h6,h7,h8])
absorbance8=np.array([i1,i2,i3,i4,i5,i6,i7,i8])
absorbance9=np.array([j1,j2,j3,j4,j5,j6,j7,j8])
absorbance10=np.array([k1,k2,k3,k4,k5,k6,k7,k8])
absorbance11=np.array([l1,l2,l3,l4,l5,l6,l7,l8])
absorbance12=np.array([m1,m2,m3,m4,m5,m6,m7,m8])
absorbance13=np.array([n1,n2,n3,n4,n5,n6,n7,n8])
absorbance14=np.array([o1,o2,o3,o4,o5,o6,o7,o8])










print(absorbance)
labels=['10_90','20_80','30_70','40_60','50_50','60_40','70_30','80_20']
plt.plot(labels,absorbance)
plt.plot(labels,absorbance1)
plt.plot(labels,absorbance2)
plt.plot(labels,absorbance3)
plt.plot(labels,absorbance4)
plt.plot(labels,absorbance5)
plt.plot(labels,absorbance6)
plt.plot(labels,absorbance7)
plt.plot(labels,absorbance8)

plt.show()

