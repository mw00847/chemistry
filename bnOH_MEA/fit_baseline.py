import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt




import scipy
import rampy

from sklearn import preprocessing

from sklearn import decomposition
from scipy.signal import savgol_filter

#wavenumbers
#df1=pd.read_csv('80_20_bnOH-MEA.csv',usecols=[0])

#these are the variables
y=np.genfromtxt('10_90_bnOH-MEA.CSV',delimiter=',',usecols=(1))
x=np.genfromtxt('80_20_bnOH-MEA.CSV',delimiter=',',usecols=(0))



# need to define some fitting regions for the spline
roi = np.array([[1755,2370],[3642,3966]])

# calculating the baselines
ycalc_poly, base_poly = rampy.baseline(x, y, roi, 'poly', polynomial_order=4)
#ycalc_gcvspl, base_gcvspl = rampy.baseline(x,y,roi,'gcvspline',s=0.1 ) # activate if you have installed gcvspline
ycalc_uni, base_uni = rampy.baseline(x, y, roi, 'unispline', s=1e0)
ycalc_als, base_als = rampy.baseline(x, y, roi, 'als', lam=10**5, p=0.05)
ycalc_arpls, base_arpsl = rampy.baseline(x, y, roi, 'arPLS', lam=10**6, ratio=0.001)
ycalc_drpls, base_drpsl = rampy.baseline(x, y, roi, 'drPLS')
ycalc_rubberband, base_rubberband = rampy.baseline(x, y, roi, 'rubberband')

# doing the figure
#plt.figure(dpi=150)
#plt.plot(x, y, "k-", label="Raw")
#plt.plot(x, bkg, "r-", label="True background")
plt.plot(x,y)
plt.plot(x, base_poly, "-", color="grey", label="polynomial")
plt.plot(x, base_uni, "b-", label="unispline baseline")
#plt.plot(x,base_gcvspl,"-",color="orange",label="gcvspline baseline") # activate if you have installed gcvspline
plt.plot(x, base_als, ":", color="purple", label="als baseline")
plt.plot(x, base_arpsl, "-.", color="cyan", label="arPLS baseline")
plt.plot(x, base_drpsl, "--", color="orange", label="drPLS baseline")
plt.plot(x, base_rubberband, "--", color="green", label="rubberband baseline")

plt.xlabel("X")
plt.ylabel("Y")

plt.legend()
plt.show()