import pandas as pd
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plt
import plotly.express as px


from sklearn import decomposition

#df1 is the wavenumbers

#df1=pd.read_csv('EG_25.csv',usecols=[0])

#these are the variables
df2=pd.read_csv('10_90_bnOH-MEA.csv',usecols=[1])
df3=pd.read_csv('20_80_bnOH-MEA.csv',usecols=[1])
df4=pd.read_csv('30_70_bnOH-MEA.csv',usecols=[1])
df5=pd.read_csv('40_60_bnOH-MEA.csv',usecols=[1])
df6=pd.read_csv('50_50_bnOH-MEA.csv',usecols=[1])
df7=pd.read_csv('60_40_bnOH-MEA.csv',usecols=[1])
df8=pd.read_csv('70_30_bnOH-MEA.csv',usecols=[1])
df9=pd.read_csv('80_20_bnOH-MEA.csv',usecols=[1])



#df8=pd.read_csv("C:\Users\Mat\Desktop\Performance\340\autoshine_1122.csv",usecols=[1])
#df10=pd.read_csv('C:\Users\Mat\Desktop\Performance\340R114.csv',usecols=[1])
#df11=pd.read_csv('C:\Users\Mat\Desktop\Performance\340R124.csv',usecols=[1])
#df12=pd.read_csv('water_MEA5050.csv',usecols=[1])
#df13=pd.read_csv('C:\Users\Mat\Desktop\Performance\340swanley_cambrian.csv',usecols=[1])
#df14=pd.read_csv('C:\Users\Mat\Desktop\Performance\340sittingbourne_IBC.csv',usecols=[1])
#df15=pd.read_csv('C:\Users\Mat\Desktop\Performance\340sittingbourne_tank.csv',usecols=[1])
#df16=pd.read_csv('C:\Users\Mat\Desktop\Performance\340bavarianwheels2_270922.csv',usecols=[1])
#df17=pd.read_csv('C:\Users\Mat\Desktop\Performance\340premier_211022.csv',usecols=[1])
#df18=pd.read_csv('C:\Users\Mat\Desktop\Performance\340premier_211022_residue.csv',usecols=[1])
#df19=pd.read_csv('C:\Users\Mat\Desktop\Performance\340myalloys_290422.csv',usecols=[1])
#df20=pd.read_csv('C:\Users\Mat\Desktop\Performance\340mus_300822.csv',usecols=[1])
#df21=pd.read_csv('C:\Users\Mat\Desktop\Performance\340mus_230522.csv',usecols=[1])
#df22=pd.read_csv('C:\Users\Mat\Desktop\Performance\340mus_110522.csv',usecols=[1])
#df23=pd.read_csv('C:\Users\Mat\Desktop\Performance\340JD_050522.csv',usecols=[1])
#df24=pd.read_csv('C:\Users\Mat\Desktop\Performance\340dicklovett_290422.csv',usecols=[1])
#df25=pd.read_csv('340_0MEA.csv',usecols=[1])
#df26=pd.read_csv('C:\Users\Mat\Desktop\Performance\340premier_031022.csv',usecols=[1])
#df27=pd.read_csv('340_IMU.csv',usecols=[1])
#df28=pd.read_csv('340R.csv',usecols=[1])
#df29=pd.read_csv('340M.csv',usecols=[1])
#df30=pd.read_csv('340K.csv',usecols=[1])



#concat the dataframes into one and name the columns


df=pd.concat([df2,df3,df4,df5,df6,df7,df8,df9],axis=1)
#df=pd.concat([df8,df10,df11,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df26,df30],axis=1)



columns=(["10","20","30","40", "50","60","70","80"])

#columns=(["autoshine","R114","MEA","swanley","sitIBC","sitTANK","bavarian","premier211022","premier_residue","myalloys","mus1","mus2","mus3","JD","dick","premier","340K"])


#fig=px.line(df,y=df,x=df1)



#normalise the data

df_normalized=(df - df.mean()) / df.std()
pca = PCA(n_components=df.shape[1])
pca.fit(df_normalized)

#pca=decomposition.PCA(n_components=4)
#pca.fit(df_normalized)
X=pca.transform(df_normalized)




loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
index=df.columns)


#plt.plot(pca.explained_variance_ratio_)
#plt.ylabel('Explained Variance')
#plt.xlabel('Components')
#plt.show()


#plt.scatter(loadings.loc["1090","PC1"],loadings.loc["1090","PC3"],loadings.loc["1090","PC2"],marker="*")
#plt.scatter(loadings.loc["2080","PC1"],loadings.loc["2080","PC3"],loadings.loc["2080","PC2"],marker="o")
#plt.scatter(loadings.loc["3070","PC1"],loadings.loc["3070","PC3"],loadings.loc["3070","PC2"],marker="1")
#plt.scatter(loadings.loc["4060","PC1"],loadings.loc["4060","PC3"],loadings.loc["4060","PC2"],marker=">")
#plt.scatter(loadings.loc["3070","PC1"],loadings.loc["3070","PC3"],loadings.loc["3070","PC2"],marker="2")
#plt.scatter(loadings.loc["8020","PC1"],loadings.loc["8020","PC3"],loadings.loc["8020","PC2"],marker="3")
#plt.scatter(loadings.loc["autoshine","PC1"],loadings.loc["autoshine","PC3"],loadings.loc["autoshine","PC2"],marker="8")
#plt.scatter(loadings.loc["water","PC1"],loadings.loc["water","PC3"],loadings.loc["water","PC2"],marker="x")
#plt.scatter(loadings.loc["MEA","PC1"],loadings.loc["MEA","PC3"],loadings.loc["MEA","PC2"],marker="d")
#plt.scatter(loadings.loc["waterMEA5050","PC1"],loadings.loc["waterMEA5050","PC3"],loadings.loc["waterMEA5050","PC2"],marker="D")
#plt.scatter(loadings.loc["swanley","PC1"],loadings.loc["swanley","PC3"],loadings.loc["swanley","PC2"],marker="_")
#plt.scatter(loadings.loc["sitIBC","PC1"],loadings.loc["sitIBC","PC3"],loadings.loc["sitIBC","PC2"],marker="|")
#plt.scatter(loadings.loc["sitTANK","PC1"],loadings.loc["sitTANK","PC3"],loadings.loc["sitTANK","PC2"],marker="H")
#plt.scatter(loadings.loc["bavarian","PC1"],loadings.loc["bavarian","PC3"],loadings.loc["bavarian","PC2"],marker="p")


#plt.legend(["1090","2080","3070","4060","3070","8020","autoshine","water","MEA","waterMEA5050","swanley","sitIBC","sitTANK","bavarian"])

#above only plots 2d principal components, use below for 3

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(loadings.loc["1090","PC1"],loadings.loc["1090","PC3"],loadings.loc["1090","PC2"],marker="*")
#ax.scatter(loadings.loc["2080","PC1"],loadings.loc["2080","PC3"],loadings.loc["2080","PC2"],marker="o")
#ax.scatter(loadings.loc["3070","PC1"],loadings.loc["3070","PC3"],loadings.loc["3070","PC2"],marker="1")
#ax.scatter(loadings.loc["4060","PC1"],loadings.loc["4060","PC3"],loadings.loc["4060","PC2"],marker=">")
#ax.scatter(loadings.loc["3070","PC1"],loadings.loc["3070","PC3"],loadings.loc["3070","PC2"],marker="2")
#ax.scatter(loadings.loc["8020","PC1"],loadings.loc["8020","PC3"],loadings.loc["8020","PC2"],marker="3")
#ax.scatter(loadings.loc["autoshine","PC1"],loadings.loc["autoshine","PC3"],loadings.loc["autoshine","PC2"],marker="8")
#ax.scatter(loadings.loc["water","PC1"],loadings.loc["water","PC3"],loadings.loc["water","PC2"],marker="x")
#ax.scatter(loadings.loc["MEA","PC1"],loadings.loc["MEA","PC3"],loadings.loc["MEA","PC2"],marker="d")
#ax.scatter(loadings.loc["waterMEA5050","PC1"],loadings.loc["waterMEA5050","PC3"],loadings.loc["waterMEA5050","PC2"],marker="D")
#ax.scatter(loadings.loc["swanley","PC1"],loadings.loc["swanley","PC3"],loadings.loc["swanley","PC2"],marker="_")
#ax.scatter(loadings.loc["sitIBC","PC1"],loadings.loc["sitIBC","PC3"],loadings.loc["sitIBC","PC2"],marker="|")
#ax.scatter(loadings.loc["sitTANK","PC1"],loadings.loc["sitTANK","PC3"],loadings.loc["sitTANK","PC2"],marker="H")
#ax.scatter(loadings.loc["bavarian","PC1"],loadings.loc["bavarian","PC3"],loadings.loc["bavarian","PC2"],marker="p")

#plt.legend(["1090","2080","3070","4060","3070","8020","autoshine","water","MEA","waterMEA5050","swanley","sitIBC","sitTANK","bavarian"])


#ax.set_xlabel("PC1")
#ax.set_ylabel("PC2")
#ax.set_zlabel("PC3")


#plt.show()


#use plotly


#columns = (["1090","2080","3070","4060","3070","8020","autoshine","water","MEA","waterMEA5050","swanley","sitIBC","sitTANK","bavarian"])

loadings['columns'] = columns

fig=px.scatter(loadings,x="PC0",y="PC2",color="columns")

#fig =px.scatter_3d(loadings,x="PC1",y="PC2",z="PC3",color='columns')

fig.show()

