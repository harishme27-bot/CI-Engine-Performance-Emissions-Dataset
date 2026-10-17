import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_excel("44_ML_Updated.xlsx"); df.columns=df.columns.str.strip()
df=df.rename(columns={"Injection Timing (deg bTDC)":"Timing","Load (%)":"Load","BTE (%)":"BTE","BSFC (kg/kWh)":"BSFC","CO (%)":"CO","HC (ppm)":"HC","NOx (ppm)":"NOx"})
df["BSFC"]*=1000
F=["Diesel","MESO-B20","MEWVO-B20"]; C={"Diesel":"#555555","MESO-B20":"#1f5aa6","MEWVO-B20":"#e08e2b"}
LAB={"BTE":"BTE (%)","BSFC":"BSFC (g/kWh)","CO":"CO (% vol)","HC":"HC (ppm)","NOx":"NOx (ppm)"}
plt.rcParams.update({"font.size":9})
fig,axs=plt.subplots(2,3,figsize=(12,6.4))
for ax,t in zip(axs.flat,["BTE","BSFC","CO","HC","NOx"]):
    bins=np.histogram_bin_edges(df[t],bins=8)
    ax.hist([df.loc[df.Fuel==f,t] for f in F],bins=bins,stacked=True,color=[C[f] for f in F],label=F,edgecolor="white")
    ax2=ax.twinx(); ax2.boxplot(df[t],vert=False,positions=[0.5],widths=0.25,patch_artist=True,
        boxprops=dict(facecolor="none"),flierprops=dict(marker="o",mfc="red",ms=5)); ax2.set_ylim(0,6); ax2.set_yticks([])
    ax.set_xlabel(LAB[t]); ax.set_ylabel("Number of conditions"); ax.set_title(f"{t}  (skewness = {df[t].skew():.2f})"); ax.grid(alpha=.3)
axs[0,0].legend(fontsize=8)
ax=axs[1,2]; ct=pd.crosstab(df.Timing,df.Fuel).reindex(columns=F,fill_value=0); b=np.zeros(len(ct))
for f in F: ax.bar(ct.index.astype(str),ct[f],bottom=b,color=C[f],label=f,edgecolor="white"); b+=ct[f].values
for x,v in zip(ct.index.astype(str),b): ax.text(x,v+0.2,int(v),ha="center")
ax.set_xlabel("Injection timing (° bTDC)"); ax.set_ylabel("Number of conditions"); ax.set_title("Design counts (each bar = 4 loads per fuel)"); ax.set_ylim(0,14); ax.grid(axis="y",alpha=.3)
plt.tight_layout(); plt.savefig("Fig_distributions.png",dpi=300)
