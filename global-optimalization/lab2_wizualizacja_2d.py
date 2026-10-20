import math, random
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = open('/home/dawid/Projects/University/global-optimalization/lab2.py').read().split('zmienne_3d')[0]
exec(src)
random.seed(7)
data=[{'a':-1,'b':1,'d':6},{'a':-1,'b':1,'d':6}]
N=2000
# odtworzenie kroków calc, żeby mieć całą populację
L=[(v['b']-v['a'])*10**v['d']+1 for v in data]; ms=[find_m(l) for l in L]
bins=[get_all_binary(m,N) for m in ms]; dec=[get_decimals(b) for b in bins]
sc=[get_scale(dl,v['a'],v['b'],m,v['d']) for dl,v,m in zip(dec,data,ms)]
pts=list(zip(*sc)); vals=[func_nd(*p) for p in pts]
best=pts[int(np.argmin(vals))]; fb=min(vals)
print('m',ms,'best',best,fb)
X,Y=np.meshgrid(np.linspace(-1,1,801),np.linspace(-1,1,801))
Z=20+X**2-10*np.cos(20*np.pi*X)+Y**2-10*np.cos(20*np.pi*Y)
fig,axs=plt.subplots(1,2,figsize=(13,6.2),dpi=150)
for ax,lim,ms_ in [(axs[0],1,6),(axs[1],0.35,14)]:
    cf=ax.contourf(X,Y,Z,levels=30,cmap='Greys_r',alpha=0.9)
    p=np.array(pts); mask=(abs(p[:,0])<=lim)&(abs(p[:,1])<=lim)
    ax.scatter(p[mask,0],p[mask,1],s=ms_,c='#e8a33a',edgecolors='none',alpha=0.85,label=f'populacja (N = {N})')
    ax.scatter(*best,s=220,marker='*',c='#1f6fd1',edgecolors='white',linewidths=1.2,zorder=5,label=f'najlepszy punkt ({best[0]:.4f}, {best[1]:.4f}), f = {fb:.4f}')
    ax.scatter(0,0,s=110,marker='X',c='#c0392b',edgecolors='white',linewidths=1,zorder=6,label='minimum globalne (0, 0), f = 0')
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.set_xlabel('x1'); ax.set_ylabel('x2')
axs[0].set_title('Cała dziedzina [-1, 1]²'); axs[1].set_title('Przybliżenie [-0,35; 0,35]²')
fig.colorbar(cf,ax=axs,shrink=0.8,label='f(x1, x2)')
axs[0].legend(loc='upper center',bbox_to_anchor=(1.1,-0.1),ncol=3,fontsize=9,frameon=False)
fig.suptitle('Losowe przeszukiwanie dla n = 2, d = 6 (m = 21 bitów na zmienną)',fontsize=13)
fig.savefig('wizualizacja_2d.png',bbox_inches='tight')
# powierzchnia 3D
fig=plt.figure(figsize=(9,6.5),dpi=150); ax=fig.add_subplot(projection='3d')
Xs,Ys=np.meshgrid(np.linspace(-1,1,301),np.linspace(-1,1,301))
Zs=20+Xs**2-10*np.cos(20*np.pi*Xs)+Ys**2-10*np.cos(20*np.pi*Ys)
ax.plot_surface(Xs,Ys,Zs,cmap='viridis',linewidth=0,rstride=1,cstride=1,antialiased=True)
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('f'); ax.view_init(45,-60)
ax.set_title('f(x1, x2) = 20 + Σ (x² − 10 cos(20πx)) na [-1, 1]²')
fig.savefig('powierzchnia_3d.png',bbox_inches='tight')
# zbieżność dla n=2: najlepsze f vs N
for N in [10,100,1000,10000,100000]:
    r=[]
    for s in range(20):
        random.seed(s); p=calc(data,N,func_nd); r.append(func_nd(*p))
    print(N, round(np.mean(r),4), round(min(r),4), round(max(r),4))
