from __future__ import annotations
import matplotlib.pyplot as plt

def draw_fully_connected(n_in,n_h,n_out,path:str):
    fig,ax=plt.subplots(figsize=(8,5))
    x_in,x_h,x_out=0.1,0.5,0.9
    y=lambda n:[0.1+i*(0.8/max(n-1,1)) for i in range(n)]
    y_in,y_h,y_out=y(n_in),y(n_h),y(n_out)
    for yi in y_in: ax.scatter(x_in,yi,s=100,edgecolors='k')
    for yh in y_h: ax.scatter(x_h,yh,s=100,edgecolors='k')
    for yo in y_out: ax.scatter(x_out,yo,s=100,edgecolors='k')
    for yi in y_in: 
        for yh in y_h: ax.plot([x_in,x_h],[yi,yh],lw=0.3)
    for yh in y_h: 
        for yo in y_out: ax.plot([x_h,x_out],[yh,yo],lw=0.3)
    ax.text(x_in,0.95,"Input",ha="center");ax.text(x_h,0.95,"Hidden",ha="center")
    ax.text(x_out,0.95,"Output",ha="center")
    ax.axis("off");fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)
