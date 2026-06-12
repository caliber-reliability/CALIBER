# -*- coding: utf-8 -*-
"""Figure 1 - the six-layer architecture of CALIBER.
Single-column width (~85 mm), 10-pt unified Times-compatible font, no outer
border, per IJIES format. Run: python figure1_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 10})

NAVY="#1f3864"; BLUE="#dbe5f1"; BLUEDGE="#aebfda"; AMBER="#fce8cc"; AMBEDGE="#e0a232"
SUB="#ffffff"; SUBEDGE="#e3c08f"; DARK="#34527e"; SUBTXT="#33415c"; DET="#6a7690"
LX,LW=3,94

fig=plt.figure(figsize=(85/25.4, 116/25.4))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.axis("off")

def rbox(x,y,w,h,fc,ec,r=2.0,lw=1.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,mutation_aspect=0.7))
def arr(y0,y1):
    ax.add_patch(FancyArrowPatch((50,y0),(50,y1),arrowstyle="-|>",mutation_scale=9,lw=1.2,color="#7e8db0",shrinkA=0,shrinkB=0))

H=16.5; HR=31.0; GAP=3.2; TOP=132.0
y=TOP
ax.text(50,y+5.0,"The CALIBER framework",ha="center",va="center",fontsize=11,fontweight="bold",color=NAVY)

def normlayer(y,title,sub,fc=BLUE,ec=BLUEDGE,tc=NAVY,sc=SUBTXT,white=False):
    rbox(LX,y-H,LW,H,fc,ec)
    ax.text(50,y-4.6,title,ha="center",va="center",fontsize=10,fontweight="bold",color=("white" if white else tc))
    ax.text(50,y-8.0,sub,ha="center",va="top",fontsize=10,color=("#e8eef7" if white else sc),linespacing=1.12)
    return y-H

y=normlayer(y,"Presentation layer","User interface and API ·\nprediction with reliability statement"); arr(y,y-GAP); y-=GAP
y=normlayer(y,"Analytics layer","Permutation feature importance ·\noperational analysis"); arr(y,y-GAP); y-=GAP

rbox(LX,y-HR,LW,HR,AMBER,AMBEDGE,r=2.4,lw=1.3)
ax.text(50,y-4.4,"Reliability layer",ha="center",va="center",fontsize=10,fontweight="bold",color="#9c5a00")
rbox(LX+4,y-18.5,LW-8,11.0,SUB,SUBEDGE,r=1.7,lw=1.0)
ax.text(50,y-11.0,"Conformal prediction intervals",ha="center",va="center",fontsize=10,color=SUBTXT)
ax.text(50,y-15.4,"(split / normalized / Mondrian)",ha="center",va="center",fontsize=10,color=DET)
rbox(LX+4,y-30.5,LW-8,11.0,SUB,SUBEDGE,r=1.7,lw=1.0)
ax.text(50,y-23.0,"Domain-of-applicability detector",ha="center",va="center",fontsize=10,color=SUBTXT)
ax.text(50,y-27.4,"(kNN + isolation forest)",ha="center",va="center",fontsize=10,color=DET)
y-=HR; arr(y,y-GAP); y-=GAP

y=normlayer(y,"Predictive core","Histogram gradient-boosting regressor\n(+ 4 reference models)"); arr(y,y-GAP); y-=GAP
y=normlayer(y,"Feature-engineering layer","Leakage-free composition-grouped\nsplitting and input preparation"); arr(y,y-GAP); y-=GAP
y=normlayer(y,"Data layer","Ingestion · schema and range\nvalidation · storage",fc=DARK,ec=DARK,white=True)

ax.set_ylim(y-2,TOP+9)
fig.savefig("figure1.png",dpi=400,facecolor="white")
print("wrote figure1.png")
