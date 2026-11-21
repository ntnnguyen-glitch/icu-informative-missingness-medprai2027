import os
ROOT=os.environ.get('P12_ROOT','.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'figure.facecolor':'white','axes.facecolor':'white',
 'savefig.facecolor':'white','savefig.dpi':600,'lines.linewidth':1.2})
C={'R0':'#4D4D4D','R1':'#2166AC','R2':'#B2182B','R3':'#E08214','MDA':'#1B7837'}
LAB={'R0':'R0: values only','R1':'R1: values + indicators','R2':'R2: values + process features','R3':'R3: process features only'}
