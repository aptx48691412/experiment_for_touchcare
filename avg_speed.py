import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import argrelmin, argrelmax
from scipy import interpolate
from sympy import Symbol


name_list=['akiyoshi','abe','kanda','kodama','ueda']
fps=30

df=pd.read_csv(r'C:\Users\imdam\exp_1117\csv\experiment_1117.csv',encoding='SHIFT-JIS')
df_=pd.read_csv(r'C:\Users\imdam\exp_1117\csv\hand_{}_1117.csv'.format(name_list[0]))

inddd=list(df[df.iloc[:,0]==name_list[0]].index)

for i in range(12):
    try:
        pre_center_new_list=df_.iloc[:,i]

        center_new_list=[kk for kk in pre_center_new_list if not kk==0]
        x=np.arange(0,len(center_new_list)*(1/fps),1/fps)
        speed___avg=np.sum(abs(np.diff(center_new_list)))/((len(center_new_list)-1)*(1/fps))
                
        
        df.iloc[inddd[i],4]=speed___avg/51
                
        if i<6:
            df.iloc[inddd[i],5]=50/(len(center_new_list)*(1/fps))
        else:
            df.iloc[inddd[i],5]=65/(len(center_new_list)*(1/fps))

        center_index_l=[ind__ for ind__,kl in enumerate(center_new_list) if ind__%10 ==0]

        f_in = interpolate.Akima1DInterpolator(x[center_index_l], np.array(center_new_list)[center_index_l])
        f_in_center=f_in(x)

        max_index=argrelmax(np.array(f_in_center))
        min_index=argrelmin(np.array(f_in_center))

        plt.text(0,0,'speed_avg={}[cm/s]'.format(speed___avg/51))

        plt.plot(x,np.array(center_new_list[:len(x)])/51,label='center')
        plt.plot(x,np.array(f_in_center[:len(x)])/51,'y--', label='scipy')
        plt.plot(x[max_index[0]], np.array(center_new_list[:len(x)])[max_index[0]]/51,'ro')
        plt.plot(x[min_index[0]], np.array(center_new_list[:len(x)])[min_index[0]]/51, 'bo')

        max_list=[[iu,iuu] for iu,iuu in zip(x[max_index[0]],np.array(center_new_list[:len(x)])[max_index[0]])]
        min_list=[[iu_,iuu_] for iu_,iuu_ in zip(x[min_index[0]],np.array(center_new_list[:len(x)])[min_index[0]])]
        max_min_list=list()
        
        for iiij,[ij,kj] in enumerate(zip(max_list,min_list)):
            if ij[0]<kj[0]:
                max_min_list+=[ij[1],kj[1]]
            else:
                max_min_list+=[kj[1],ij[1]]
        
        if len(max_list)>len(min_list):
            max_min_list+=[max_list[-1][1]]

        elif len(max_list)<len(min_list):
            max_min_list+=[min_list[-1][1]]

        else:
            None


        print(np.array(max_min_list)/51)
        print(abs(np.diff(max_min_list)/51))


        df.iloc[inddd[i],6]=np.sum(abs(np.diff(max_min_list)/51))/len(max_min_list)

        df.iloc[inddd[i],7]=len(max_min_list)

        print(center_new_list)
        print(abs(np.diff(center_new_list)))
                
        plt.grid()
        plt.xlabel('Time[sec]')
        plt.ylabel('center_placement')
        plt.title('center_placement_graph')
        plt.tight_layout()
        plt.legend()
        plt.show()

    except:
        None


df.to_csv(r'C:\Users\imdam\exp_1117\csv\experiment_1117.csv',encoding='SHIFT-JIS',index=False)