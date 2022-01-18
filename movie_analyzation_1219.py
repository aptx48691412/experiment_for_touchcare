import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.signal import argrelmin, argrelmax
from scipy import interpolate
from sympy import Symbol

fps=30
name_list=['akiyoshi','abe','asada','ueda','okami','kanda','kodama'
#'test','ueda','kanda','abe','akiyoshi','tainaka','asada','okami','tainaka'
]

video_list=list()
sentence_list=["sentence{}".format(i) for i in np.arange(1,3)]
count_list=["count{}".format(i) for i in np.arange(1,4)]
stroke_speed_list=["slow","fast"]

for i in name_list:
    for ii in stroke_speed_list:
        for iii in count_list:
            for iiii in sentence_list:
                video_list.append("{}_{}_{}_{}".format(i,ii,iii,iiii))  


def detect_green_color(frame):
    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    low_filter=np.array([30, 64, 0])
    high_filter=np.array([90,255,255])
    mask=cv2.inRange(hsv_frame,low_filter,high_filter)
    bitwise_frame=cv2.bitwise_and(frame,frame,mask=mask)
   

    return bitwise_frame


def get_center(img):
    try:
        y,x=np.where(img==255)
        y_avg,x_avg=np.average(y),np.average(x)
    
        return [int(x_avg),int(y_avg)]

    except:
        try:
            return [int(x_avg),int(y_avg)]
        except:
            None




#df=pd.read_csv(f'./csv/{name_list[0]}.csv')
df=pd.read_csv('./csv/speed__1220.csv')

for i in range(12*len(name_list)):
    movie=r'C:\Users\imdam\exp_1117\movie\{}.mp4'.format(video_list[i])    
    cap=cv2.VideoCapture(movie)
    center_list=list()
    df.iloc[i,0]=name_list[int(i/12)]

    try:
        while True:
            ret,frame=cap.read()
            bitwise_frame=detect_green_color(frame)
            gray_frame=cv2.cvtColor(bitwise_frame, cv2.COLOR_BGR2GRAY)
            ret_,threshold=cv2.threshold(gray_frame, 0,255,cv2.THRESH_OTSU)
            center=get_center(threshold)
            if center:
                center_list.append(center[0])
            cv2.circle(frame, center, 10, [0,255,255],3,10)
            cv2.imshow('frame',frame) 
            #print(ret)

            key=cv2.waitKey(1)
            if key==27:
                break
        

    except:
        pass

    try:
        print(i)
        fig=plt.figure(dpi=200)
        x=np.arange(0,len(center_list)/fps,1/fps)
        center_index_list=[index for index,ii in enumerate(center_list) if index%10==0]
        f_in = interpolate.Akima1DInterpolator(x[center_index_list], np.array(center_list)[center_index_list])
        f_in_center=f_in(x)
        max_index=argrelmax(np.array(f_in_center))
        min_index=argrelmin(np.array(f_in_center))
        

        max_min_index_list=np.sort(np.array(list(max_index[0])+list(min_index[0])))
        idx_list=list()
        for inin in range(len(max_min_index_list)):
            if inin==0:
                sa=abs(max_min_index_list[inin]-max_min_index_list[inin+1])
                idx_list.append(max_min_index_list[inin]+int(sa/4))
            elif inin==len(max_min_index_list)-1:
                sa_=(max_min_index_list[inin]-max_min_index_list[inin-1])
                idx_list.append(max_min_index_list[inin]-int(sa_/4))
            else:
                sa__=max_min_index_list[inin]-max_min_index_list[inin-1]
                sa___=abs(max_min_index_list[inin]-max_min_index_list[inin+1])
                
                idx_list.append(max_min_index_list[inin]-int(sa__/4))
                idx_list.append(max_min_index_list[inin]+int(sa___/4))

        #print(abs(np.diff(np.array(center_list)[max_min_index_list])))
        print(idx_list)
        print(max_min_index_list)

        plt.plot(x,np.array(center_list[:len(x)])/51,label='center',linewidth=3,linestyle=':')
        #plt.plot(x,np.array(f_in_center[:len(x)])/51,'y--', label='scipy')
        plt.plot(x[max_index[0]], np.array(center_list[:len(x)])[max_index[0]]/51,'ro')
        plt.plot(x[min_index[0]], np.array(center_list[:len(x)])[min_index[0]]/51, 'bo')
        speed_ave_list=list()

        for ijij in np.arange(0,len(idx_list),2):
            try:
                plt.plot(x[idx_list[ijij]:idx_list[ijij+1]+1],np.array(center_list)[idx_list[ijij]:idx_list[ijij+1]+1]/51,linewidth=2)
                
                speed_ave_list.append(abs(center_list[idx_list[ijij]]-center_list[idx_list[ijij+1]])/51/(idx_list[ijij+1]-idx_list[ijij])*30)
                print('uyfkgkvj')
                print(idx_list[ijij+1]-idx_list[ijij])

            except:
                import traceback
                traceback.print_exc()


        df.iloc[i,1]=np.average(speed_ave_list)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('placement_of_center')
        plt.title('placement_of_center')
        plt.legend()
        fig.savefig(f'./1219_speed/{video_list[i]}.png')
        #plt.show()
        



    except:
        None

#df.iloc[14,0]
df.to_csv('./csv/speed__1220.csv',header=False,index=False)