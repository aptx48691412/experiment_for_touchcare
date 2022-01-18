import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import cv2
from scipy.signal import argrelmin, argrelmax
from scipy import interpolate
from sympy import Symbol

def get_center(img):

    try:
        y,x=np.where(img==255)
        x_avg,y_avg=np.average(x),np.average(y)

        return [int(x_avg),int(y_avg)]

    except:
        try:
            return [int(x_avg),int(y_avg)]
            #None
        
        except:
            None

def detect_green_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 緑色のHSVの値域1
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90,255,255])

    # 緑色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img

count=0
name_list=['kodama'
#,'akiyoshi','abe','kanda','kodama','ueda','okami','kodama','test','ueda','kanda','abe','akiyoshi','tainaka','asada','okami','tainaka'
]
sentence_list=["sentence{}".format(i) for i in np.arange(1,3)]
count_list=["count{}".format(i) for i in np.arange(1,4)]
stroke_speed_list=["slow","fast"]

video_list=list()
for i in name_list:
    for ii in stroke_speed_list:
        for iii in count_list:
            for iiii in sentence_list:
                video_list.append("{}_{}_{}_{}".format(i,ii,iii,iiii))  

df=pd.read_csv('./csv/experiment_1117.csv',encoding='SHIFT-JIS')
#df=pd.read_csv(r'C:\Users\imdam\exp_1117\csv\hand_{}_1117.csv'.format(name_list[0]),encoding='SHIFT-JIS')



#center_all_list=list()

#try:
 #   dssdds=pd.read_csv(r"C:\Users\imdam\exp_1117\csv\hand_{}_1117.csv".format(name_list[0]),header=None)
  #  for huihui in range(count+6,count+12):
   #     center_all_list__=list()
    #    for indexex,row_ in enumerate(dssdds.iloc[:,huihui]):
     #       if indexex>0:
      #          center_all_list__.append(row_)
       # center_all_list.append(np.array(center_all_list__))
#except:
 #   None



center_list_s=list()

with open(r"C:\Users\imdam\exp_1117\csv\hand_{}_1117.csv".format(name_list[0]),"w",newline='') as fgh:
    writer=csv.writer(fgh)
    writer.writerow(['---'+str(i_fghj)+'---' for i_fghj in np.arange(1,13) ])

    #cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
        
    for hjkjh in range(12):
                
        center_list=list()
        movie=r'C:\Users\imdam\exp_1117\movie\{}.mp4'.format(video_list[hjkjh])    
        cap=cv2.VideoCapture(movie)
        #fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        fps=30
        print('fps={}'.format(fps))

        while True:

            try:
                ret,frame=cap.read()
                masked_img=detect_green_color(frame)
                gray_img=cv2.cvtColor(masked_img,cv2.COLOR_RGB2GRAY)
                ret_,thresh_img=cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
                        
                center=get_center(thresh_img)
                if center:
                    center_list.append(center)

                cv2.circle(frame, center, 10, [255,0,0],10)
                print(center)
                        
                cv2.imshow('frame',frame)
                #cv2.imshow('frame',frame)
                        
                #print(frame.shape)
                #print(masked_img[0].shape)

                key=cv2.waitKey(1)

                if key==27:
                    break
                    
            except:
                #print(center_list)
                #print(np.array(center_list).T)
                #print(center_all_list)
                center_new_list=[trd[0] for trd in center_list]
                #center_all_list.append(center_new_list)
                #center_new_list.append(center_all_list)
                        
                center_list_s.append(np.array(center_new_list))
                #print(center_all_list)
                #print(np.array(center_all_list).T)

                try:
                    x=np.arange(0,len(center_list_s[-1])*(1/fps),1/fps)

                    speed___avg=np.sum(abs(np.diff(center_list_s[-1])))/((len(center_list_s[-1])-1)*(1/fps))
                
                    inddd=list(df[df.iloc[:,0]==name_list[0]].index)
                    df.iloc[inddd[hjkjh],4]=speed___avg/51
                        
                    if count<6:
                        df.iloc[inddd[hjkjh],5]=50/(len(center_list_s[-1])*(1/fps))
                    else:
                        df.iloc[inddd[hjkjh],5]=65/(len(center_list_s[-1])*(1/fps))

                    center_index_l=[ind__ for ind__,kl in enumerate(center_list_s[-1]) if ind__%10 ==0]

                    f_in = interpolate.Akima1DInterpolator(x[center_index_l], np.array(center_list_s[-1])[center_index_l])
                    f_in_center=f_in(x)

                    max_index=argrelmax(np.array(f_in_center))
                    min_index=argrelmin(np.array(f_in_center))

                    #fig=plt.figure()

                    plt.text(0,0,'speed_avg={}[cm/s]'.format(speed___avg/51))

                    

                    plt.plot(x,np.array(center_list_s[-1][:len(x)])/51,label='center')
                    plt.plot(x,np.array(f_in_center[:len(x)])/51,'y--', label='scipy')
                    plt.plot(x[max_index[0]], np.array(center_list_s[-1][:len(x)])[max_index[0]]/51,'ro')
                    plt.plot(x[min_index[0]], np.array(center_list_s[-1][:len(x)])[min_index[0]]/51, 'bo')
                    plt.show()

                    

                    max_list=[[iu,iuu] for iu,iuu in zip(x[max_index[0]],np.array(center_list_s[-1][:len(x)])[max_index[0]])]
                    min_list=[[iu_,iuu_] for iu_,iuu_ in zip(x[min_index[0]],np.array(center_list_s[-1][:len(x)])[min_index[0]])]
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


                    df.iloc[inddd[hjkjh],6]=np.sum(abs(np.diff(max_min_list)/51))/len(max_min_list)

                    df.iloc[inddd[hjkjh],7]=len(max_min_list)

                except:
                    None    

                break

    #for sdf in center_all_list:
        #center_list_s.append(sdf)
    df.to_csv('./csv/experiment_1117.csv',encoding='SHIFT-JIS',index=False)
    
    center_all_new_list=list()

    for iuy in center_list_s:
        if len(iuy) < np.max([len(ser) for ser in center_list_s]):
            center_all_new_list.append(np.array(list(iuy)+list(np.zeros(np.max([len(tgb) for tgb in center_list_s])-len(iuy)))))
                                
        else:
            center_all_new_list.append(iuy)
                                    
    
    writer.writerows(np.array(center_all_new_list).T)
                    
                    
                