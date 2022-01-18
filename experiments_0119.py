import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import csv

from scipy.signal import argrelmin, argrelmax
from scipy import interpolate
from sympy import Symbol

from multiprocessing import Process
import time

import threading

import speech2text_20211102_new
import sound_cut
import short_speech

import pandas as pd

import pyaudio  #録音機能を使うためのライブラリ
import wave     #wavファイルを扱うためのライブラリ

import sys
sys.path.append('c:\python39\lib\site-packages')
from playsound import playsound

count=0
name_list=['yama','okami','kodama','test','ueda','kanda','abe','akiyoshi','tainaka','asada','okami','tainaka']
sentence_list=["sentence{}".format(i) for i in np.arange(1,3)]
count_list=["count{}".format(i) for i in np.arange(1,4)]
stroke_speed_list=["slow","fast"]

df=pd.read_csv('./csv/experiment.csv',encoding='SHIFT-JIS')

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fps=30
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）

video_list=list() 

for i in name_list:
    for ii in stroke_speed_list:
        for iii in count_list:
            for iiii in sentence_list:
                video_list.append("{}_{}_{}_{}".format(i,ii,iii,iiii))  

video=cv2.VideoWriter("movie/{}.mp4".format(video_list[count]), fourcc, fps, (w, h)) # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

def mediapipe_detection(frame):

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    results=hands.process(frame_rgb)
    ii=results.multi_hand_landmarks
    if ii:
        for i in ii:
            mpDraw.draw_landmarks(frame,i,mpHands.HAND_CONNECTIONS)

    gray_new=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    global threshold
    ret_,threshold=cv2.threshold(gray_new,0,255,cv2.THRESH_OTSU)

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

def recording_sound(name,ind,all_start):
    
    RECORD_SECONDS = 100 #録音する時間の長さ（秒）
    WAVE_OUTPUT_FILENAME = "./segmentation-kit/wav/{}.wav".format(name[ind]) #音声を保存するファイル名
    iDeviceIndex = 0 #録音デバイスのインデックス番号
 
    #基本情報の設定
    FORMAT = pyaudio.paInt16 #音声のフォーマット
    CHANNELS = 1             #モノラル
    RATE = 16000           #サンプルレート
    
    CHUNK = 2**11            #データ点数
    audio = pyaudio.PyAudio() #pyaudio.PyAudio()
 
    stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index = iDeviceIndex, #録音デバイスのインデックス番号
        frames_per_buffer=CHUNK)

    start_video=time.time()
    print('sound_{}'.format(start_video-all_start))

    time.sleep(10-(start_video-all_start))
    print('sound_{}'.format(time.time()-all_start))


    #--------------録音開始---------------

    print ("sound recording...")
    frames = []
    try:
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            #print('--------------------------------')

    except :
        print ("finished sound recording")
            
        #--------------録音終了---------------
    
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
def recording_movie(video,count):

    center_list=list()
    cnt__=0

    try:    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
        while True:
            ret, frame = cap.read()  
            mediapipe_detection(frame)    
            lower_filter,higher_filter=np.array([224,220,224]),np.array([224,227,224])
            #lower_filter,higher_filter=np.array([0,254,0]),np.array([0,255,0])
            
            inRange_mask=cv2.inRange(frame,lower_filter,higher_filter)
            bitwise=cv2.bitwise_and(frame,frame,mask=inRange_mask)
            bitwise_gray=cv2.cvtColor(bitwise,cv2.COLOR_BGR2GRAY)
            ret__,bitwise_threshold=cv2.threshold(bitwise_gray,0,255,cv2.THRESH_OTSU)
            center=get_center(bitwise_threshold)
            if center!=None:
                center_list.append(center)
                if len(center_list)==1:
                    speed_realtime=0
                
                else:
                    speed_realtime=abs(center_list[-1][0]-center_list[-2][0])

                if cnt__%6==0:
                    print('stroke_speed={}[cm/s]'.format((speed_realtime/(1/fps))/54))
                
                cv2.circle(bitwise,center,0,[150,255,30],15,200)
                

            video.write(frame)

            cv2.imshow('frame',bitwise)
            key_=cv2.waitKey(1) 
            cnt__+=1

    except:
            
        global x,center_new_list

        center_new_list=[i[0] for i in center_list]
        x=np.arange(0,len(center_new_list)*(1/fps),1/fps)

        speed___avg=np.sum(abs(np.diff(center_new_list)))/((len(center_new_list)-1)*(1/fps))
            
        global df

        inddd=list(df[df.iloc[:,0]==name_list[0]].index)
        df.iloc[inddd[count],4]=speed___avg/51
            
        if count<6:
            df.iloc[inddd[count],5]=50/(len(center_new_list)*(1/fps))
        else:
            df.iloc[inddd[count],5]=65/(len(center_new_list)*(1/fps))

        center_all_list.append(center_new_list)
        #print(center_all_list)
        center_index_l=[ind__ for ind__,kl in enumerate(center_new_list) if ind__%10 ==0]

        f_in = interpolate.Akima1DInterpolator(x[center_index_l], np.array(center_new_list)[center_index_l])
        f_in_center=f_in(x)

        max_index=argrelmax(np.array(f_in_center))
        min_index=argrelmin(np.array(f_in_center))

        #fig=plt.figure()

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

        df.iloc[inddd[count],6]=np.sum(abs(np.diff(max_min_list)/51))/len(max_min_list)
        df.iloc[inddd[count],7]=len(max_min_list)

        print(center_new_list)
        print(abs(np.diff(center_new_list)))
        
        plt.grid()
        plt.xlabel('Time[sec]')
        plt.ylabel('center_placement')
        plt.title('center_placement_graph')
        #plt.tight_layout()
        plt.legend()
        

        fig.savefig("img_raw/{}_raw.png".format(video_list[count]))

        

if __name__ == '__main__':

    center_all_list=list()
    try:
        dssdds=pd.read_csv('csv/hand_{}.csv'.format(name_list[0]))
        for huihui in range(count):
            center_all_list__=list()
            for indexex,row_ in enumerate(dssdds.iloc[:,huihui]):
                if indexex>0:
                    center_all_list__.append(row_)
            center_all_list.append(center_all_list__)
    except:
        None


    with open("csv/hand_{}.csv".format(name_list[0]),"w",newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['---'+str(i_fghj)+'---' for i_fghj in np.arange(1,13) ])
        
        
        while True:

            ret,frame=cap.read()
            mediapipe_detection(frame)
            cv2.imshow('frame',frame)
            key=cv2.waitKey(1)
                            
            if key==27:
                center_all_new_list=list()
                for iuy in center_all_list:
                    if len(iuy) < np.max([len(ser) for ser in center_all_list]):
                        center_all_new_list.append(iuy+list(np.zeros(np.max([len(tgb) for tgb in center_all_list])-len(iuy))))
                        
                    else:
                        center_all_new_list.append(iuy)
                
                writer.writerows(np.array(center_all_new_list).T)
                        
                df.to_csv('./csv/experiment.csv',encoding='SHIFT-JIS',index=False)

                break

            elif key==ord('r'):
                            
                all_start=time.time()
                    
                video=cv2.VideoWriter("movie/{}.mp4".format(video_list[count]), fourcc, fps, (w, h)) # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
                            
                fig=plt.figure()
                p1=Process(target=recording_sound, args=(video_list,count,all_start))
                p1.start()

                print('start_movie={}'.format(time.time()-all_start))

                time.sleep(1-(time.time()-all_start))

                for jjjh in range(9):
                    print(9-jjjh)
                    time.sleep(1)

                print('start_movie_new={}'.format(time.time()-all_start))

                recording_movie(video,count)
                            
                plt.show()
                fig.savefig("img/{}.png".format(video_list[count]))
                count+=1

            elif key==ord('u'):
                playsound(r"C:\Users\imdam\0321.wav")
                    
                    



