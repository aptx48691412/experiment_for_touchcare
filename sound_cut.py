import librosa
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin')

from inaSpeechSegmenter import Segmenter,seg2csv

from pydub import AudioSegment
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\imdam\Downloads\ffmpeg-2021-11-03-git-08a501946f-essentials_build\bin')

def main(input_file):
    # 入力のwavファイルのパスを指定
    #input_file = './kishi.wav'

    # loadメソッドでy=音声信号の値（audio time series）、sr=サンプリング周波数（sampling rate）を取得
    # 参考：https://librosa.org/doc/latest/generated/librosa.load.html?highlight=load#librosa.load
    y, sr = librosa.load(input_file)
    # 時間 = yのデータ数 / サンプリング周波数
    # 参考：https://note.nkmk.me/python-numpy-arange-linspace/
    time = np.arange(0,len(y)) / sr

    # xにtime、yにyとしてプロット
    plt.plot(time, y)
    # x軸とy軸にラベルを設定（x軸は時間、y軸は振幅）
    # 参考：https://techacademy.jp/magazine/19316
    plt.xlabel("Time(s)")
    plt.ylabel("Sound Amplitude")

    # グラフを表示
    #plt.show()


    # 出力のwavファイルのフォルダとプレフィックスまで指定
    # → ./output/segment0.wav, ./output/segment1.wav, のような出力を想定
    #output_file = './output/segment.wav'

    # 'smn' は入力信号を音声区間(speeech)、音楽区間(music)、
    # ノイズ区間(noise)にラベル付けしてくれる
    # detect_genderをTrueにすると、音声区間は男性(male) / 女性(female)のラベルに
    # 細分化される
    seg = Segmenter(vad_engine='smn', detect_gender=False)

    # 区間検出実行（たったこれだけでOK）
    segmentation = seg(input_file)

    # ('区間ラベル',  区間開始時刻（秒）,  区間終了時刻（秒）)というタプルが
    # リスト化されているのが変数 segmentation
    # print(segmentation)

    # inaSpeechSegmenter単体では分割されたwavを作成してくれないので、
    # pydubのAudioSegmentにお世話になる (ありがたいライブラリ)
    speech_segment_index = 0

    speech_list=list()
    noEnergy_list=list()
    noise_list=list()



    yyy_list=list()
    for ijn in segmentation:
        if ijn[0]=='speech':
            yyy_list+=list(np.zeros(len(np.arange(ijn[1],ijn[2],0.1)))+600)
        elif ijn[0]=='noEnergy':
            yyy_list+=list(np.zeros(len(np.arange(ijn[1],ijn[2],0.1)))+0)
        else:
            yyy_list+=list(np.zeros(len(np.arange(ijn[1],ijn[2],0.1)))+0)

    plt.plot(np.linspace(0,segmentation[-1][2],len(yyy_list)),yyy_list,c='r')


    #print(speech_list,noEnergy_list,noise_list)
        
    #plt.show()

    with open('segtable.csv','w', newline='') as fff:
        writer=csv.writer(fff)
        writer.writerow(['labels','start','stop'])
        for ii in segmentation:
            writer.writerow(ii)


    #seg2csv(segmentation, 'myseg.csv')

    print(segmentation)
    print(segmentation[0])
    print(type(segmentation))
    print(type(segmentation[0]))



    for segment in segmentation:

        output_file = './output/segment'

        # segmentはタプル
        # タプルの第1要素が区間のラベル
        segment_label = segment[0]

        if (segment_label == 'speech'):  # 音声区間

            # 区間の開始時刻の単位を秒からミリ秒に変換
            start_time = segment[1] * 1000
            end_time = segment[2] * 1000

            # 分割結果をwavに出力
            newAudio = AudioSegment.from_wav(input_file)
            newAudio = newAudio[start_time:end_time]
            output_file = output_file + str(speech_segment_index) + '.wav'
            newAudio.export(output_file, format="wav")

            speech_segment_index += 1
            del newAudio

if __name__ == '__main__':
    main(input_file)
