import sys
sys.path.append("c:\python39\lib\site-packages")


from gtts import gTTS 
from playsound import playsound
import cv2
import librosa

from pydub import AudioSegment
from pydub.playback import play



myText = "やはりショパンの英雄ポロネーズは美しいですね。何度でも聴きたくなる。"
language ='ja'
output = gTTS(text=myText, lang=language, slow=False)
output.save("output.mp3")

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret_,threshold=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    cv2.imshow('frame',threshold)
    key=cv2.waitKey(1)

    if key==27:
        playsound("output.mp3")
        #sound = AudioSegment.from_file("output.wav", format="wav")
        #play(sound)
        break


#playsound('output.mp3')