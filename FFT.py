# -*- coding:utf-8 -*-
import pyaudio
import numpy as np
import wave
import struct

RATE=44100
p=pyaudio.PyAudio()
N=100
CHUNK=1024*N
r= 1.059463094
r12=r*r*r*r*r*r
sk=0

stream=p.open(  format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする
stream1=p.open( format = pyaudio.paInt16,
        channels = 1,
        rate = int(RATE*r12),
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

def write_wave(i,sin_wave,fs,sig_name='sample'):#fs:サンプリング周波数
    sin_wave = [int(x * 32767.0) for x in sin_wave]    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    wav_file='./'+str(i)+'_'+sig_name+'.wav'
    w = wave.Wave_write(wav_file)
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    return wav_file

while stream.is_active():
    input = stream.read(CHUNK)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    write_wave(sk, sig, fs=RATE*r12, sig_name='-4x')
    write_wave(sk, sig, fs=RATE/r12, sig_name='4x')
    write_wave(sk, sig, fs=RATE, sig_name='original')
    output = stream1.write(input)
    sk += 1