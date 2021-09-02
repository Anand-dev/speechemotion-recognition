import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import librosa
import librosa.display
import soundfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)


model=pickle.load(open('model_pickle','rb')) #loads our ML model using pickle file 

def extract_feature(file_name, mfcc, chroma, mel):   #funtion to extract audio features
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


#st.sidebar.title("sidebar") 
st.title("Speech Emotion Recognition")
st.write("Apllication using ML/AI to recognize different emotion form  a soundfile")

#uploading out audio file 
file_audio= st.file_uploader(label="", type=".wav")

#file directory to save the file 
file_dir="./out/file.wav"


if file_audio is not None: #cheking for audio file validity 
   #sucess message 
   st.success("File uploaded")
   #playback audio file
   st.audio(file_audio) 
   
   sound= AudioSegment.from_file(file_audio)
   #saving file
   sound.export(file_dir,format='wav')  

   submit= st.button("Predit Emotion")

   if submit: 
       #plotting the wave curve of the audio  
      data,srt=librosa.load(file_dir)
      plt.figure(figsize=(12, 4))
      plt.plot(data)
      st.pyplot()

        
      test_f=extract_feature(file_dir, mfcc=True, chroma=True, mel=True)
      test_f=test_f.reshape(1,-1)
        #making model prediction
      result=model.predict(test_f)
      
      #displaying result
      st.header("Emotion of the audio is " + result[0].upper())
        #removing the file
      os.remove(file_dir)