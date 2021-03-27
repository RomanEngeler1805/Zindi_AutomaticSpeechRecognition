## read into memory (small) -> storing takes around 2hrs
## IMPORTANT: it's not persisted across sessions (why not?)
import os.path
import pandas as pd
import librosa as lb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Train.csv')
nsamples = len(df)

# check if already existent
if os.path.isfile('drive/MyDrive/Colab Notebooks/data/ASR_train_audio'+str(nsamples)+'.ft'):
    print ("File exist")
    df = pd.read_feather('drive/MyDrive/Colab Notebooks/data/ASR_train_audio'+str(nsamples)+'.ft')
else:
    print("File does not exist")

    # initialize with list
    audio_signals = len(df['ID'])*[[0]]
    df['audio_signal'] = audio_signals

    # functional but not elegant (nor fast probably)
    # split due to disconnects from kernel

    for k in range(nsamples):
      id = df.iloc[k]['ID']
      path_data = os.path.join('./clips/', id+'.mp3')
      waveform, rate = lb.load(path_data, sr=16*1e3)
      df.at[k, 'audio_signal'] = waveform

      if k % 100 == 0:
        print('file '+ str(k))

    # store as faster feather format
    df[:nsamples].to_feather('./ASR_train_audio'+str(nsamples)+'.ft')

    #
    df = df[:nsamples]
