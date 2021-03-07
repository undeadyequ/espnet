"""
All parameter setting
1. Data
2. Network
3. Training
4. Speech processing
"""
import torch

class Hyperparameters():
    ########## 1.Data ###########
    style_wav = "./ref_wav/style_audio.wav"
    eval_text = 'it took me a long time to develop a brain . now that i have it i\'m not going to be silent !'


    #### LJ speech
    lj_data = "/data/luo/LJSpeech-1.1"

    #### IEMOCAP
    ## rate = 16000
    iemocap_data = "/data/luo/IEMOCAP_full_release/"
    # Emotion Iemocap
    emoab = ['ang', 'fru', 'exc', 'fea', 'sad', 'sur', 'neu', 'hap', 'xxx', "oth", "dis"]
    EMO_NOT_EXIST = ["xxx", "oth"]
    emo2idx = {emo: idx for idx, emo in enumerate(emoab)}
    idx2emo = {idx: emo for idx, emo in enumerate(emoab)}
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    #----------------------#
    # Text
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding??? E: End of Sentence  Other symbols, eg. !!! ???
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}

    ########## 2. Network ###########
    #### 2.1 Encoder Part
    # Encoder
    num_highways_e = 4
    n_fft = 1024

    E = 256  # Embedding
    K = 16   # Conv1d bank numbers
    n_mels = 80
    r = 5


    K_POST = 8

    # Refence filter
    ref_enc_filters = [32, 32, 64, 64, 128, 128]

    # Post Encoder
    num_highways_d = 4

    #### 2.2 Deocder part
    # Multi-head attention
    nums_head = 8
    nums_token = 10

    # Post Encoder
    num_highways_d = 4

    ########## 3. Training ###########
    epoch_num = 300
    epoch_num_emo = 300
    batch_size = 32
    batch_size_emo = 32

    lr_modify_step = [500000, 1000000, 2000000]
    log_per_batch = 20

    # Epoch for 2 step
    epoch_num_acous = 200  # step1
    epoch_num_emo = 300 # step2

    DEVICE = "cuda:0"
    lr = 0.001
    max_Ty = 200
    save_per_epoch = 2


    ########## 4. Speech processing ###########
    sr = 22050  # Sample rate.
    #sr = 16000  # keda, thchs30, aishell
    n_fft = 1024  # fft points (samples) - ALE changed this from 2048
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    #hop_length = int(sr * frame_shift)  # samples.
    #win_length = int(sr * frame_length)  # samples.
    hop_length = 256  # samples.
    win_length = 1024  # samples.

    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 100  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20
    n_priority_freq = int(3000  / (sr * 0.5) * (n_fft /2))
