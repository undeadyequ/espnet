import torch

from espnet2.tts.ser.ser_model import SER_NN
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torch.nn as nn

def run(epoch=100):
    speech_dataset = SpeechEmoDataset(csv_f="")
    speech_dataloader = DataLoader(speech_dataset, batch_size=5, shuffle=True, num_workers=2)

    ser_model = SER_NN(8, 5, units=128, layers=2)
    optimizer = optim.SGD(ser_model.parameters(), lr=0.001, momentum=0.9)

    for e in range(epoch):
        running_loss = 0.0
        # train
        for i, b in enumerate(speech_dataloader):
            emo_feats_pred = ser_model(b)
            loss = abs(emo_feats_pred - b)
            loss.backward()
            optimizer.step()

            running_loss += loss

            if i % 200 == 199:
                print("[%d, %5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        # evaluate


    print("training finished")

    PATH = "epoch.pth"
    torch.save(ser_model.state_dict(), PATH)



class SpeechEmoDataset(Dataset):
    def __init__(self, csv_f):
        pass
    def __getitem__(self, item):
        pass
    def __len__(self):
        pass


if __name__ == '__main__':
    run()