import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        
        self.rnn = nn.LSTM(
            input_size=32*40,   # depends on pooling
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(512, vocab_size) # depends on bidirectionality and hidden size

    def forward(self, x):
        x = self.cnn(x)
        b, c, t, f = x.size()
        x = x.permute(0,2,1,3).contiguous().view(b, t, c*f)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
