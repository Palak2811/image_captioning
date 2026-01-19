import torch
import torch.nn as nn
import torchvision.models as models
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Remove avgpool + fc
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        # (B, 2048, 7, 7)

        features = features.permute(0, 2, 3, 1)
        # (B, 7, 7, 2048)

        features = features.view(features.size(0), -1, 2048)
        # (B, 49, 2048)

        return features
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim):
        super().__init__()
        self.enc = nn.Linear(encoder_dim, att_dim)
        self.dec = nn.Linear(decoder_dim, att_dim)
        self.fc = nn.Linear(att_dim, 1)

    def forward(self, encoder_out, hidden):
        att = self.fc(torch.tanh(
            self.enc(encoder_out) + self.dec(hidden).unsqueeze(1)
        )).squeeze(2)

        alpha = torch.softmax(att, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()

        self.hidden_size = hidden_size  # ✅ STORE IT

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(2048, hidden_size, 512)
        self.lstm = nn.LSTMCell(embed_size + 2048, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)

        # ✅ USE self.hidden_size
        hidden = torch.zeros(batch_size, self.hidden_size).to(encoder_out.device)
        cell   = torch.zeros(batch_size, self.hidden_size).to(encoder_out.device)

        embeddings = self.embedding(captions[:, :-1])
        outputs = []

        for t in range(embeddings.size(1)):
            context = self.attention(encoder_out, hidden)
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            hidden, cell = self.lstm(lstm_input, (hidden, cell))
            outputs.append(self.fc(hidden))

        return torch.stack(outputs, dim=1)