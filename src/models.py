import torch 
import torch.nn as nn

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = torch.permute(embeddings, (0, 2, 1))
        conv_out = self.conv1(embeddings)
        relu_out = self.relu(conv_out)
        pooled = nn.functional.avg_pool1d(relu_out, kernel_size=conv_out.shape[2])
        z = pooled.squeeze(2)
        return z

class AutoregressiveGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoregressiveGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        _, x = self.gru(x)
        return x

class CpcModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim, steps = 3):
        super(CpcModel, self).__init__()
        self.steps = steps
        self.encoder = SentenceEncoder(vocab_size, embedding_dim, output_dim)
        self.ar = AutoregressiveGRU(output_dim, hidden_dim)
        self.W = nn.Parameter(torch.randn((steps, output_dim, output_dim)))  
    
    def forward(self, x):
        #x is a list containing x_hist, x_pos, x_neg, step which are all tensors
        x_hist = x[0]
        x_pos = x[1]
        x_neg = x[2]
        step = x[3]

        x_hist = x_hist.view(-1, x_hist.shape[2])
        z_hist = self.encoder(x_hist)
        z_hist = z_hist.view(x[0].shape[0], x[0].shape[1], -1)
        c_t = self.ar(z_hist) #batch dim
        c_t = c_t.squeeze(0) #remove batch dim
        
        x_neg = x_neg.view(-1, x_neg.shape[2])
        z_neg = self.encoder(x_neg)
        z_pos = self.encoder(x_pos)

        #print(c_t.shape) #torch.Size([64, 2400])
        #print(z_neg.shape) #torch.Size([640, 2400]) => 64,2400 ?
        #print(z_pos.shape) #torch.Size([64, 2400])

        f_pos = torch.matmul(z_pos, torch.matmul(self.W[step-1], c_t.T))
        f_neg = torch.matmul(z_neg, torch.matmul(self.W[step-1], c_t.T))

        #print(f_pos.shape) #torch.Size([64, 64, 64])
        #print(f_neg.shape) #torch.Size([64, 640, 64])
        f_neg = f_neg.sum(dim=1)
        #print(f_neg.shape) #torch.Size([64, 64])

        return - (torch.exp(f_pos)/torch.exp(f_neg)) #NAN bc of torch.exp
        
if __name__ == "__main__":
    sentence = "Hello I am Nicolas"
    tokenized_sentence = [1,2,3,4]
    tokenized_sentence = torch.tensor(tokenized_sentence)
    batch = torch.stack([tokenized_sentence, tokenized_sentence, tokenized_sentence])
    model = SentenceEncoder(5, 128, 2400)
    output = model(batch)
    print(output.shape)
    print(output.unsqueeze(0).shape)
    model = AutoregressiveGRU(2400, 2400)
    output = model(output.unsqueeze(0)) #batch dim
    print(output)
    print(output.shape)
    
