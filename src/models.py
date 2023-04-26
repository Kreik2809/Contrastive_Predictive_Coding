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
    
    def forward(self, x_hist, x_neg, x_pos, step):
        z_hist = self.encoder(x_hist)
        c_t = self.ar(z_hist)
        
        z_neg = self.encoder(x_neg)
        z_pos = self.encoder(x_pos)

        f_neg = torch.exp(torch.matmul(c_t, torch.matmul(self.W[step], z_neg)))
        f_pos = torch.exp(torch.matmul(c_t, torch.matmul(self.W[step], z_pos)))

        return f_neg, f_pos
        
if __name__ == "__main__":
    sentence = "Hello I am Nicolas"
    tokenized_sentence = [1,2,3,4]
    tokenized_sentence = torch.tensor(tokenized_sentence)
    batch = torch.stack([tokenized_sentence, tokenized_sentence, tokenized_sentence])
    model = SentenceEncoder(5, 128, 2400)
    output = model(batch)
    print(output)
    print(output.shape)

    model = AutoregressiveGRU(2400, 2400)
    output = model(output)
    print(output)
    print(output.shape)
    
