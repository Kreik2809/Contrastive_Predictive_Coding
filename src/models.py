import torch 
import torch.nn as nn

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #x is a tensor of size (batch_size, elem_seq, seq_len)
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
    def __init__(self, device, vocab_size, embedding_dim, output_dim, hidden_dim):
        super(CpcModel, self).__init__()
        self.device = device
        self.encoder = SentenceEncoder(vocab_size, embedding_dim, output_dim)
        self.ar = AutoregressiveGRU(output_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim+1, hidden_dim) 

    
    def forward(self, x):
        #x is a list containing x_hist, x_pos, x_neg, step which are all tensors
        x_hist = x[0].to(self.device) #64 10 132
        x_pos = x[1].unsqueeze(1).to(self.device) #64 1 132
        x_neg = x[2].to(self.device) #64 10 132 #concatenate 
        step = x[3].to(self.device)


        #print(x_hist.shape) #64 10 132
        #print(x_pos.shape) #64 1 132
        #print(x_neg.shape) #64 10 132

        hist_size = x_hist.shape[1]
        pos_size = x_pos.shape[1]
        neg_size = x_neg.shape[1]
        batch_size = x_hist.shape[0]

        #concat x_hist and x_pos and x_neg along axis 1
        x = torch.cat((x_hist, x_pos, x_neg), dim=1) #64 21 132
        #print(x.shape) #64 21 132
        total_size = x.shape[1]

        #encode x
        x = x.view(-1, x.shape[2]) #64*21 132
        z = self.encoder(x) #64*21 2400

        z = z.view(batch_size, total_size, -1) #64 21 2400

        z_hist = z[:, :hist_size, :] #64 10 2400
        z_pos = z[:, hist_size:hist_size+pos_size, :] #64 1 2400
        z_neg = z[:, hist_size+pos_size:, :] #64 10 2400

        #print(z_hist.shape)
        #print(z_pos.shape)
        #print(z_neg.shape)

        c_t = self.ar(z_hist) 
        c_t = c_t.squeeze(0)    

        c_t_step = torch.cat((c_t, step.unsqueeze(1)), dim=1)

        preds = self.W(c_t_step)
        #print(preds.shape) #64 2400
        #print(z_pos.shape) #64 1 2400
        #print(z_neg.shape) #64 10 2400

        #perform elementwise multiplication between preds and z_pos and z_neg
        f_pos = torch.exp(torch.bmm(preds.unsqueeze(1), z_pos.permute(0, 2, 1)).squeeze(1))
        f_neg = torch.exp(torch.bmm(preds.unsqueeze(1), z_neg.permute(0, 2, 1)).squeeze(1))

        #print(f_pos.shape) #64 1
        #print(f_neg.shape) #64 10
        
        accuracy = torch.sum(torch.diag(f_pos > f_neg.max(dim=1)[0])).item() / batch_size

        loss = torch.log(f_pos / (f_pos + f_neg))

        return -loss, accuracy

    def save(self, file_name):
        torch.save(self.encoder.state_dict(), file_name + "_encoder.pt")
        torch.save(self.ar.state_dict(), file_name + "_ar.pt")
        torch.save(self.state_dict(), file_name+".pt")
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
               

        
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
    
