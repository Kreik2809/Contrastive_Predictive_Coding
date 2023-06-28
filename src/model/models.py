import torch 
import torch.nn as nn

class SentenceEncoder(nn.Module):
    """ Sentence Encoder from CPC paper
    """
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).requires_grad_(False)
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

class AutoregressiveLSTM(nn.Module):
    """ Autoregressive LSTM from CPC paper
    """
    def __init__(self, input_dim, hidden_dim):
        super(AutoregressiveLSTM, self).__init__()
        self.LSTM = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        output, (hidden, cell)  = self.LSTM(x)
        return hidden + output.sum(1)

class CpcModel(nn.Module):
    """ CPC Model from CPC paper
    """
    def __init__(self, device, vocab_size, embedding_dim, output_dim, hidden_dim):
        super(CpcModel, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder = SentenceEncoder(vocab_size, embedding_dim, output_dim).to(self.device, non_blocking=False)
        self.ar = AutoregressiveLSTM(output_dim, hidden_dim).to(self.device, non_blocking=False)
        self.W = nn.Linear(hidden_dim+1, hidden_dim).to(self.device, non_blocking=False)   
        
    def forward(self, x):
        #x is a list containing x_hist, x_pos, x_neg, step. They are all tensors
        x_hist = x[0].to(self.device, non_blocking=False) #64 10 132
        x_pos = x[1].unsqueeze(1).to(self.device, non_blocking=False) #64 1 132
        x_neg = x[2].to(self.device, non_blocking=False) #64 10 132
        step = x[3].to(self.device, non_blocking=False) #64

        hist_size = x_hist.shape[1]
        pos_size = x_pos.shape[1]
        neg_size = x_neg.shape[1]
        batch_size = x_hist.shape[0]

        x = torch.cat((x_hist, x_pos, x_neg), dim=1) #64 21 132
        total_size = x.shape[1]

        #encode x
        x = x.view(-1, x.shape[2]) #64*21 132
        z = self.encoder(x) #64*21 2400

        z = z.view(batch_size, total_size, -1) #64 21 2400

        z_hist = z[:, :hist_size, :] #64 10 2400
        z_pos = z[:, hist_size:hist_size+pos_size, :] #64 1 2400
        z_neg = z[:, hist_size+pos_size:, :] #64 10 2400

        c_t = self.ar(z_hist) 
        c_t = c_t.squeeze(0)    

        c_t_step = torch.cat((c_t, step.unsqueeze(1)), dim=1)

        preds = self.W(c_t_step)
        #print(preds.shape) #64 2400
        #print(z_pos.shape) #64 1 2400
        #print(z_neg.shape) #64 10 2400

        f_pos = torch.exp(torch.bmm(preds.unsqueeze(1), z_pos.permute(0, 2, 1)).squeeze(1))
        f_neg = torch.exp(torch.bmm(preds.unsqueeze(1), z_neg.permute(0, 2, 1)).squeeze(1))

        #print(f_pos.shape) #64 1
        #print(f_neg.shape) #64 10
        
        accuracy = torch.sum(torch.diag(f_pos > f_neg.max(dim=1)[0])).item() / batch_size

        loss = torch.log(f_pos / (f_pos + f_neg))

        return -loss, accuracy

    def save(self, model_dir, models_file_prefix):
        torch.save(self.encoder.state_dict(), model_dir + models_file_prefix + "_encoder.pt")
        torch.save(self.ar.state_dict(), model_dir+models_file_prefix + "_ar.pt")
        torch.save(self.state_dict(), model_dir+models_file_prefix+".pt")
    
    def load(self, model_dir, model_file):
        self.load_state_dict(torch.load(model_dir + model_file))

    def summary(self):
        print(self.encoder)
        print(self.ar)
        print(self.W)
