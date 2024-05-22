import torch
import torch.nn as nn

torch.manual_seed(1337)

# Hyperparameters

# Following params taking too long with my GPU
#batch_size = 64
#block_size = 256
#n_embed = 384
#n_head = 6
#n_layer = 6
batch_size = 64
block_size = 256
n_embed = 64
n_head = 3
n_layer = 2
dropout_ratio = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
training_steps = 5000
learning_rate = 3e-4
eval_interval = 10
eval_size = 200

# Data setup
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

chtoi = {ch:i for i,ch in enumerate(chars)}
itoch = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [chtoi[c] for c in s]
decode = lambda l: ''.join([itoch[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
training_size = int(0.9 * len(data))
train_data = data[:training_size]
val_data = data[training_size:]

def get_batch(dataset):
    data = train_data if dataset == 'train' else val_data
    rng = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in rng]).to(device)
    y = torch.stack([data[i+1:i+1+block_size] for i in rng]).to(device)
    return x, y

# Model definition
class SelfAttentionHead(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.key = nn.Linear(n_embed, size, bias=False)
        self.query = nn.Linear(n_embed, size, bias=False)
        self.value = nn.Linear(n_embed, size, bias=False)
        self.dropout = nn.Dropout(dropout_ratio)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape # (B=batch_size,T=block_size,C=n_embed)
        
        k = self.key(x) # (B,T,size)
        q = self.query(x) # (B,T,size)
        
        wei = q @ k.transpose(-2, -1) * self.size**-0.5 # (B,T,size) @ (B,size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,size) -> (B,T,size)
        return out

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, count, size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(size) for _ in range(count)])
        self.proj = nn.Linear(count * size, n_embed)
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over the last dimension (size)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout_ratio),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, size, head_count):
        super().__init__()
        self.self_attention = SelfAttentionMultiHead(head_count, int(size / head_count))
        self.feed_forward = FeedForward(size)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)
    
    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class GPTModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range (n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.model_head = nn.Linear(n_embed, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, contexts, targets=None):
        B,T = contexts.shape
        token_embed = self.token_embedding_table(contexts) # (B,T,n_embed)
        position_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embed)
        x = token_embed + position_embed # (B,T,n_embed)
        x = self.blocks(x) # (B,T,n_embed)
        x = self.layer_norm(x) # (B,T,n_embed)
        logits = self.model_head(x) # (B,T,C=vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, contexts, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(contexts[:, -block_size:])
            # Just get the last time step
            # This is silly as we are passing the full context but a Bigram model is not using it,
            # but this gives the structure for upgrading to a Transformer later
            logits = logits[:, -1, :] # (B,T,C) -> (B,C)
            proba = nn.functional.softmax(logits, dim=1)
            next_batch = torch.multinomial(proba, num_samples=1) # (B,1)
            contexts = torch.cat((contexts, next_batch), dim=1) # (B,T+1)
        return contexts

# Train or Load and Run
train_model_from_scratch = False
model = GPTModel()
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters in the model')

if train_model_from_scratch:
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for data_split in ['train', 'eval']:
            losses = torch.zeros(eval_size)
            for eval_step in range(eval_size):
                x, y = get_batch(data_split)
                logits, loss = model(x, y)
                losses[eval_step] = loss.item()
            out[data_split] = losses.mean()
        model.train()
        return out

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(training_steps):
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, eval loss: {losses['eval']:.4f}")
            torch.save(model.state_dict(), 'model')
    
        x, y = get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
else:
    model.load_state_dict(torch.load('model'))

# Generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=1000)[0]
print(decode(output.tolist()))
output = model.generate(context, max_new_tokens=10000)[0]
open('output.txt', 'w').write(decode(output.tolist()))
