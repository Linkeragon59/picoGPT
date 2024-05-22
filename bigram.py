import torch
import torch.nn as nn

torch.manual_seed(1337)

# Hyperparameters
block_size = 8 # max size of a chunk that can be passed to a transformer (not useful for a bigram model :P)
batch_size = 32 # number of rows of the tensors, they are independent batches that are processed independently
training_steps = 2000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 200
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
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, contexts, targets=None):
        logits = self.token_embedding_table(contexts) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, contexts, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(contexts)
            # Just get the last time step
            # This is silly as we are passing the full context but a Bigram model is not using it,
            # but this gives the structure for upgrading to a Transformer later
            logits = logits[:, -1, :] # (B,T,C) -> (B,C)
            proba = nn.functional.softmax(logits, dim=1)
            next_batch = torch.multinomial(proba, num_samples=1) # (B,1)
            contexts = torch.cat((contexts, next_batch), dim=1) # (B,T+1)
        return contexts

model = BigramLanguageModel(vocab_size)
model.to(device)

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
    
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=1000)[0]
print(decode(output.tolist()))
