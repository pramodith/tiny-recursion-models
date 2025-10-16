# tiny-recursion-models
Reproduction of Less is More: Recursive Reasoning with Tiny Networks

```
def latent recursion(x, y, z, n=6):
    for i in range(n): # latent reasoning
        z = net(x, y, z)
    y = net(y, z) # refine output answer
    return y, z

def deep recursion(x, y, z, n=6, T=3):
    # recursing T−1 times to improve y and z (no gradients needed)
    with torch.no grad():
        for j in range(T−1):
            y, z = latent recursion(x, y, z, n)
    # recursing once to improve y and z
    y, z = latent recursion(x, y, z, n)
    return (y.detach(), z.detach()), output head(y), Q head(y)

# Deep Supervision
for x input, y true in train dataloader:
    y, z = y init, z init
    for step in range(N supervision):
        x = input embedding(x input)
        (y, z), y hat, q hat = deep recursion(x, y, z)
        loss = softmax cross entropy(y hat, y true)
        loss += binary cross entropy(q hat, (y hat == y true))
        loss.backward()
        opt.step()
        opt.zero grad()
        if q hat > 0: # early−stopping
            break
```