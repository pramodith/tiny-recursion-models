# tiny-recursion-models
Reproduction of Less is More: Recursive Reasoning with [Tiny Networks.](https://arxiv.org/pdf/2510.04871)

## Note
**The codebase is still under development.** Feel free to fork and tinker yourself!

## Algorithm
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

## References
* Hierarchical Reasoning Model. [Code](https://github.com/sapientinc/HRM/tree/main).
```
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```
