# EC

Episodic control algorithms:

- [MFEC](https://arxiv.org/abs/1606.04460)
- [NEC](https://arxiv.org/abs/1703.01988)

Default options for MFEC:

```
python main.py --kernel mean
```

Default options for NEC:

```
python main.py --algorithm NEC \
               --key-size 128 \
               --num-neighbours 50 \
               --dictionary-capacity 500000 \
               --episodic-multi-step 100 \
               --epsilon-final 0.001 \
               --discount 0.99 \
               --learn-start 50000
```

Used in [Memory-efficient episodic control reinforcement learning with dynamic online k-means](https://arxiv.org/abs/1911.09560) and [Sample-Efficient Reinforcement Learning with Maximum Entropy Mellowmax Episodic Control](https://arxiv.org/abs/1911.09615), both presented at the [Workshop on Biological and Artificial Reinforcement Learning, NeurIPS 2019](https://sites.google.com/view/biologicalandartificialrl/home).

## Acknowledgements

- Alex Pritzel for implementation details
