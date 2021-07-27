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

## Notes

Notes about this codebase based on conversations with Robert Kirk

- This uses brute-force kNN search, as opposed to the approximate kNN search used in the original works
- This uses separate hash functions, as in the [original works](https://github.com/EndingCredits/Neural-Episodic-Control/issues/4)
- The keys and values in the NEC table are not updated - this is a bug!
- Key/value gradients are taken from the first instance of each updated key, but averaging would make more sense
- Despite the bug, this can still reproduce the results from the NEC paper (hence those updates may not be important)

## Acknowledgements

- Alex Pritzel for implementation details
- [@RobertKirk](https://github.com/RobertKirk) for debugging
