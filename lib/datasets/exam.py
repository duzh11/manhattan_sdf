import torch
a=[i for i in range(10)]
sampler = torch.utils.data.RandomSampler(a)
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 2, drop_last=False)
iteration=0
while iteration <= 10:
            for batch in batch_sampler:
                iteration += 1
                if iteration > 10:
                    break
                print(batch)