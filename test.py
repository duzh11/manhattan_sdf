import torch
dataset=[1,2,3,4,5,6,7]
sampler = torch.utils.data.sampler.RandomSampler(dataset)
for i in sampler:
    print(i)
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
for i in batch_sampler:
    print(i)

print('complete')