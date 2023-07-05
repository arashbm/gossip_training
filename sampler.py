import numpy as np
import torch


def zipf_probs(alpha: float, users: int, support: list[int],
               random_state: np.random.Generator):
    samples = random_state.zipf(alpha, users)
    while sum(samples) > min(support):
        samples = random_state.zipf(alpha, users)

    return samples/sum(samples)


def zipf_sampler(alpha: float, users: int, dataset: torch.utils.data.Dataset,
                 validation_split: float, random_state: np.random.Generator):
    label_indecies = dataset.class_to_idx.values()
    classes = len(label_indecies)
    support = [torch.count_nonzero(dataset.targets == c).item()
               for c in label_indecies]

    probs = zipf_probs(alpha, users, support, random_state)

    partitions = [[] for u in range(users)]
    for c in range(classes):
        user_probs = np.roll(probs, c)
        class_ids = torch.where(dataset.targets == c)[0].numpy()

        offset = 0
        for user, p in enumerate(user_probs):
            upper_bound = offset + int(p*class_ids.shape[0])

            if user == users-1:
                upper_bound = class_ids.shape[0]

            partitions[user].extend(class_ids[offset:upper_bound])

            offset = upper_bound

    user_splits = []
    for i, ids in enumerate(partitions):
        ids = random_state.permutation(ids)
        val_size = int(validation_split*len(ids))
        training = torch.utils.data.Subset(dataset, ids[val_size:])
        validation = torch.utils.data.Subset(dataset, ids[:val_size])
        user_splits.append((training, validation))

    return user_splits
