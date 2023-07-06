from typing import Optional
import sys

import numpy as np
import torch


def zipf_probs(alpha: float, users: int, support: list[int],
               random_state: np.random.Generator):
    samples = random_state.zipf(alpha, users)
    while sum(samples) > min(support):
        samples = random_state.zipf(alpha, users)

    return samples/sum(samples)


def zipf_sampler(alpha: float, users: int, dataset: torch.utils.data.Dataset,
                 validation_split: float,
                 random_state: np.random.Generator):
    label_indeces = dataset.class_to_idx.values()
    classes = len(label_indeces)
    support = [torch.count_nonzero(dataset.targets == c).item()
               for c in label_indeces]

    probs = zipf_probs(alpha, users, support, random_state)

    partitions = [[] for u in range(users)]
    for c in range(classes):
        user_probs = np.roll(probs, c)
        class_ids = torch.where(dataset.targets == c)[0].numpy()
        random_state.shuffle(class_ids)

        offset = 0
        for user, p in enumerate(user_probs):
            upper_bound = offset + int(p*class_ids.shape[0])
            partitions[user].extend(
                    class_ids[offset:upper_bound])

            offset = upper_bound

    user_splits = []
    for i, ids in enumerate(partitions):
        ids = random_state.permutation(ids)
        val_size = int(validation_split*len(ids))
        training = torch.utils.data.Subset(dataset, ids[val_size:])
        validation = torch.utils.data.Subset(dataset, ids[:val_size])
        user_splits.append((training, validation))

    print("trianing splits:",
          [len(t) for t, _ in user_splits], file=sys.stderr)
    print("validation splits:",
          [len(v) for _, v in user_splits], file=sys.stderr)

    return user_splits


def balanced_iid_sampler(users: int, dataset: torch.utils.data.Dataset,
                         validation_split: float,
                         random_state: np.random.Generator,
                         items_per_user: Optional[int] = None):
    label_indeces = dataset.class_to_idx.values()
    classes = len(label_indeces)

    if items_per_user is None:
        smallest_class = min(
                torch.count_nonzero(dataset.targets == c).item()
                for c in range(classes))
        items_per_user = (smallest_class*classes)//users
        print(smallest_class)

    items_per_class = items_per_user//classes
    training_items_per_class = int(items_per_class*(1-validation_split))

    user_training = [[] for i in range(users)]
    user_validation = [[] for i in range(users)]
    for c in range(classes):
        class_ids = torch.where(dataset.targets == c)[0].numpy()
        random_state.shuffle(class_ids)
        for u in range(users):
            user_training[u].extend(
                class_ids[
                    u*items_per_class:
                    u*items_per_class+training_items_per_class])
            user_validation[u].extend(
                class_ids[
                    u*items_per_class+training_items_per_class:
                    (u+1)*items_per_class])

    print("training splits:",
          [len(u) for u in user_training], file=sys.stderr)
    print("validation splits:",
          [len(u) for u in user_validation], file=sys.stderr)

    return [(torch.utils.data.Subset(dataset, t),
             torch.utils.data.Subset(dataset, v))
            for t, v in zip(user_training, user_validation)]
