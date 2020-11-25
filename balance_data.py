import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
# torch.cuda.set_device(1)

class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="random"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from.
            class_choice: a string indicating how class will be selected for every
            sample.
                "random": class is chosen uniformly at random.
                "cycle": the sampler cycles through the classes sequentially.
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))
        self.map = []
        for class_ in range(self.labels.shape[1]):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.map.append(lst)
        all_zero = []
        for row in range(self.labels.shape[0]):
            if not np.any(labels[row]):
                all_zero.append(row)

        print("all zero sample number is: ",len(all_zero))
        self.map.append(all_zero)
        print("counting-----")
        for i in range(len(self.map)):
            print("class {0} has {1} samples:".format(i,len(self.map[i])))

        assert class_choice in ["random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # if self.count >= len(self.indices):
        if self.count >= 20000:
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1])# - 1)
            # print(class_)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        class_indices = self.map[class_]
        return np.random.choice(class_indices)

    def __len__(self):
        return 20000
        # return len(self.indices)

# if __name__ == "__main__":
#     train_dataset = Video2RollDataset(subset='train')
#     train_sampler = MultilabelBalancedRandomSampler(train_dataset.train_labels)
#     train_data_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
#     for i, data in enumerate(train_data_loader):
#         print(i)
#         imgs,label,ref_imgs,rng = data
#         print(torch.unique(torch.nonzero(label)[:,1]))
#         for j in range(len(label)):
#             if label[j].sum()==0:
#                 print("yes")
#         if i == 1:
#             break