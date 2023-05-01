import torch


def get_class_weights(loader):
    print(f'{"#" * 10} Getting Class Weights {"#" * 10}')
    counts = torch.zeros(size=(4, ))

    for batch, sample in enumerate(loader):
        counts += sample['label'].sum(dim=0)
    weights = (len(loader.dataset) - counts) / counts

    print(f'Total: {len(loader.dataset)}')
    print(f'Positive Counts: {counts}')
    print(f'Positive Weights: {weights}')

    return weights


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1