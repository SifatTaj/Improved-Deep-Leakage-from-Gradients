import numpy as np
import torch
from ptflops import get_model_complexity_info
from aug_models.anon_densenet import AnonDenseNet121

from aug_utils.print_nonzero import print_nonzeros
from main_vgg import deanon


def count_flop(net):
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)

    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print(f'FLOPs: {macs}')
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    aug_percents = [0, .25, .50, .75, 1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for aug_percent in aug_percents:
        num_classes = 10
        num_epochs = 100

        aug_fc1 = int(512 * aug_percent)
        aug_out = int(num_classes * aug_percent)

        aug_indices_fc1 = np.arange(start=512, stop=512 + aug_fc1, step=1)
        aug_indices_fc2 = np.arange(start=512, stop=num_classes + aug_out, step=1)

        net = AnonDenseNet121(fc_aug_percent=aug_percent, aug_indices_fc1=aug_indices_fc1,
                              aug_indices_fc2=aug_indices_fc2)

        print_nonzeros(net)
        print('after deanonymization')
        print_nonzeros(deanon(net, aug_indices_fc1, aug_indices_fc2, device='cuda'))
        # count_flop(net)
