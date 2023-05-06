import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn._reduction as _Reduction


class SADLikeLoss(torch.nn.modules.loss._Loss):
   
    __constants__ = ['reduction', 'eta', 'epsilon']

    def __init__(self,
                    size_average=None,
                    reduce=None,
                    reduction: str = 'mean',
                    eta: float = 1.,
                    epsilon: float = 1e-6) -> None:
        super(SADLikeLoss, self).__init__(size_average,
                                                reduce,
                                                reduction)

        self.eta = eta
        self.epsilon = epsilon


    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                label: torch.Tensor = None) -> torch.Tensor:

        # Unlabeled case

        if (label == None) or\
                (torch.count_nonzero(label) == 0):

            return F.mse_loss(input, target, reduction=self.reduction)

        # Labeled case

        else:

            losses = F.mse_loss(input, target, reduction='none')

            # print(label.detach().cpu().numpy())

            label = torch.unsqueeze(label, 1)
            label = torch.unsqueeze(label, 2)

            label = label.expand(*losses.shape)

            losses = torch.where(label == 0,
                                    losses,
                                    self.eta*(1/(losses + self.epsilon)))

            return torch.mean(losses)


