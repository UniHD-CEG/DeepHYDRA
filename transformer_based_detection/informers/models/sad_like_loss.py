import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn._reduction as _Reduction


class SADLikeLoss(torch.nn.modules.loss._Loss):
   
    __constants__ = ['seq_len',
                        'pred_len',
                        'reduction',
                        'eta',
                        'epsilon']

    def __init__(self,
                    size_average=None,
                    reduce=None,
                    reduction: str = 'mean',
                    seq_len: int = 64,
                    pred_len: int = 1,
                    eta: float = 0.1,
                    epsilon: float = 1e-6,
                    device='cuda:0') -> None:
        super(SADLikeLoss, self).__init__(size_average,
                                                reduce,
                                                reduction)

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.eta = eta
        self.epsilon = epsilon

        self.temporal_mask =\
                torch.linspace(0.5, 1, self.seq_len).to(device)


    def _get_losses_constant_mask(self,
                                    losses: torch.Tensor,
                                    label: torch.Tensor) -> torch.Tensor:
        
        # print(losses.shape)
        # print(label.shape)

        anomaly_counts = torch.count_nonzero(label, dim=1)

        factors = anomaly_counts/self.seq_len

        # factors *= self.temporal_mask

        factors = factors.unsqueeze(-1)
        factors = factors.unsqueeze(-1)

        # anomaly_counts = anomaly_counts.expand(*losses.shape)

        # losses[mask, :, :] = self.eta/\
        #                         (factors[mask, :, :]*\
        #                             losses[mask, :, :] +\
        #                             self.epsilon)

        # losses = (1 - factors)*losses +\
        #             0.12*factors*F.tanh(1/(losses + self.epsilon))
                                    
        losses = (1 - factors)*losses +\
                    0.05*factors*F.tanh(1/(losses + self.epsilon))

        # print(losses)

        return losses


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

            # label_np = label.detach().cpu().numpy()

            # for count, row in enumerate(label_np):

            #     print(f'{count}: ', end='')
            #     for element in row:
            #         print(int(element), end='')
            #     print()
                
            # exit()

            losses = self._get_losses_constant_mask(losses, label)

            return torch.mean(losses)


