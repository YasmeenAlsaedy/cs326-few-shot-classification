from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from .lenet import LeNet




class ProtoNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        # TODO(protonet): your code here
        # Use the same embedder as in LeNet
        self.embedder = nn.Sequential(   # Body
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),

            # Neck
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            - x: images of shape [num_classes * batch_size, c, h, w]
        """
        # Aggregating across batch-size
        num_classes = self.config['training']['num_classes_per_task']
        batch_size = len(x) // num_classes
        c, h, w = x.shape[1:]
        embeddings = self.embedder(x) # [num_classes * batch_size, dim]

        # TODO(protonet): compute prototypes given the embeddings
        embedds = embeddings.view([num_classes, batch_size, -1]) # num_samples * dim [[0.232],[0.12],[0.12]]


        prototypes = embedds.mean(1)  # num_class * dim [[0.12],[0.43],[0.12],[0.43],[0.12]]

        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        # torch.

        n = embeddings.size(0)
        m = prototypes.size(0)
        d = embeddings.size(1)

        embeddings = embeddings.unsqueeze(1).expand(n, m, d)
        prototypes = prototypes.unsqueeze(0).expand(n, m, d)

        logits = self.fun_L(embeddings,  prototypes)
        #  cross entropy return the index of the highest distance
        #  adding - to get the smallest distance
        return -logits

    def fun_L(self, x,  pro):
        # [[0.12, 0.23, 0.23,0.12,0.123]
        # [0.12, 0.23, 0.23,0.12,0.123],
        # [0.12, 0.23, 0.23,0.12,0.123],
        # [0.12, 0.23, 0.23, 0.12, 0.123],
        # [0.12, 0.23, 0.23,0.12,0.123]]
        # output = num_sample * num_classes
        return torch.pow(x-pro, 2).sum(2)
