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
        self.embedder = LeNet(config)

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
        # embeddings.sum()/num_classes
        prototypes = embeddings.sum()/num_classes

        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        # torch.
        funcs = lambda x: torch.sqrt(torch.pow(x-pro, 2))
        softmax = nn.Softmax(dim=1)
        result = softmax(embeddings).clone().detach()
        logits = torch.tensor([funcs(f).tolist() for f in result], requires_grad = True).to(self.config['device'])

        return logits