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
        print(x.shape[1:])
        print(x)
        embeddings = self.embedder(x) # [num_classes * batch_size, dim]
        print(embeddings)

        # TODO(protonet): compute prototypes given the embeddings
        # embeddings.sum()/num_classes
        prototypes = embeddings.sum()/num_classes

        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        # torch.
        print(batch_size)
        print(num_classes)
        print(prototypes)
        funcs = lambda x: torch.pow(x - prototypes, 2)
        softmax = nn.Softmax(dim=1)
        result = torch.tensor(softmax(embeddings))
        logits = torch.Tensor( [funcs(f).tolist() for f in result]).to(self.config['device'])

        return logits