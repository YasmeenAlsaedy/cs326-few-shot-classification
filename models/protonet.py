from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class ProtoNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        # TODO(protonet): your code here
        # Use the same embedder as in LeNet
        self.embedder = nn.Sequential(
            # Body
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
            nn.Flatten(),

            # Head
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, config['training']['num_classes_per_task']),
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
        print(embeddings)
        # TODO(protonet): compute prototypes given the embeddings
        # embeddings.sum()/num_classes
        #
        prototypes = embeddings.sum()/num_classes

        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        # torch.
        print(batch_size)
        print(num_classes)
        print(prototypes)
        cos = nn.CosineSimilarity(eps=1e-6)
        logits = cos(prototypes, nn.Softmax(embeddings))

        return logits
