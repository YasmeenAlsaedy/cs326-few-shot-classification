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
        print(embeddings)

        # TODO(protonet): compute prototypes given the embeddings
        # embeddings.sum()/num_classes
        #prototypes = embeddings.sum()/num_classes
        prototypes  = [(pclass.sum()/num_classes).item() for embedd in embeddings for pclass in embedd]
        print(prototypes)
        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        # torch.
        #euclidean_dist = lambda x: torch.sqrt(torch.pow(x-prototypes, 2))
        softmax = nn.Softmax(dim=1)
        result = softmax(embeddings).clone().detach()
        print(result)
        #logits = torch.tensor([euclidean_dist(r).tolist() for r in result], requires_grad = True).to(self.config['device'])
        logits = torch.tensor([(self.fun_L(v, prototypes[i])).tolist() for f in result for i, v in enumerate(f)]).clone().detach().view([batch_size, num_classes, -1]).to(self.config['device'])
        print(logits)

        return logits

    def fun_L(x,  pro):
        return torch.sqrt(torch.pow(x-pro, 2))