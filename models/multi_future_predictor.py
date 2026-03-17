import torch
import torch.nn as nn


class MultiFuturePredictor(nn.Module):

    def __init__(self, in_dim=768, hidden_dim=512, num_branches=3):
        super().__init__()

        self.num_branches = num_branches

        # shared embedding
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # future branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_branches)
        ])

        # event heads
        self.event_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_branches)
        ])

    def forward(self, z):

        shared_feat = self.shared(z)

        branch_features = []
        branch_logits = []

        for i in range(self.num_branches):

            f = self.branches[i](shared_feat)
            logit = self.event_heads[i](f)

            branch_features.append(f)
            branch_logits.append(logit)

        branch_logits = torch.stack(branch_logits, dim=1)

        return {
            "branch_logits": branch_logits,
            "branch_features": branch_features,
            "shared_feature": shared_feat
        }