import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the Relation Network.

        Args:
            input_dim (int): Dimension of the input features (d).
            hidden_dim (int): Dimension of the hidden layer in the relation head.
        """
        super(RelationNet, self).__init__()
        # Relation network head
        self.relation_head = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Concatenated features have dimension 2 * d
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output a score between 0 and 1
        )

    def forward(self, query, support, support_labels, n_way, n_shot):
        """
        Forward pass of the Relation Network.

        Args:
            query (torch.Tensor): A tensor of shape (tasks_per_batch, n_query, d).
            support (torch.Tensor): A tensor of shape (tasks_per_batch, n_support, d).
            support_labels (torch.Tensor): A tensor of shape (tasks_per_batch, n_support).
            n_way (int): Number of classes in the few-shot task.
            n_shot (int): Number of support examples per class.

        Returns:
            torch.Tensor: A tensor of shape (tasks_per_batch, n_query, n_way) representing class scores.
        """
        tasks_per_batch, n_query, d = query.size()
        _, n_support, _ = support.size()

        # Expand query and support to compute pairwise relations
        query_expanded = query.unsqueeze(2).expand(-1, -1, n_support, -1)  # (tasks_per_batch, n_query, n_support, d)
        support_expanded = support.unsqueeze(1).expand(-1, n_query, -1, -1)  # (tasks_per_batch, n_query, n_support, d)

        # Concatenate query and support features
        relation_pairs = torch.cat((query_expanded, support_expanded), dim=3)  # (tasks_per_batch, n_query, n_support, 2*d)

        # Compute relation scores
        relation_scores = self.relation_head(relation_pairs)  # (tasks_per_batch, n_query, n_support, 1)
        relation_scores = relation_scores.squeeze(3)  # (tasks_per_batch, n_query, n_support)

        # Aggregate relation scores for each class
        support_labels_one_hot = F.one_hot(support_labels, n_way).float()  # (tasks_per_batch, n_support, n_way)
        class_scores = torch.bmm(relation_scores, support_labels_one_hot)  # (tasks_per_batch, n_query, n_way)

        return class_scores