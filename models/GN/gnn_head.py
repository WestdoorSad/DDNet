import random
from collections import OrderedDict
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class PARN_CHAtt(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)

        return out * x


class PARN_SPAtt(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding="same", bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out * x

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.normtype = 'batch'
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        if self.normtype == 'batch':
            self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine, track_running_stats=track_running_stats))
        elif self.normtype == 'instance':
            self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out



class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 groups,
                 ratio,
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                groups=groups[l],
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(node_feat.device)

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)

        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 groups,
                 ratio=[1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       groups=groups[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).to(node_feat.device)
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0),
                                     torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)),
                                    0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(edge_feat.device)
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat

def pl_loss(feature, label):
    """
    Prototype loss
    """
    # 计算每种类别的中心
    unique_labels = torch.unique(label)
    center = torch.stack([feature[label == l].mean(dim=0) for l in unique_labels])

    # 将label映射到center的索引
    label_to_index = {l.item(): i for i, l in enumerate(unique_labels)}
    idx = torch.tensor([label_to_index[l.item()] for l in label], device=feature.device)

    # 根据索引获取每个样本对应的中心
    batch_center = center[idx]

    # 计算损失
    loss = ((feature - batch_center) ** 2).mean()
    return loss

class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0,
                 transductive=True):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.transductive = transductive

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            # nn.BatchNorm2d(8),
            # nn.GELU(),  # vit 用 relu
            PARN_SPAtt(64),
            PARN_CHAtt(64),
        )

        self.edge_loss = nn.BCELoss(reduction='none')

        self.node_loss = nn.CrossEntropyLoss(reduction='none')

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              groups=[1, 1] if l == 0 else [1],
                                              ratio=[4, 1] if l == 0 else [1],
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              groups=[1, 1],
                                              ratio=[2, 1],
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand
        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)
        return edge

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes).to("cuda")[class_idx].to("cuda")

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def forward(self, query_data, support_data, query_label, support_label, n_way, n_shot, tran=False):
        # Bs, Ms, _ = support_data.shape
        # support_data = support_data.view(support_data.shape[0] * support_data.shape[1], 64, 1, 64).transpose(1, -1)
        # support_data = self.conv(support_data).view(Bs, Ms, -1)
        #
        # Bq, Mq, _ = query_data.shape
        # query_data = query_data.view(query_data.shape[0] * query_data.shape[1], 64, 1, 64).transpose(1, -1)
        # query_data = self.conv(query_data).view(Bq, Mq, -1)

        if not tran:
            splice = support_data.shape[-1] // self.in_features
            rand = random.randint(0, splice - 1)
            query_data = query_data[:, :, rand * self.in_features:(rand + 1) * self.in_features]
            support_data = support_data[:, :, rand * self.in_features:(rand + 1) * self.in_features]

        # set as single data
        full_data = torch.cat([support_data, query_data], 1)
        full_label = torch.cat([support_label, query_label], 1)
        full_edge = self.label2edge(full_label)

        num_supports = support_data.shape[1]
        num_queries = query_data.shape[1]
        num_samples = num_supports + num_queries

        support_edge_mask = torch.zeros(query_data.shape[0], num_samples, num_samples).to(query_data.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        evaluation_mask = torch.ones(query_data.shape[0], num_samples, num_samples).to(query_data.device)

        # set init edge
        init_edge = full_edge.clone()  # batch_size x 2 x num_samples x num_samples
        init_edge[:, :, num_supports:, :] = 0.5
        init_edge[:, :, :, num_supports:] = 0.5
        for i in range(num_queries):
            init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
            init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

        # predict edge logit (consider only the last layer logit, num_tasks x 2 x num_samples x num_samples)
        full_logit_layers, node_feat_list = self.predict(node_feat=full_data, edge_feat=init_edge)
        full_logit = full_logit_layers[-1]

        node_pl_loss = 0
        for l in range(self.num_layers - 1):
            node_pl_loss = node_pl_loss + pl_loss(node_feat_list[l][:, :support_data.shape[1]].reshape(-1, self.node_features),
                                                  support_label.reshape(-1))

        # compute loss
        full_edge_loss_layers = [self.edge_loss((1 - full_logit_layer[:, 0]), (1 - full_edge[:, 0])) for
                                 full_logit_layer in full_logit_layers]

        # weighted edge loss for balancing pos/neg
        pos_query_edge_loss_layers = [
            torch.sum(full_edge_loss_layer * query_edge_mask * full_edge[:, 0] * evaluation_mask) / torch.sum(
                query_edge_mask * full_edge[:, 0] * evaluation_mask) for full_edge_loss_layer in full_edge_loss_layers]
        neg_query_edge_loss_layers = [
            torch.sum(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) / torch.sum(
                query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) for full_edge_loss_layer in
            full_edge_loss_layers]
        query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                  (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                  zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]

        # compute accuracy
        full_edge_accr_layers = [self.hit(full_logit_layer, 1 - full_edge[:, 0].long()) for full_logit_layer in
                                 full_logit_layers]
        query_edge_accr_layers = [torch.sum(full_edge_accr_layer * query_edge_mask * evaluation_mask) / torch.sum(
            query_edge_mask * evaluation_mask) for full_edge_accr_layer in full_edge_accr_layers]

        # # compute node loss & accuracy (num_tasks x num_quries x num_ways)
        # query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, num_supports:, :num_supports],
        #                                     self.one_hot_encode(n_way, support_label.long())) for
        #                           full_logit_layer in
        #                           full_logit_layers]  # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
        # query_node_accr_layers = [torch.eq(torch.max(query_node_pred_layer, -1)[1], query_label.long()).float().mean()
        #                           for query_node_pred_layer in query_node_pred_layers]

        total_loss_layers = query_edge_loss_layers

        # compute accuracy
        full_edge_accr = self.hit(full_logit, 1 - full_edge[:, 0].long())
        query_edge_accr = torch.sum(full_edge_accr * query_edge_mask * evaluation_mask) / torch.sum(
            query_edge_mask * evaluation_mask)

        # compute node accuracy (num_tasks x num_quries x num_ways)
        query_node_pred = torch.bmm(full_logit[:, 0, num_supports:, :num_supports],
                                    self.one_hot_encode(n_way, support_label.long()))  # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)

        # update model
        total_loss = []
        for l in range(self.num_layers - 1):
            total_loss += [total_loss_layers[l].view(-1) * 0.5]
        total_loss += [total_loss_layers[-1].view(-1) * 1.0]
        total_loss = torch.mean(torch.cat(total_loss, 0))
        total_loss = node_pl_loss + total_loss

        return query_node_pred, total_loss, query_edge_accr_layers[-1], query_edge_accr


    # forward
    def predict(self, node_feat, edge_feat):
        # for each layer
        node_feat_list = []
        edge_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)

            node_feat_list.append(node_feat)


        return edge_feat_list, node_feat_list

if __name__ == "__main__":
    emb_support = torch.zeros((8, 4, 8192))
    emb_query = torch.zeros((8, 40, 8192))

    labels_support = torch.zeros((8, 4))
    labels_query = torch.zeros((8, 40))

    labels_support[0] = 0
    labels_support[1] = 1
    labels_support[2] = 2
    labels_support[3] = 3
    labels_query[:10] = 0
    labels_query[10:20] = 1
    labels_query[20:30] = 2
    labels_query[30:] = 3

    model = GraphNetwork(in_features=8192, node_features=1024, edge_features=1024, num_layers=1)
    loss = model(emb_query, emb_support, labels_query, labels_support, 4, 1)