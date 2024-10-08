import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, HGTConv, TransformerConv, Linear, SimpleConv 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, roc_curve
from torch.nn import ReLU, Sigmoid, Softmax
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np


class MiLk-FD(torch.nn.Module):
    def __init__(self, hgraph, hidden_channels, out_channels, num_layers, num_heads):
        super().__init__()

        # nlinear transform
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in hgraph.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
     
        for i in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, hgraph.metadata(), num_heads, group='sum')
            # conv = HGTConv(hidden_channels, hidden_channels, hgraph.metadata(), num_heads, 'mean')
            # conv = HGTConv(hidden_channels, hidden_channels, hgraph.metadata(), num_heads, group='max')
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)
        self.lin = Linear(hidden_channels, out_channels)

        # self.sigmoid = Sigmoid()

    def forward(self, x_dict, edge_index_dict):
    # def forward(self, x_dict, edge_index_dict, kg_dict, kg_edge_dict):

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
            # x_dict[node_type] = F.silu(self.lin_dict[node_type](x))


        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # for conv in self.convs_kg:
        #     kg_dict = conv(kg_dict,kg_edge_dict)
        #     kg_dict = {key: x.relu() for key, x in kg_dict.items()}

        # x_dict['news'] = x_dict['news'] + kg_dict['news']

        # out = self.sigmoid(self.lin1(x_dict["news"]))
        return self.lin(x_dict['news'])

class Aggregation(torch.nn.Module):
    def __init__(self, ent_nums, hidden_channels=128):
        super().__init__()

        self.num = ent_nums

        self.project = Linear(-1, hidden_channels)
        self.project1 = Linear(-1, hidden_channels)
        self.atten = Linear(ent_nums, hidden_channels)

    def forward(self, kg, kg1):
        kg = self.project(kg)
        kg1 = self.project1(kg1)

        kg_attr = torch.add(kg, kg1) / self.num
        # print(kg_attr.shape)

        return kg_attr


def train(model, data, args):
    model.train()
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        # out = model(data.x_dict, data.edge_index_dict, kg.x_dict, kg.edge_index_dict)
        mask = data['news'].train_mask
        #print(out[mask].shape, data['news'].y[mask])
        loss = criterion(out[mask], data['news'].y[mask])
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()

def test(model, data, args):
# def test(model, data, kg, args):
    # news add
    # model.eval()

    out = model(data.x_dict, data.edge_index_dict)
    # out = model(data.x_dict, data.edge_index_dict, kg.x_dict, kg.edge_index_dict)
    pred = out[data['news'].test_mask].argmax(dim=1).cpu()
    # pred = out[data['news'].test_mask].argmax(dim=-1).cpu()

    y = data['news'].y[[data['news'].test_mask]].cpu()
    # pred_list = out[data['news'].test_mask].tolist()
    # predict = []
    def softmax(p):
        e_x = torch.exp(p)
        partition_x = e_x.sum(1, keepdim=True)
        return e_x / partition_x
    predict = softmax(out[data['news'].test_mask])
    col, row = predict.shape
    # print(col)
    pred_list = []
    for i in range(col):
        pred_list.append(predict[i][1].cpu().tolist())
    pred_list = torch.Tensor(pred_list)

    # plus = []
    # for i in predict:
    #     x = i[0] + i[1]
    #     plus.append(x)


    # revised "average="
    # acc = accuracy_score(y, pred)
    # precision = precision_score(y, pred, average='micro')
    # f1 = f1_score(y, pred, average='micro')
    # recall = recall_score(y, pred, average='micro')
    # auc = roc_auc_score(y, pred, average='micro')
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, )
    f1 = f1_score(y, pred)
    recall = recall_score(y, pred,)
    # f1_1 = f1_score(y, pred, average='weighted')
    # f1_2  = f1_score(y, pred, pos_label=0)

    auc = roc_auc_score(y, pred,)
    print(f"Testing Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},F1: {f1:.4f}")
    # with open("./parameter_2.result", "a", encoding="utf8") as f:
    #     f.write(f"dim0: {args.hidden_channels}; epoch: {args.epochs}; Acc:{acc:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1: {f1:.4f} \n")


    # draw ROC curve 
    # fpr, tpr, thresholds = roc_curve(y, pred_list)
    # np.savetxt(".//ROC_CURVE/MiLK-FD_FPR_{}.txt".format(args.dataset), fpr)
    # np.savetxt("./ROC_CURVE/MiLK-FD_TPR_{}.txt".format(args.dataset), tpr)
    # # np.loadtxt()
    # # print(fpr)
    # roc_auc = sm.auc(fpr, tpr)

    # plt.figure()
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='red', lw=lw, label="ROC curve (area = %0.3f)" % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])

    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')


    # plt.title('Receiver operating characteristic curve(ROC)_{}'.format(args.dataset))
    # plt.legend(loc="lower right")
    # # plt.savefig('./ROC/roc_curve_{}_{}.png'.format(args.dataset, i))
    # plt.savefig('./ROC_CURVE/roc_curve_{}.png'.format(args.dataset))
    # # plt.show()
    
