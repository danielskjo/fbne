import argparse
from math import sqrt
import os
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data

from FoldedEncoder import FoldedEncoder
from GraphRec import GraphRec
from UVAggregator import UVAggregator
from UVEncoder import UVEncoder


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
            epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0

    return 0


def test(model, device, test_loader, epoch):
    model.eval()
    tmp_pred = []
    target = []

    with torch.no_grad():
        f = open(f"epoch/{epoch}.txt", "a")

        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            val_output = torch.clamp(val_output, min=0, max=4)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

            for i in range(len(test_u)):
                f.write(f"{test_u[i]} {test_v[i]} {val_output[i]} {tmp_target[i]}\n")

        f.write("\n")

    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)

    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')  # 32, 64, 128
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')  # .001, 0.0005, .0001
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    with open('data/list.pkl', 'rb') as f:
        """
        history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
        history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

        train_u, train_v, train_r: training_set (user, item, rating)
        test_u, test_v, test_r: testing set (user, item, rating)
        """
        # Users' history of purchased items
        history_u_lists = pickle.load(f)

        # Users' ratings on their purchased items
        history_ur_lists = pickle.load(f)

        # Items' history of users who purchased it
        history_v_lists = pickle.load(f)

        # Item's ratings by the users who purchased it
        history_vr_lists = pickle.load(f)
        walks_u = pickle.load(f)
        walks_v = pickle.load(f)
        train_u = pickle.load(f)
        train_v = pickle.load(f)
        train_r = pickle.load(f)
        test_u = pickle.load(f)
        test_v = pickle.load(f)
        test_r = pickle.load(f)
        # social_adj_lists = pickle.load(f)
        ratings_list = pickle.load(f)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    print("number of users, items, ratings: ", num_users, num_items, num_ratings)

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    agg_u_history = UVAggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True).to(device)
    enc_u_history = UVEncoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True).to(device)
    enc_u = FoldedEncoder(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, 5, walks_u, cuda=device).to(device)

    agg_v_history = UVAggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False).to(device)
    enc_v_history = UVEncoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False).to(device)
    enc_v = FoldedEncoder(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, 5, walks_v, cuda=device).to(device)

    graphrec = GraphRec(enc_u, enc_v).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(graphrec, device, test_loader, epoch)

        # TODO: Add validation set to tune hyper parameters
        # early stopping (no validation set)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 10:
            break


if __name__ == "__main__":
    main()
