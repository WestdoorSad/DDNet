# -*- coding: utf-8 -*-
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from thop import profile

from models.BERT.bert import BERT
from models.CL.data2vec import Data2Vec, contrastive_loss
from models.GN.gnn_head import GraphNetwork

import numpy as np
from utils import (
    set_gpu,
    Timer,
    count_accuracy,
    check_dir,
    log,
    calculate_accuracy_each_snr,
    strip_thop_from_state_dict,
)

from data.RML201610A import (
    RML201610A_Dataset,
    RML201610A_EVAL_CLASSES,
    FewShotDataloader,
    Pretrain_RML201610A_Dataset,
)


def get_model():
    network = BERT(64, 3, 4).cuda()
    cls_head = GraphNetwork(
        in_features=64 * 16,
        node_features=256,
        edge_features=256,
        num_layers=2,
        dropout=0.1,
    ).cuda()
    return network, cls_head


def get_dataset():
    dataset_train = RML201610A_Dataset(phase='train')
    dataset_val = RML201610A_Dataset(phase='val')
    return dataset_train, dataset_val, FewShotDataloader


def run_data2vec_pretrain(data2vec_model, embedding_net, loader_pretrain, optimizer, save_path,
                          pre_epochs=20, checkpoint_interval=10):
    for i in range(pre_epochs):
        num_total = 0
        running_loss = 0
        with tqdm(total=len(loader_pretrain)) as t:
            for batch_x, batch_next, batch_rand, batch_labels in loader_pretrain:
                batch_x = batch_x.to("cuda")
                batch_next = batch_next.to("cuda")
                batch_labels = batch_labels.to("cuda")

                batch_size = batch_x.size(0)
                num_total += batch_size

                x, y = data2vec_model(batch_x, batch_x)
                mae_loss = F.mse_loss(x, y) * 2

                bs_k, _ = data2vec_model.encoder.pre_forward(batch_next)
                bs_k_reg = data2vec_model.contrastive_reg_head_k(bs_k)
                with torch.no_grad():
                    bs_q, _ = data2vec_model.ema.model.pre_forward(batch_x)
                bs_q_reg = data2vec_model.contrastive_reg_head_q(bs_q)
                con_loss = contrastive_loss(bs_k_reg, bs_q_reg) * 0.8

                bs_k = torch.swapaxes(bs_k, 1, 2)
                cls_loss = F.cross_entropy(
                    data2vec_model.cls(bs_k[batch_labels != -1]),
                    batch_labels[batch_labels != -1],
                )

                loss = con_loss + cls_loss + mae_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                data2vec_model.ema_step()

                running_loss += batch_size * loss.item()
                t.set_description(f"[{i + 1}/{pre_epochs}]")
                t.set_postfix(
                    total_train_loss=running_loss / num_total,
                    mae_loss=mae_loss.item(),
                    cls_loss=cls_loss.item(),
                    con_loss=con_loss.item(),
                )
                t.update(1)

        torch.save(
            strip_thop_from_state_dict(embedding_net.state_dict()),
            os.path.join(save_path, "pre_last.pth"),
        )
        if checkpoint_interval and i % checkpoint_interval == 0:
            torch.save(
                strip_thop_from_state_dict(embedding_net.state_dict()),
                os.path.join(save_path, "pre_{}.pth".format(i)),
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='frequency of model saving')
    parser.add_argument('--train-query', type=int, default=5,
                        help='number of query examples per training class')
    parser.add_argument('--val-query', type=int, default=5,
                        help='number of query examples per validation class')
    parser.add_argument('--train-episode', type=int, default=500,
                        help='number of episodes per training')
    parser.add_argument('--val-episode', type=int, default=1000,
                        help='number of episodes per validation')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--train-way', type=int, default=4,
                        help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=4,
                        help='number of classes in one test (or validation) episode')
    parser.add_argument('--train-shot', type=int, default=1,
                        help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                        help='number of support examples per validation class')
    parser.add_argument('--episodes-per-batch', type=int, default=32,
                        help='number of episodes per batch')

    opt = parser.parse_args()

    dataset_train, dataset_val, data_loader = get_dataset()

    loader_pretrain = torch.utils.data.DataLoader(
        dataset=Pretrain_RML201610A_Dataset(),
        batch_size=512,
        shuffle=True,
        num_workers=4,
    )

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=2,
        epoch_size=opt.episodes_per_batch * opt.train_episode,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model()

    from torchsummary import summary
    summary(embedding_net, (1, 2, 128))

    macs, params = profile(embedding_net, inputs=(torch.Tensor(torch.randn(1, 1, 2, 128)).to(device="cuda"),))
    print(
        "Param: %.2fK | FLOPs: %.3fM" % (params / (1000 ** 1), macs / (1000 ** 2))
    )

    load_checkpoint = 0
    if load_checkpoint:
        embedding_net.load_state_dict(torch.load(os.path.join(opt.save_path, 'pre_last.pth')))
        print('load checkpoint')
    data2vec_model = Data2Vec(embedding_net, embed_dim=64).cuda()

    optimizer = torch.optim.SGD(data2vec_model.parameters(), lr=0.1, momentum=0.9, \
                                weight_decay=5e-5, nesterov=True)
    pre_epochs = 20
    run_data2vec_pretrain(
        data2vec_model,
        embedding_net,
        loader_pretrain,
        optimizer,
        opt.save_path,
        pre_epochs=pre_epochs,
        checkpoint_interval=10,
    )

    if pre_epochs > 0 or load_checkpoint:
        # 冻结embeding编码层
        for param in embedding_net.embedding.parameters():
            param.requires_grad = False
        embedding_net.position_embeddings.requires_grad = False
        for param in embedding_net.transformer_blocks.parameters():
            param.requires_grad = False


    # 重置优化器参数
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                weight_decay=2e-4, nesterov=True)

    lambda_epoch = lambda e: 1.0 if e < 10 else (0.84 if e < 20 else (0.12 if e < 40 else 0.024 if e < 50 else (0.0048)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        train_accuracies = []
        train_losses = []

        with tqdm(total=len(dloader_train(epoch))) as t:
            for i, batch in enumerate(dloader_train(epoch)):
                data_support, labels_support, data_query, labels_query, _, _, lbl_support, lbl_query = [x.cuda() for x in batch]

                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

                logit_query, loss, _, query_edge_accr = cls_head(
                    emb_query, emb_support, labels_query, labels_support, opt.train_way, opt.train_shot
                )
                con_loss = loss

                acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

                train_accuracies.append(acc.item())
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                append = " EdgeAccr: {:.2f} %, con_loss: {:.2f}".format(query_edge_accr * 100, con_loss * 1000)
                train_acc_avg = np.mean(np.array(train_accuracies))
                t.set_description(
                    f"Train Epoch: {epoch}\t [{i}/{len(dloader_train)}]\t Loss: {loss.item():.4f}\t"
                    f"Accuracy: {train_acc_avg:.2f} % ({acc:.2f} %)" + append
                )
                t.update(1)


        # Evaluate on the validation split
        val_accuracies = []
        val_losses = []

        val_true_labels = []
        val_pred_labels = []
        val_lbls = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, Kall, _, lbl_support, lbl_query = [x.cuda() for x in batch]

                test_n_support = opt.test_way * opt.val_shot
                test_n_query = opt.test_way * opt.val_query


                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query, loss, _, query_edge_accr = cls_head(
                    emb_query, emb_support, labels_query, labels_support, opt.test_way, opt.val_shot
                )

                acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
                pred_query = torch.argmax(logit_query.reshape(-1, opt.test_way), dim=1).reshape(-1)
                Kall = Kall.reshape(-1).detach()
                val_pred_labels.append(Kall[pred_query])
                val_true_labels.append(Kall[labels_query.reshape(-1)])
                val_lbls.append(lbl_query.reshape(-1))

                val_accuracies.append(acc.item())
                val_losses.append(loss.item())

        val_true_labels = torch.cat(val_true_labels).detach().cpu().numpy()
        val_pred_labels = torch.cat(val_pred_labels).detach().cpu().numpy()
        val_lbls = torch.cat(val_lbls).detach().cpu().numpy()

        calculate_accuracy_each_snr(val_true_labels, val_pred_labels, val_lbls, list(RML201610A_EVAL_CLASSES), epoch)

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        append = " EdgeAccr: {:.2f} %".format(query_edge_accr * 100)

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save(
                {
                    'embedding': strip_thop_from_state_dict(embedding_net.state_dict()),
                    'head': strip_thop_from_state_dict(cls_head.state_dict()),
                },
                os.path.join(opt.save_path, 'best_model.pth'),
            )
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95) + append)
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95) + append)

        torch.save(
            {
                'embedding': strip_thop_from_state_dict(embedding_net.state_dict()),
                'head': strip_thop_from_state_dict(cls_head.state_dict()),
            },
            os.path.join(opt.save_path, 'last_epoch.pth'),
        )

        if epoch % opt.save_epoch == 0:
            torch.save(
                {
                    'embedding': strip_thop_from_state_dict(embedding_net.state_dict()),
                    'head': strip_thop_from_state_dict(cls_head.state_dict()),
                },
                os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)),
            )

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
