############### Train ############################
import os
import argparse
import numpy as np
import time

from torchvision import transforms, utils
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import sys
import random
from utils import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def train(model, train_loader, device, optimizer, betas, c_epoch, txtt, scheduler=None):
    model.train()
    train_loss = 0
    loss_dict = {}
    controler = 0

    print("Looks grate, go ahead!........................")
    for batch_idx, (a, b, c, d, e, lbl) in enumerate(train_loader):
        print(batch_idx + 1, end=", ", flush=True)
        a = a.to(device)
        b = b.to(device)
        c = c.to(device)
        d = d.to(device)

        if e is not None:
            e = e.to(device)
        if lbl is not None:
            lbl = lbl.to(device)

        optimizer.zero_grad(set_to_none=True)  # otherwise grads accumulate in backward

        if type(model) is SSAE or type(model) is SS_AEmvtec:
            loss, rec_im, loss_dict_new = model.step((a, b, c, d))

            loss.backward()
        elif (
            type(model) is SSVAE
            or type(model) is SS_CVAE
            or type(model) is SS_CVAEmvtec
        ):
            loss, rec_im, loss_dict_new = model.step((a, b, c, d, e))
        elif type(model) is DIS_SSVAE:
            loss, rec_im, loss_dict_new = model.step((a, b, c, d, e, lbl))

            (-loss).backward()
        train_loss += loss.item()

        if type(model) is DIS_SSVAE:
            txtt.write(
                f"{loss_dict_new['kld']},{loss_dict_new['beta*kld'].item()},{loss_dict_new['rec_term'].item()},{loss_dict_new['dis_loss'].item()},{loss_dict_new['loss'].item()}\n"
            )
            print(
                f"Ep: {c_epoch + 1} --> |kld:"
                f" {loss_dict_new['kld'].item()} |b*kld:"
                f" {loss_dict_new['beta*kld'].item()} |rec:"
                f" {loss_dict_new['rec_term'].item()} |dis_loss:{loss_dict_new['dis_loss'].item()}, |total: {loss_dict_new['loss'].item()}|"
            )
        else:
            txtt.write(
                f"{loss_dict_new['kld']},{loss_dict_new['beta*kld'].item()},{loss_dict_new['rec_term'].item()},{loss_dict_new['loss'].item()}\n"
            )
            print(
                f"Ep: {c_epoch + 1} --> |kld: {loss_dict_new['kld'].item()} |b*kld: {loss_dict_new['beta*kld'].item()} |rec: {loss_dict_new['rec_term'].item()} |total: {loss_dict_new['loss'].item()}|"
            )

        loss_dict = update_loss_dict(
            loss_dict, loss_dict_new
        )  # update_loss_dict need fixation
        optimizer.step()
        # scheduler.step()
        controler += 1

    train_loss /= controler
    loss_dict = {k: v / controler for k, v in loss_dict.items()}
    return train_loss, b, rec_im, loss_dict  # b is input modified image


def eval(model, test_loader, device, with_mask=False):
    model.eval()
    if with_mask:  # added to acommodate q(z|x,y) and q(x|y,zl)
        input_mb, gt_mb, lbl_mb = next(iter(test_loader))  # .next()
        input_mb = input_mb.to(device)
    else:
        input_mb, gt_mb = next(iter(test_loader))  # .next()
        input_mb = input_mb.to(device)
    if type(model) is SSAE or type(model) is SS_AEmvtec:
        recon_mb = model(input_mb)
    elif (
        type(model) is SS_CVAE
        or type(model) is SSVAE
        or type(model) is SS_CVAEmvtec
        or type(model) is DIS_SSVAE
    ):
        if with_mask:
            recon_mb, _ = model(input_mb)
        else:
            recon_mb, _ = model(input_mb)
        recon_mb = model.mean_from_lambda(recon_mb)
    return input_mb, recon_mb, gt_mb


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Cuda available ?", torch.cuda.is_available())
    print("Pytorch device:", device)
    seed = 11
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = load_ssae(args)
    model.to(device)

    # We get the train_dataset first and not directly the dataloaders for
    # further processing
    train_dataset = get_train_dataloader(args, return_dataset=True)
    test_dataset = get_test_dataloader(args, fake_dataset_size=16, return_dataset=True)
    if type(train_dataset) is ConcatDataset or type(train_dataset) is Dataset:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=12,
        )
    if type(test_dataset) is ConcatDataset or type(test_dataset) is Dataset:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            num_workers=12,
            drop_last=True,
        )

    nb_channels = args.nb_channels

    img_size = args.img_size
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    print(
        "Nb channels", nb_channels, "img_size", img_size, "mini batch size", batch_size
    )

    out_dir = args.dst_dir + "/torch_logs"  # './torch_logs'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    checkpoints_dir = args.dst_dir + "/torch_checkpoints"  # "./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
    res_dir = args.dst_dir + "/torch_results"  # './torch_results'
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    data_dir = args.dst_dir + "/torch_datasets"  # './torch_datasets'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    #     try:
    #         if not args.force_train:
    #             print(f'Force train enforced: {args.force_train}')
    #             #print(f'Force train no enforced: {args.force_train}')
    #             raise FileNotFoundError

    #             file_name = f"{args.exp}_{args.params_id}.pth"
    #             model = load_model_parameters(model, file_name, checkpoints_dir, device)
    #     except FileNotFoundError:
    print("Starting training")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_metric = open(f"{args.dst_dir}/torch_logs/step_metrics.txt", "a+")
    # step_metric.write("|||============================|||\n")
    step_metric.write("beta,kld, beta_kld,rec_term,total\n")

    ###############################################################################
    if "all" in args.data or len(args.data) >= 0:
        nsteps = sum(
            [len(loader) for loader in train_dataloader]
        )  # steps per per epoch
    else:
        nsteps = len(train_dataloader)
    ###############################################################################
    scale = (args.max_beta - args.min_beta) / (args.num_epochs * nsteps)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=nsteps * args.num_epochs, eta_min=1e-7
    )

    if args.anneal_beta:
        if args.anneal_cyclic:
            betas = computeLinearBeta(
                num_epochs=args.num_epochs,
                steps_per_epoch=nsteps,
                cycle=args.cycle,
                ratio=args.ratio,
            )
            assert (
                len(betas) == nsteps * args.num_epochs
            ), f"number of betas {len(betas)} and total training steps {nsteps*args.num_epochs} were not the same"
        else:
            betas = [np.round(i * scale, 4) for i in range(args.num_epochs * nsteps)]
    else:
        betas = [args.max_beta] * nsteps * args.num_epochs

    for epoch in range(args.num_epochs):
        if epoch == 0:
            if args.anneal_beta:
                print("Training will be progressing with kld annealing")
                if args.anneal_cyclic:
                    print(
                        "The kld annealing will follow cyclic annealing with linear increase strategy"
                    )
                else:
                    print("The kld annealing will follow a linear strategy")

            else:
                print(f"Training will be progressing with fixed beta: {args.max_beta}")

        print("Epoch", epoch + 1)

        loss, input_mb, recon_mb, loss_dict = train(
            model=model,
            train_loader=train_dataloader,
            device=device,
            optimizer=optimizer,
            betas=betas,
            c_epoch=epoch,
            txtt=step_metric,
            scheduler=lr_scheduler,
        )
        print(f"epoch [{epoch+1}/{args.num_epochs}], train loss: {round(loss,6)}")

        f_name = os.path.join(out_dir, f"{args.exp}_loss_values.txt")
        print_loss_logs(f_name, out_dir, loss_dict, epoch, args.exp)  #

        # save model parameters
        if (epoch + 1) % 100 == 0 or epoch in [0, 4, 9, 24]:
            # to resume a training optimizer state dict and epoch
            # should also be saved
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, f"{args.exp}_{epoch + 1}.pth"),
            )

            # print some reconstrutions
        if (epoch + 1) % 50 == 0 or epoch in [
            0,
            4,
            9,
            14,
            19,
            24,
            29,
            49,
        ]:
            img_train = utils.make_grid(
                torch.cat(
                    (
                        torch.flip(input_mb[:, :3, :, :], dims=(1,)),
                        torch.flip(recon_mb[:, :3, :, :], dims=(1,)),
                    ),
                    dim=0,
                ),
                nrow=batch_size,
            )
            utils.save_image(
                img_train, f"{res_dir}/{args.exp}_img_train_{epoch + 1}.png"
            )
            model.eval()
            input_test_mb, recon_test_mb, gts = eval(
                model=model,
                test_loader=test_dataloader,
                device=device,
                with_mask=args.with_mask,
            )
            img_test = utils.make_grid(
                torch.cat(
                    (
                        torch.flip(input_test_mb[:, :3, :, :], dims=(1,)),
                        torch.flip(recon_test_mb[:, :3, :, :], dims=(1,)),
                    ),
                    dim=0,
                ),
                nrow=batch_size_test,
            )
            utils.save_image(img_test, f"{res_dir}/{args.exp}_img_test_{epoch + 1}.png")

            model.to("cpu")  # move to CPU for processing the whole dataset

            idx_train_plot = np.random.choice(
                np.arange(len(train_dataset)), 200, replace=False
            )
            mu_train = model.encoder(
                torch.stack([train_dataset[i][1] for i in idx_train_plot], axis=0)
            )[0]
            # Plot the latent rv, all, constrained and unconstrained
            mu_train_cons = mu_train[:, : model.z_dim_constrained]
            mu_train_free = mu_train[:, model.z_dim_constrained :]
            mu_train_cons = (
                torch.reshape(mu_train_cons, (mu_train_cons.shape[0], -1))
                .detach()
                .numpy()
            )
            mu_train_free = (
                torch.reshape(mu_train_free, (mu_train_free.shape[0], -1))
                .detach()
                .numpy()
            )
            mu_train = torch.reshape(mu_train, (mu_train.shape[0], -1)).detach().numpy()
            mu_train_labels = (
                torch.stack([train_dataset[i][-1] for i in idx_train_plot], axis=0)
                .detach()
                .numpy()
            )

            idx_test_plot = np.random.choice(
                np.arange(len(test_dataset)), len(test_dataset), replace=False
            )
            mu_test = model.encoder(
                torch.stack([test_dataset[i][0] for i in idx_test_plot], axis=0)
            )[0]
            mu_test_cons = mu_test[:, : model.z_dim_constrained]
            mu_test_free = mu_test[:, model.z_dim_constrained :]
            mu_test_cons = (
                torch.reshape(mu_test_cons, (mu_test_cons.shape[0], -1))
                .detach()
                .numpy()
            )
            mu_test_free = (
                torch.reshape(mu_test_free, (mu_test_free.shape[0], -1))
                .detach()
                .numpy()
            )
            mu_test = torch.reshape(mu_test, (mu_test.shape[0], -1)).detach().numpy()
            mu_test_labels = (
                torch.stack([test_dataset[i][-1] for i in idx_test_plot], axis=0)
                .detach()
                .numpy()
                + model.nb_dataset  # current trick to differentiate
                # train from test
            )

            tsne = PCA(n_components=2)  # PCA instead of TSNE to go fast
            mu_train_embedded = tsne.fit_transform(mu_train)
            mu_test_embedded = tsne.transform(mu_test)
            mu_train_cons_embedded = tsne.fit_transform(mu_train_cons)
            mu_test_cons_embedded = tsne.transform(mu_test_cons)
            mu_train_free_embedded = tsne.fit_transform(mu_train_free)
            mu_test_free_embedded = tsne.transform(mu_test_free)

            fig, axes = plt.subplots(1, 3)
            axes[0].scatter(
                mu_train_embedded[:, 0],
                mu_train_embedded[:, 1],
                c=mu_train_labels
                # np.concatenate(
                #    [mu_train_embedded[:, 0], mu_test_embedded[:, 0]], axis=0
                # ),
                # np.concatenate(
                #    [mu_train_embedded[:, 1], mu_test_embedded[:, 1]], axis=0
                # ),
                # c=np.concatenate([mu_train_labels, mu_test_labels], axis=0),
            )
            axes[1].scatter(
                mu_train_cons_embedded[:, 0],
                mu_train_cons_embedded[:, 1],
                c=mu_train_labels
                # np.concatenate(
                #    [mu_train_cons_embedded[:, 0], mu_test_cons_embedded[:, 0]], axis=0
                # ),
                # np.concatenate(
                #    [mu_train_cons_embedded[:, 1], mu_test_cons_embedded[:, 1]], axis=0
                # ),
                # c=np.concatenate([mu_train_labels, mu_test_labels], axis=0),
            )
            axes[2].scatter(
                mu_train_free_embedded[:, 0],
                mu_train_free_embedded[:, 1],
                c=mu_train_labels
                # np.concatenate(
                #    [mu_train_free_embedded[:, 0], mu_test_free_embedded[:, 0]], axis=0
                # ),
                # np.concatenate(
                #    [mu_train_free_embedded[:, 1], mu_test_free_embedded[:, 1]], axis=0
                # ),
                # c=np.concatenate([mu_train_labels, mu_test_labels], axis=0),
            )
            # plt.show()

            idx_train_modified = np.random.choice(
                np.arange(len(train_dataset)), 10, replace=False
            )
            ori_img = torch.stack(
                [train_dataset[i][1] for i in idx_train_modified], axis=0
            )
            mu_train_modified = model.encoder(
                torch.stack([train_dataset[i][1] for i in idx_train_modified], axis=0)
            )[0]
            # Plot the latent rv, all, constrained and unconstrained
            # mu_train_modified[:, : model.z_dim_constrained] = torch.randn(
            #    mu_train_modified[:, : model.z_dim_constrained].shape) * 0

            b, c, h, w = mu_train_modified.shape
            mu_train_modified = mu_train_modified.reshape(
                (b, model.nb_dataset, model.z_dim_constrained, h, w)
            )
            for d in [1, 2]:  # range(model.nb_dataset):
                for z_d in range(model.z_dim_constrained):
                    mu_train_modified[:, d, z_d] = mu_train_modified[:, 0, z_d]
                    # mu_train_modified[:, d, z_d] = torch.mean(
                    #    torch.stack(
                    #        [
                    #            mu_train_modified[:, d_, z_d]
                    #            for d_ in range(model.nb_dataset)
                    #        ],
                    #        axis=1,
                    #    ),
                    #    axis=1,
                    # )
            mu_train_modified = mu_train_modified.reshape((b, -1, h, w))
            # mu_train_modified[:] = torch.mean(
            #        mu_train_modified[:], axis=1, keepdims=True
            # )
            rec_modified = model.mean_from_lambda(model.decoder(mu_train_modified))

            img_train = utils.make_grid(
                torch.cat(
                    (
                        torch.flip(ori_img[:, :3, :, :], dims=(1,)),
                        torch.flip(rec_modified[:, :3, :, :], dims=(1,)),
                    ),
                    dim=0,
                ),
                nrow=10,
            )
            utils.save_image(
                img_train, f"{res_dir}/{args.exp}_img_train_modified_{epoch + 1}.png"
            )

            model.to(device)  # move back to the training device
            model.train()

    step_metric.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
