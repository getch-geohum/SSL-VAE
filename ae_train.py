############### Train ############################
import os
import argparse
import numpy as np
import time

from torchvision import transforms, utils
import torch
from torch import nn

import matplotlib
matplotlib.use('Agg')
import sys
import random 
from utils import * 


def train(model, train_loader, device, optimizer, betas, c_epoch, txtt,scheduler=None):
    random.shuffle(train_loader)
    model.train()
    train_loss = 0
    loss_dict = {}
    controler = 0
    step_per_epoch = sum([len(dloader) for dloader in train_loader])

    print("Looks grate, go ahead!........................")
    print(f'Got {len(train_loader)} dataaloaders for training')
    for _train_loader in train_loader:
        for batch_idx, (a, b, c, d, e) in enumerate(_train_loader): # xm/x, Mm, Mn, prob
            # print(f"Shape of a is: {a.shape}")
            print(batch_idx + 1, end=", ", flush=True)
            a = a.to(device) 
            b = b.to(device)
            c = c.to(device)
            d = d.to(device)
            e = e.to(device)

            optimizer.zero_grad(set_to_none=True)   # otherwise grads accumulate in backward

            #loss, rec_im, loss_dict_new = model.step(
            #    (a, b, c, d, e), beta=beta)

            if type(model) is SSAE or type(model) is SS_AEmvtec:
                loss, rec_im, loss_dict_new = model.step(
                (a, b, c, d))

                loss.backward()
            elif type(model) is SSVAE or type(model) is SS_CVAE or type(model) is SS_CVAEmvtec:
                step_beta = betas[(c_epoch)*step_per_epoch + controler] # betas[c_epoch*len(train_loader)+batch_idx]  # betas[(c_epoch-1)*step_per_epoch + controler] 
                loss, rec_im, loss_dict_new = model.step(
                (a, b, c, d, e), beta=step_beta)

                (-loss).backward()
            train_loss += loss.item()

            txtt.write(f"{step_beta},{loss_dict_new['kld']},{loss_dict_new['beta*kld'].item()},{loss_dict_new['rec_term'].item()},{loss_dict_new['loss'].item()}\n")
            print(f"Ep: {c_epoch} --> |beta: {step_beta} |kld: {loss_dict_new['kld'].item()} |b*kld: {loss_dict_new['beta*kld'].item()} |rec: {loss_dict_new['rec_term'].item()} |total: {loss_dict_new['loss'].item()}|")

            loss_dict = update_loss_dict(loss_dict, loss_dict_new)   # update_loss_dict need fixation
            optimizer.step()
            scheduler.step()
            controler+=1
    
    train_loss /= controler
    loss_dict = {k:v / controler for k, v in loss_dict.items()}
    return train_loss, b, rec_im, loss_dict  # b is input modified image

def eval(model, test_loader, device, with_mask=False):
    model.eval()
    if with_mask:  # added to acommodate q(z|x,y) and q(x|y,zl)
        input_mb, gt_mb = next(iter(test_loader)) # .next()
        input_mb = input_mb.to(device)
        # msk_mb = msk_mb.to(device)
        # gt_mb = gt_mb.to(device)
    else:
        input_mb, gt_mb = next(iter(test_loader)) # .next()
        # gt_mb = gt_mb.to(device)
        input_mb = input_mb.to(device)
    if type(model) is SSAE or type(model) is SS_AEmvtec:
        recon_mb = model(input_mb)
    elif type(model) is SS_CVAE or type(model) is SSVAE or type(model) is SS_CVAEmvtec:
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

    train_dataloader = get_train_dataloader(args)
    test_dataloader = get_test_dataloader(
        args,
        fake_dataset_size=16
    )

    nb_channels = args.nb_channels

    img_size = args.img_size
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    print("Nb channels", nb_channels, "img_size", img_size, 
        "mini batch size", batch_size)

    out_dir = args.dst_dir + '/torch_logs' # './torch_logs' 
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    checkpoints_dir = args.dst_dir + '/torch_checkpoints' # "./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
    res_dir = args.dst_dir + '/torch_results'  # './torch_results'
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    data_dir = args.dst_dir + '/torch_datasets' # './torch_datasets'
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr)
    step_metric = open(f'{args.dst_dir}/torch_logs/step_metrics.txt', 'a+')
    #step_metric.write("|||============================|||\n")
    step_metric.write("beta,kld, beta_kld,rec_term,total\n")
    
    ###############################################################################
    if 'all' in args.data or len(args.data)>=0:
        nsteps = sum([len(loader) for loader in train_dataloader])  # steps per per epoch
    else:
        nsteps = len(train_dataloader)
    ###############################################################################
    scale = (args.max_beta - args.min_beta) / (args.num_epochs*nsteps)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=nsteps*args.num_epochs, eta_min=1e-7)

    if args.anneal_beta:
        if args.anneal_cyclic:
            betas = computeLinearBeta(num_epochs=args.num_epochs, steps_per_epoch=nsteps, cycle=args.cycle, ratio=args.ratio)
            assert len(betas) == nsteps*args.num_epochs, f'number of betas {len(betas)} and total training steps {nsteps*args.num_epochs} were not the same'
        else:
            betas = [np.round(i*scale, 4) for i in range(args.num_epochs*nsteps)]
    else:
        betas = [args.max_beta]*nsteps*args.num_epochs

    for epoch in range(args.num_epochs):
        if epoch == 0:
            if args.anneal_beta:
                print('Training will be progressing with kld annealing')
                if args.anneal_cyclic:
                    print('The kld annealing will follow cyclic annealing with linear increase strategy')
                else:
                    print('The kld annealing will follow a linear strategy')

            else:
                print('Training will be progressing with fixed beta: {args.max_beta}')

        print("Epoch", epoch + 1)


        loss,input_mb, recon_mb, loss_dict = train(model=model,
                train_loader=train_dataloader,
                device=device,
                optimizer=optimizer,
                betas=betas,
                c_epoch=epoch,
                txtt=step_metric,
                scheduler=lr_scheduler)
        print(f'epoch [{epoch+1}/{args.num_epochs}], train loss: {round(loss,6)}')

        f_name = os.path.join(out_dir, f"{args.exp}_loss_values.txt")
        print_loss_logs(f_name, out_dir, loss_dict, epoch, args.exp)   # 

            # save model parameters
        if (epoch + 1) % 100 == 0 or epoch in [0, 4, 9, 24]:
                # to resume a training optimizer state dict and epoch
                # should also be saved
            torch.save(model.state_dict(), os.path.join(
                    checkpoints_dir, f"{args.exp}_{epoch + 1}.pth"
                    )
                )

            # print some reconstrutions
        if (epoch + 1) % 50 == 0 or epoch in [0, 4, 9, 14, 19, 24, 29, 49]:   # check this part
            img_train = utils.make_grid(
                #tensor_img_to_01(
                    torch.cat((
                        torch.flip(input_mb[:, :3, :, :], dims=(1,)),
                        torch.flip(recon_mb[:, :3, :, :], dims=(1,))),
                        dim=0),
                nrow=batch_size
            )
            utils.save_image(
                    img_train,
                    f"{res_dir}/{args.exp}_img_train_{epoch + 1}.png"
                )  # f"torch_results/{args.exp}_img_train_{epoch + 1}.png"
            model.eval()
            input_test_mb, recon_test_mb, gts = eval(model=model,
                                                         test_loader=test_dataloader[0],
                                                         device=device, with_mask=args.with_mask)

            model.train()
            #print(input_test_mb.shape, 'input test shape')
            #print(recon_test_mb.shape,'reconstructed test image shape')

            img_test = utils.make_grid(
                    torch.cat((
                        torch.flip(input_test_mb[:, :3, :, :], dims=(1,)),
                        torch.flip(recon_test_mb[:, :3, :, :], dims=(1,))),
                        dim=0),
                        nrow=batch_size_test
                )
            utils.save_image(
                    img_test,
                    f"{res_dir}/{args.exp}_img_test_{epoch + 1}.png"  
                )  # f"torch_results/{args.exp}_img_test_{epoch + 1}.png"
    step_metric.close()
if __name__ == "__main__":
    args = parse_args()
    main(args)
