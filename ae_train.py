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
from utils import * 



def train(model, train_loader, device, optimizer, epoch):

    model.train()
    train_loss = 0
    loss_dict = {}
    controler = 0

    for batch_idx, (a, b, c, d) in enumerate(train_loader):
        print(batch_idx + 1, end=", ", flush=True)
        a = a.to(device) 
        b = b.to(device)
        c = c.to(device)
        d = d.to(device)
        optimizer.zero_grad(set_to_none=True)   # otherwise grads accumulate in backward

        loss, rec_im, loss_dict_new = model.step(
            (a, b, c, d)
        )


        if type(model) is SSAE:
            loss.backward()
        elif type(model) is SSVAE:
            (-loss).backward()
        train_loss += loss.item()
        loss_dict = update_loss_dict(loss_dict, loss_dict_new)   # update_loss_dict need fixation
        optimizer.step()
        controler+=1
    
    train_loss /= controler
    loss_dict = {k:v / controler for k, v in loss_dict.items()}
    return train_loss, b, rec_im, loss_dict  # b is input modified image

def eval(model, test_loader, device):
    model.eval()
    input_mb, gt_mb = iter(test_loader).next()
    gt_mb = gt_mb.to(device)
    input_mb = input_mb.to(device)
    if type(model) is SSAE:
        recon_mb = model(input_mb)
    elif type(model) is SSVAE:
        recon_mb, _ = model(input_mb)
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
        fake_dataset_size=4
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
        
        
    try:
        if args.force_train:
            raise FileNotFoundError
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir, checkpoints_saved_dir, device)
    except FileNotFoundError:
        print("Starting training")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr)
        for epoch in range(args.num_epochs):
            print("Epoch", epoch + 1)     
            loss,input_mb, recon_mb, loss_dict = train(model=model,
                    train_loader=train_dataloader,
                    device=device,
                    optimizer=optimizer,
                    epoch=epoch)
            print(f'epoch [{epoch+1}/{args.num_epochs}], train loss: {round(loss,4)}')

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
                        torch.cat((
                            torch.flip(input_mb[:, :3, :, :], dims=(1,)),
                            torch.flip(recon_mb[:, :3, :, :], dims=(1,)),
                        ), dim=0), nrow=batch_size
                    )
                utils.save_image(
                        img_train,
                        f"{res_dir}/{args.exp}_img_train_{epoch + 1}.png"
                    )  # f"torch_results/{args.exp}_img_train_{epoch + 1}.png"
                model.eval()
                input_test_mb, recon_test_mb, gts = eval(model=model,
                                                             test_loader=test_dataloader,
                                                             device=device)

                model.train()
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

if __name__ == "__main__":
    args = parse_args()
    main(args)
