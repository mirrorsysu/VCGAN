import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from utils import utils
from utils import dataset

import csv


def trainer_noGAN(opt):
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    opt.save_path.mkdir(parents=True, exist_ok=True)
    opt.sample_path.mkdir(parents=True, exist_ok=True)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))

    fieldnames = ["Epoch", "Batch", "Pixel-level Loss", "Perceptual Loss"]
    with open(opt.log_path, "w+") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Save the model if pre_train == True
    def save_model(opt, epoch, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = "First_Stage_epoch%d_bs%d.pth" % (epoch, opt.batch_size)
        save_name = os.path.join(opt.save_path, model_name)
        to_save = generator
        if opt.multi_gpu == True:
            to_save = generator.module
        torch.save(to_save.state_dict(), save_name)
        print("The trained model is saved as %s" % (model_name))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt, train=True)
    test_trainset = dataset.ColorizationDataset(opt, train=False)
    print("The overall number of images:", len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_trainset, batch_size=1, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    adjust_learning_rate(opt, opt.begin_epoch, optimizer_G)

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.begin_epoch, opt.epochs):
        for i, (true_L, true_RGB) in enumerate(dataloader):
            # To device
            true_L = true_L.cuda()
            true_RGB = true_RGB.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            fake_RGB = generator(true_L)

            # Pixel-level L1 Loss
            loss_L1 = criterion_L1(fake_RGB, true_RGB)

            # Percpetual Loss
            fake_feature = perceptualnet(fake_RGB)
            true_feature = perceptualnet(true_RGB)
            loss_percep = criterion_L1(fake_feature, true_feature)

            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_percep * loss_percep
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Pixel-level Loss: %.4f] [Perceptual Loss: %.4f] Time_left: %s"
                % (epoch, opt.epochs, i, len(dataloader), loss_L1.item(), loss_percep.item(), time_left)
            )

            with open(opt.log_path, "a") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writerow(dict(zip(fieldnames, [epoch, i, loss_L1.item(), loss_percep.item()])))
            """
            img_list = [fake_RGB, true_RGB]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = opt.sample_path, sample_name = 'iter%d' % (i + 1), img_list = img_list, name_list = name_list)
            """
        # Save model at certain epochs or iterations
        if epoch % opt.save_by_epoch == 0:
            save_model(opt, epoch, generator)

        # Learning rate decrease at certain epochs
        adjust_learning_rate(opt, (epoch + 1), optimizer_G)
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                (test_true_L, test_true_RGB) = next(test_dataloader.__iter__())
                test_true_L = test_true_L.cuda()
                test_fake_RGB = generator(test_true_L)
            img_list = [fake_RGB, true_RGB, test_fake_RGB, test_true_RGB]
            name_list = ["val_pred", "val_gt", "test_pred", "test_gt"]
            utils.save_sample_png(sample_folder=opt.sample_path, sample_name="epoch%d" % (epoch + 1), img_list=img_list, name_list=name_list)


def trainer_LSGAN(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(generator)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == "epoch":
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if opt.lr_decrease_mode == "iter":
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == "epoch":
            model_name = "First_Stage_epoch%d_bs%d.pth" % (epoch, opt.batch_size)
        if opt.save_mode == "iter":
            model_name = "First_Stage_iter%d_bs%d.pth" % (iteration, opt.batch_size)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == "epoch":
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
            if opt.save_mode == "iter":
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
        else:
            if opt.save_mode == "epoch":
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
            if opt.save_mode == "iter":
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt)
    print("The overall number of images:", len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_RGB) in enumerate(dataloader):

            # To device
            true_L = true_L[:, [0], :, :].cuda()
            true_RGB = true_RGB.cuda()

            # Adversarial ground truth
            valid = Tensor(np.ones((true_L.shape[0], 1, 30, 30)))
            fake = Tensor(np.zeros((true_L.shape[0], 1, 30, 30)))

            ### Train Discriminator
            optimizer_D.zero_grad()

            # Generator output
            fake_RGB = generator(true_L)

            # Fake colorizations
            fake_scalar_d = discriminator(true_L, fake_RGB.detach())
            loss_fake = criterion_MSE(fake_scalar_d, fake)

            # True colorizations
            true_scalar_d = discriminator(true_L, true_RGB)
            loss_true = criterion_MSE(true_scalar_d, valid)

            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_D.step()

            ### Train Generator
            optimizer_G.zero_grad()

            fake_RGB = generator(true_L)

            # Pixel-level L1 Loss
            loss_L1 = criterion_L1(fake_RGB, true_RGB)

            # Perceptual Loss
            feature_fake_RGB = perceptualnet(fake_RGB)
            feature_true_RGB = perceptualnet(true_RGB)
            loss_percep = criterion_L1(feature_fake_RGB, feature_true_RGB)

            # GAN Loss
            fake_scalar = discriminator(true_L, fake_RGB)
            loss_GAN = criterion_MSE(fake_scalar, valid)

            # Overall Loss and optimize
            loss_G = opt.lambda_l1 * loss_L1 + opt.lambda_gan * loss_GAN + opt.lambda_percep * loss_percep
            loss_G.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Pixel-level Loss: %.4f] [Perceptual Loss: %.4f] [D Loss: %.4f] [G Loss: %.4f] Time_left: %s"
                % ((epoch + 1), opt.epochs, i, len(dataloader), loss_L1.item(), loss_percep.item(), loss_D.item(), loss_GAN.item(), time_left)
            )

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_RGB, true_RGB]
            name_list = ["pred", "gt"]
            utils.save_sample_png(sample_folder=opt.sample_path, sample_name="epoch%d" % (epoch + 1), img_list=img_list, name_list=name_list)


def trainer_WGAN(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(generator)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == "epoch":
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if opt.lr_decrease_mode == "iter":
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == "epoch":
            model_name = "First_Stage_epoch%d_bs%d.pth" % (epoch, opt.batch_size)
        if opt.save_mode == "iter":
            model_name = "First_Stage_iter%d_bs%d.pth" % (iteration, opt.batch_size)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == "epoch":
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
            if opt.save_mode == "iter":
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
        else:
            if opt.save_mode == "epoch":
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
            if opt.save_mode == "iter":
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt)
    print("The overall number of images:", len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_RGB) in enumerate(dataloader):

            # To device
            true_L = true_L[:, [0], :, :].cuda()
            true_RGB = true_RGB.cuda()

            ### Train Discriminator
            optimizer_D.zero_grad()

            # Generator output
            fake_RGB = generator(true_L)

            # Fake colorizations
            fake_scalar_d = discriminator(true_L, fake_RGB.detach())

            # True colorizations
            true_scalar_d = discriminator(true_L, true_RGB)

            # Overall Loss and optimize
            loss_D = -torch.mean(true_scalar_d) + torch.mean(fake_scalar_d)
            loss_D.backward()
            optimizer_D.step()

            ### Train Generator
            optimizer_G.zero_grad()

            fake_RGB = generator(true_L)

            # Pixel-level L1 Loss
            loss_L1 = criterion_L1(fake_RGB, true_RGB)

            # Perceptual Loss
            feature_fake_RGB = perceptualnet(fake_RGB)
            feature_true_RGB = perceptualnet(true_RGB)
            loss_percep = criterion_L1(feature_fake_RGB, feature_true_RGB)

            # GAN Loss
            fake_scalar = discriminator(true_L, fake_RGB)
            loss_GAN = -torch.mean(fake_scalar)

            # Overall Loss and optimize
            loss_G = opt.lambda_l1 * loss_L1 + opt.lambda_gan * loss_GAN + opt.lambda_percep * loss_percep
            loss_G.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Pixel-level Loss: %.4f] [Perceptual Loss: %.4f] [D Loss: %.4f] [G Loss: %.4f] Time_left: %s"
                % ((epoch + 1), opt.epochs, i, len(dataloader), loss_L1.item(), loss_percep.item(), loss_D.item(), loss_GAN.item(), time_left)
            )

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_RGB, true_RGB]
            name_list = ["pred", "gt"]
            utils.save_sample_png(sample_folder=opt.sample_path, sample_name="epoch%d" % (epoch + 1), img_list=img_list, name_list=name_list)


def trainer_WGANGP(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(generator)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == "epoch":
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if opt.lr_decrease_mode == "iter":
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == "epoch":
            model_name = "First_Stage_epoch%d_bs%d.pth" % (epoch, opt.batch_size)
        if opt.save_mode == "iter":
            model_name = "First_Stage_iter%d_bs%d.pth" % (iteration, opt.batch_size)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == "epoch":
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
            if opt.save_mode == "iter":
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
        else:
            if opt.save_mode == "epoch":
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))
            if opt.save_mode == "iter":
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print("The trained model is saved as %s" % (model_name))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt)
    print("The overall number of images:", len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Calculate the gradient penalty loss for WGAN-GP
    def compute_gradient_penalty(D, input_samples, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(input_samples, interpolates)
        # For PatchGAN
        fake = Variable(Tensor(real_samples.shape[0], 1, 30, 30).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_RGB) in enumerate(dataloader):

            # To device
            true_L = true_L[:, [0], :, :].cuda()
            true_RGB = true_RGB.cuda()

            ### Train Discriminator
            optimizer_D.zero_grad()

            # Generator output
            fake_RGB = generator(true_L)

            # Fake colorizations
            fake_scalar_d = discriminator(true_L, fake_RGB.detach())

            # True colorizations
            true_scalar_d = discriminator(true_L, true_RGB)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, true_L.data, true_RGB.data, fake_RGB.data)

            # Overall Loss and optimize
            loss_D = -torch.mean(true_scalar_d) + torch.mean(fake_scalar_d) + opt.lambda_gp * gradient_penalty
            loss_D.backward()
            optimizer_D.step()

            ### Train Generator
            optimizer_G.zero_grad()

            fake_RGB = generator(true_L)

            # Pixel-level L1 Loss
            loss_L1 = criterion_L1(fake_RGB, true_RGB)

            # Perceptual Loss
            feature_fake_RGB = perceptualnet(fake_RGB)
            feature_true_RGB = perceptualnet(true_RGB)
            loss_percep = criterion_L1(feature_fake_RGB, feature_true_RGB)

            # GAN Loss
            fake_scalar = discriminator(true_L, fake_RGB)
            loss_GAN = -torch.mean(fake_scalar)

            # Overall Loss and optimize
            loss_G = opt.lambda_l1 * loss_L1 + opt.lambda_gan * loss_GAN + opt.lambda_percep * loss_percep
            loss_G.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Pixel-level Loss: %.4f] [Perceptual Loss: %.4f] [D Loss: %.4f] [G Loss: %.4f] Time_left: %s"
                % ((epoch + 1), opt.epochs, i, len(dataloader), loss_L1.item(), loss_percep.item(), loss_D.item(), loss_GAN.item(), time_left)
            )

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_RGB, true_RGB]
            name_list = ["pred", "gt"]
            utils.save_sample_png(sample_folder=opt.sample_path, sample_name="epoch%d" % (epoch + 1), img_list=img_list, name_list=name_list)
