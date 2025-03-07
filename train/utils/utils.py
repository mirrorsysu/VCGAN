import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from networks import network
from networks import pwcnet


def create_generator(opt):
    if opt.stage == "First":
        colorizationnet = network.FirstStageNet(opt)
        print("First stage Generator is created!")
    else:
        colorizationnet = network.SecondStageNet(opt)
        print("Second stage Generator is created!")

    if not opt.load_model:
        # Init the networks
        network.weights_init(colorizationnet, init_type=opt.init_type, init_gain=opt.init_gain)
        if opt.load_fe:
            pretrained_dict = torch.load(opt.fe_path)
            load_dict(colorizationnet.fenet, pretrained_dict, "fenet")
            load_dict(colorizationnet.fenet2, pretrained_dict, "fenet2")
            print("Generator is loaded [fenet:%s]" % (opt.fe_path))
        else:
            print("Generator without load fe")
    else:
        # Initialize the networks
        pretrained_dict = torch.load(opt.load_path)
        load_dict(colorizationnet, pretrained_dict, "colorizationnet")
        print("Generator is loaded! [load_path:%s]" % (opt.load_path))
    return colorizationnet


def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator70(opt)
    # Init the networks
    network.weights_init(discriminator, init_type=opt.init_type, init_gain=opt.init_gain)
    return discriminator


def create_pwcnet(opt):
    # Initialize the network
    flownet = pwcnet.PWCNet().eval()
    # Load a pre-trained network
    data = torch.load(opt.pwcnet_path)
    if "state_dict" in data.keys():
        flownet.load_state_dict(data["state_dict"])
    else:
        flownet.load_state_dict(data)
    print("PWCNet is loaded!")
    # It does not gradient
    for param in flownet.parameters():
        param.requires_grad = False
    return flownet


def create_perceptualnet(opt):
    # Get the first 15 layers of vgg16, which is conv4_3
    perceptualnet = network.PerceptualNet()
    if opt.load_perceptual:
        # Pre-trained VGG-16
        pretrained_dict = torch.load(opt.perceptual_path)
        load_dict(perceptualnet, pretrained_dict, "perceptualnet")
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet


def load_dict(process_net, pretrained_dict, label=""):
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict

    for k, _ in process_dict.items():
        if k not in pretrained_dict:
            print("[%s] not load layer: %s" % (label, k))

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net


class SubsetSeSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def create_dataloader(dataset, opt):
    return DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    if opt.pre_train:
        dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    else:
        # Generate random index
        indices = np.random.permutation(len(dataset))
        indices = np.tile(indices, opt.batch_size)
        # Generate data sampler and loader
        datasampler = SubsetSeSampler(indices)
        dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, sampler=datasampler, pin_memory=True)
    return dataloader


def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)


def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, "r")
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][: len(content[i]) - 1]
    file.close()
    return content


def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret


def get_dirs(path):
    # Read a folder, return a list of names of child folders
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split("\\")[-2]
            ret.append(a)
    return ret


def get_jpgs(path):
    # Read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret


def get_relative_dirs(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split("\\")[-2] + "/" + a.split("\\")[-1]
            ret.append(a)
    return ret


def text_save(content, filename, mode="a"):
    # Save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + "\n")
    file.close()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_sample_png(sample_folder, sample_name, img_list, name_list):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = (img * 128) + 128
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + "_" + name_list[i] + ".png"
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)
