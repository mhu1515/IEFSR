import os
from glob import glob

from torchvision import transforms
from torchvision.utils import save_image
from model.model import generator
from predataset.VAE import *
from PIL import Image


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def test(img_path):
    device = torch.device("cuda")
    x_image = Image.open(img_path)

    x_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    # torch.Size([3, 32, 32])
    x_image = x_transform(x_image)
    x_image = x_image.unsqueeze(0).to(device)

    with torch.no_grad():
    
        save_enc_model = "./save_model/" + "9_enc.pth"
        save_gen_model = "./save_model/" + "9_gen.pth"

        encoder = gaussian_resnet_encoder(image_size=32).to(device)
        encoder = nn.DataParallel(encoder)
        # cudnn.benchmark = True
        G = generator().to(device)
        G = nn.DataParallel(G)
        state_enc_dict = torch.load(save_enc_model)
        state_gen_dict = torch.load(save_gen_model)
        encoder.load_state_dict(state_enc_dict)
        G.load_state_dict(state_gen_dict)
        encoder.eval()
        G.eval()
        z, mu, logvar = encoder(x_image)
        output = G(x_image, mu)
        x_image = x_image.cpu()
        # torch.Size([1, 3, 256, 256])
        output = output.cpu()
        # print(output)
    x_image[0] *= 255
    output[0] *= 255
    # print(output)
    out_name = './test_results/sr_' + img_path.split('/')[-1].split('\\')[-1]
    print(out_name)
    save_image(out_name, output[0])


def test_encoder(img_path):
    device = torch.device("cuda")
    x_image = Image.open(img_path)

    x_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    # torch.Size([3, 32, 32])
    x_image = x_transform(x_image)
    x_image = x_image.unsqueeze(0).to(device)

    with torch.no_grad():
        save_enc_model = "./save_model/save_model_ep10_dis/" + "enc.pth"
        encoder = gaussian_resnet_encoder(image_size=32).to(device)
        state_enc_dict = torch.load(save_enc_model)
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_enc_dict.items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            encoder.load_state_dict(new_state_dict)
            encoder.eval()
            z, mu, logvar = encoder(x_image)
        except Exception as e:
            print(e)

all_list = glob('./test_2_32/*.*')

for i in all_list:
    test(i)
