import os
from torchvision import transforms
from torchvision.utils import save_image
from models.SFSR import generator
from models.VAE import *
from PIL import Image

ori = None


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def test(base_path):
    img_name = '1.png'
    img_path = 'G:\\csproject\\faceSR\\images\\1.png'
    device = torch.device("cuda")
    x_image = Image.open(img_path)
    # image = np.asarray(x_image)
    # x_image = load_image(img_path)
    x_transform1=transforms.Compose( [
        transforms.Resize(32),
        transforms.CenterCrop(32),
    ] )

    x_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # torch.Size([3, 32, 32])
    x_image = x_transform1(x_image)
    # out_img = transforms.ToPILImage()(x_image)
    # x_image.save(os.path.join('./test/', 'lr32_' + img_path.split('/')[-1][-10:]),quality=95)
    x_image = x_transform(x_image)

    x_image = x_image.unsqueeze(0).to(device)

    with torch.no_grad():
        # save_enc_model = "./save_model/save_model-256-32/" + "enc.pth"
        # save_gen_model = "./save_model/save_model-256-32/" + "gen.pth"
        save_enc_model = base_path + '/pretrained_models/enc.pth'
        save_gen_model = base_path + '/pretrained_models/gen.pth'
        encoder = gaussian_resnet_encoder(image_size=32).to(device)
        G = generator().to(device)
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
    x_image[0] *= 255
    output[0] *= 255
    out_name = os.path.join(base_path,'results',img_name[:-4] + '_pred' + '.jpg')
    print(output.shape)
    save_image(out_name, output[0])


# test('./','./test/012796_256.jpg')
test('./')
