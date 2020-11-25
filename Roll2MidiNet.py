import torch.nn as nn
import torch.nn.functional as F
import torch
##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size, 0.8))
        model.append(nn.LeakyReLU(0.2))
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out


class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        channels, _ , _ = input_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 1024, dropout=0.5)
        self.down6 = UNetDown(1024, 1024, dropout=0.5)

        self.up1 = UNetUp(1024, 512, dropout=0.5)
        self.up2 = UNetUp(1024+512, 256, dropout=0.5)
        self.up3 = UNetUp(512+256, 128, dropout=0.5)
        self.up4 = UNetUp(256+128, 64)
        self.up5 = UNetUp(128+64, 16)
        self.conv1d = nn.Conv2d(80, 1, kernel_size=1)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)

        d2 = self.down2(d1)

        d3 = self.down3(d2)

        d4 = self.down4(d3)

        d5 = self.down5(d4)

        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)

        u2 = self.up2(u1, d4)

        u3 = self.up3(u2, d3)

        u4 = self.up4(u3, d2)

        u5 = self.up5(u4, d1)

        out = self.conv1d(u5)

        out = F.sigmoid(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape #1 51 50

        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(height / 2 ** 3)+1, int(width / 2 ** 3)+1
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    input_shape = (1,51, 100)
    gnet = Generator(input_shape)
    dnet = Discriminator(input_shape)
    print(dnet.output_shape)
    imgs = torch.rand((64,1,51,100))
    gen = gnet(imgs)
    print(gen.shape)
    dis = dnet(gen)
    print(dis.shape)

