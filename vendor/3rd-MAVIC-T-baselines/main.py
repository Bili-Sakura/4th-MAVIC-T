import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import os
import cv2
import glob
import random
import matplotlib.pyplot as plt



class CustomDataset(Dataset):
    def __init__(self, root_dir="/Volumes/Storage/train/", transform=None , train=True, input_domain="SAR", output_domain="RGB"):
        """
        :param root_dir: Directory pointing to root training data folder
        :param transform: List of transforms to be applied to each image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.input_domain = input_domain
        self.output_domain = output_domain
        self.rgb_images = []
        self.ir_images = []
        self.sar_images = []

        if self.train:
            for root, dirs, files in os.walk(self.root_dir):
                for dir in dirs:
                    rgb_images = glob.glob(os.path.join(root, dir) + '/*rgb.tiff')
                    ir_images = glob.glob(os.path.join(root, dir) + '/*ir.tiff')
                    sar_images = glob.glob(os.path.join(root, dir) + '/*UMBRA*.tiff')

                    if len(rgb_images) > 0 and len(ir_images) > 0 and len(sar_images) > 0:
                        self.rgb_images.append(rgb_images)
                        self.ir_images.append(ir_images)
                        self.sar_images.append(sar_images)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        input_image = np.empty([])
        output_image = np.empty([])
        # Randomly sample from each index
        rgb_sample = random.randint(0, len(self.rgb_images[idx]) - 1)
        ir_sample = random.randint(0, len(self.ir_images[idx]) - 1)
        sar_sample = random.randint(0, len(self.sar_images[idx]) - 1)

        rgb_image = cv2.imread(self.rgb_images[idx][rgb_sample])
        ir_image = cv2.imread(self.ir_images[idx][ir_sample])
        sar_image = cv2.imread(self.sar_images[idx][sar_sample])

        if self.transform:
            rgb_image = self.transform(rgb_image)
            ir_image = self.transform(ir_image)
            sar_image = self.transform(sar_image)

        if self.input_domain == "SAR":
            input_image = sar_image
        elif self.input_domain == "IR":
            input_image = ir_image
        elif self.input_domain == "RGB":
            input_image = rgb_image

        if self.output_domain == "SAR":
            output_image = sar_image
        elif self.output_domain == "IR":
            output_image = ir_image
        elif self.output_domain == "RGB":
            output_image = rgb_image

        return (input_image, output_image)




# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define your dataset and dataloader (adjust path and transformations accordingly)
# Example of using the custom dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1024, 1024), antialias=True),

])

dataset = CustomDataset(root_dir="/Volumes/Storage/train/", transform=transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        input_image, target_image = batch
        target_image = Variable(target_image).to(device)
        input_image = Variable(input_image).to(device)

        # Train discriminator with real images
        discriminator.zero_grad()
        real_labels = Variable(torch.ones((target_image.size(0)), 1, 64, 64)).to(device)
        real_outputs = discriminator(target_image)
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_real.backward()

        # Train discriminator with fake images
        fake_labels = Variable(torch.zeros((target_image.size(0)), 1, 64, 64)).to(device)
        fake_images = generator(input_image)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()
        discriminator_optimizer.step()

        # Train generator
        generator.zero_grad()
        g_loss = criterion(discriminator(fake_images), real_labels)
        g_loss.backward()
        generator_optimizer.step()



        # Plot the images
        if step % 100 == 0:
            # Convert images to NumPy arrays for visualization
            real_images_np = target_image[0].squeeze().permute(1, 2, 0).numpy()
            fake_images_np = fake_images[0].detach().squeeze().permute(1, 2, 0).numpy()
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.title('Input (Real)')
            plt.imshow(real_images_np)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Generated')
            plt.imshow(fake_images_np)
            plt.axis('off')

            plt.show()


        # Print losses
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

# Save the trained generator model
torch.save(generator.state_dict(), 'pix2pix_generator.pth')
