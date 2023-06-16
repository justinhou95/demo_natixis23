from torch import nn
from tqdm import tqdm
from collections import defaultdict
from os import path as pt
import pickle
import torch
import matplotlib.pyplot as plt
    
class VAETrainer(nn.Module):
    def __init__(self, G, E, config, **kwargs):
        super(VAETrainer, self).__init__()
        self.losses_history = defaultdict(list)
        self.config = config
        self.beta = config.beta
        self.steps = 0
        # Decoder
        self.G = G 
        self.G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)
            )
        # Encoder
        self.E = E 
        self.E_optimizer = torch.optim.Adam(
            self.E.parameters(), lr=config.lr_E, betas=(0, 0.9)   
        )

        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=config.gamma
        )
        self.E_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.E_optimizer, gamma=config.gamma
        )


    def train(self, train_dl, epochs, device, plot_path = False):
        self.train_dl = train_dl
        self.device = device
        self.G.to(device)
        self.E.to(device)
        for epoch in tqdm(range(epochs)):
            print('Epoch: ', epoch)
            for data in tqdm(train_dl):
                R_loss, kl_loss = self.train_step(data)
                self.losses_history["R_loss"].append(R_loss)
                self.losses_history["kl_loss"].append(kl_loss)
                self.steps += 1
            print('R_loss: ',R_loss)
            print('KL: ', kl_loss)
            if epoch % 20 ==0: 
                self.G_lr_scheduler.step()
                self.E_lr_scheduler.step()
                self.record_parameter()
                if plot_path :
                    with torch.no_grad():
                        self.z_fake = torch.randn_like(self.z_real)
                        self.x_fake = self.G(self.z_fake)
                    self.plot_real_recons_fake(self.x_real.detach().to('cpu'), self.y_real.detach().to('cpu'), self.x_fake.to('cpu'))  
        filepath = pt.join(self.config.exp_dir, "losses_history.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.losses_history, f)

    def train_step(self, data):
        self.G_optimizer.zero_grad()
        self.E_optimizer.zero_grad()
        self.x_real = data[0].to(self.device)
        self.z_real, self.kl = self.E(self.x_real)
        self.y_real = self.G(self.z_real)
        # self.R_loss = torch.square(self.x_real - self.y_real).mean(0).sum()
        self.R_loss = torch.abs(self.x_real - self.y_real).mean(0).sum()        
        self.loss = self.R_loss + self.beta * self.kl
        self.loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.G_optimizer.step()
        self.E_optimizer.step()
        return self.R_loss.item(), self.kl.item()
    
    def generate_all(self, x_reals):
        with torch.no_grad():
            z_reals = self.E(x_reals)[0]
            y_reals = self.G(z_reals)
            z_fakes = torch.randn_like(z_reals)
            x_fakes = self.G(z_fakes)
        return x_reals, y_reals, x_fakes
    
    def record_parameter(self):
        file_path = pt.join(self.config.exp_dir, "parameter" +str(self.steps) + ".pth")
        torch.save(self.state_dict(), file_path)

    def plot_real_recons_fake(self, x_real, y_real, x_fake):
        x_real_dim = x_real.shape[-1]
        for i in range(x_real_dim):
            fig, ax = plt.subplots(1,3, figsize = [16,4], sharex=True, sharey=True)
            ax[0].plot(x_real[..., i].T, alpha=0.3)
            ax[1].plot(y_real[..., i].T, alpha=0.3)
            ax[2].plot(x_fake[..., i].T, alpha=0.3)
            plt.savefig(pt.join(self.config.exp_dir, "x_real_y_real_x_fake_dim" + str(i) + "_" +str(self.steps) + ".png"))
            plt.close(fig)


def print_end():
    print('Thank you for you attention! Paper is coming :)')
