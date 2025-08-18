from typing import Optional, Any, Callable, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import argparse
from tqdm import tqdm
from tkinter import Tk
from tkinter import filedialog
import pandas as pd
import json

#------- My modules --------
import plots as pl

# -------------------------------------
#        1. PREPARE AND LOAD DATA     #
# -------------------------------------
class AutoEncDataLoader(Dataset):
    def __init__(self,
                 args: argparse.Namespace) -> None:
        self.args = args
        self.train_path = args.train_path
        self.batch_size = args.batch_size
        self.validation_size = args.validation_size
        self.img_list = [
            f for f in os.listdir(args.train_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transforms = transforms.Compose([
            transforms.Resize(args.img_resize),
            transforms.ToTensor(), # Rearranges shape from [H, W, C] (height, width, channels) to [C, H, W].
            # transforms.Normalize([0.5]*3, [0.5]*3),  # activate this if you want to standardize [-1, 1] for RGB
            transforms.Lambda(lambda x: x.view(-1)) # to flatten step so the data is [channels * height * width]
        ])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.train_path, self.img_list[index])
        img = Image.open(img_path).convert("RGB")
        return self.transforms(img)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        print(f"\nloading data from:\n{self.train_path}\n")
        validation_size = int(self.validation_size * len(self))
        train_size = len(self) - validation_size

        # ** split and load **
        train_dataset, validation_dataset = random_split(self, [train_size, validation_size])  
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        if self.args.dataloader_verbose == 1:  
            print(f"\n{'='*100}")
            print(f"Total images in dataset: {len(self)}")
            print(f"Images in train set: {len(train_dataset)}")
            print(f"Images in validation set: {len(validation_dataset)}")
            print(f"Train batches: {len(train_loader)}")
            print(f"Validation batches: {len(validation_loader)}")

            for imgs in train_loader:
                print(f"\ntrain_loader batch dimensions: {imgs.shape}, should be like --> ([batch_size, flatten_tensor(1D)])")
                break

            for imgs in validation_loader:
                print(f"\nvalidation_loader batch dimensions: {imgs.shape}, should be like --> ([batch_size, flatten_tensor(1D)])")
                break

            if self.args.debug:
                raise SystemExit(f'\nExecution stopped intentionally\n{"="*100}\n')
        
        return train_loader, validation_loader
    
# -------------------------------------
#          2. My Autoencoder         #
# -------------------------------------
class MyAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, out_activation="sigmoid", use_bn=True):
        super().__init__()
        encoder_dims = [input_dim] + hidden_layers + [latent_dim]
        decoder_dims = [latent_dim] + hidden_layers[::-1] + [input_dim]
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.out_activation = out_activation
        self.use_bn = use_bn
        
        # *** Encoder *** #
        self.encoder_W = nn.ParameterList()
        self.encoder_b = nn.ParameterList()
        self.encoder_bn = nn.ModuleList() if use_bn else None
       
        for i in range(len(encoder_dims) - 1):
            self.encoder_W.append(nn.Parameter(torch.empty(encoder_dims[i+1], encoder_dims[i])))
            self.encoder_b.append(nn.Parameter(torch.zeros(encoder_dims[i+1])))
            if use_bn:
                self.encoder_bn.append(nn.BatchNorm1d(encoder_dims[i+1]))
            nn.init.kaiming_normal_(self.encoder_W[-1], nonlinearity="relu")

        # *** Decoder *** #
        self.decoder_W = nn.ParameterList()
        self.decoder_b = nn.ParameterList()
        self.decoder_bn = nn.ModuleList() if use_bn else None
        
        for i in range(len(decoder_dims) - 1):
            self.decoder_W.append(nn.Parameter(torch.empty(decoder_dims[i+1], decoder_dims[i])))
            self.decoder_b.append(nn.Parameter(torch.zeros(decoder_dims[i+1])))
            if use_bn and i < len(decoder_dims)-2:
                self.decoder_bn.append(nn.BatchNorm1d(decoder_dims[i+1]))

            if i == len(decoder_dims) - 2 and self.out_activation in ("sigmoid", "tanh"):
                nn.init.xavier_uniform_(self.decoder_W[-1])
            else:
                nn.init.kaiming_normal_(self.decoder_W[-1], nonlinearity="relu")

    def encode(self, x):
        for i in range(len(self.encoder_W)):
            x = torch.matmul(x, self.encoder_W[i].T) + self.encoder_b[i]
            if self.use_bn:
                x = self.encoder_bn[i](x)
            x = self.relu(x) 
        return x
    
    def decode(self, z):
        for i in range(len(self.decoder_W)):
            z = torch.matmul(z, self.decoder_W[i].T) + self.decoder_b[i]

            if i < len(self.decoder_W)-1:
                if self.use_bn:
                    z = self.decoder_bn[i](z)
                z = self.relu(z)
            else:
                z = self.sigmoid(z) if self.out_activation == "sigmoid" else self.tanh(z)
        return z
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
        
# ---------------------------------------------------------
#        3.Model Operations : Training, Validation, Test.
#----------------------------------------------------------
class AutoencoderOps:
    def __init__(self,
                 args: argparse.Namespace):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.loader = AutoEncDataLoader(args=args)
        self.input_dim = self.args.img_resize[0] * self.args.img_resize[1] * 3
        self.early_stop_counter = 0
        self.start_epoch = 0
        self.best_epoch = 0
    
        # Inicialize Models.
        self.model = MyAutoencoder(input_dim=self.input_dim,
                                   hidden_layers=self.args.hidden_layers,
                                   latent_dim=self.args.latent_dim).to(self.device)

        # Loss function and optimizers.
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        #Best loss.
        self.best_loss = float('inf')

        # Loss hystory.
        self.history = {
            "training_loss": [],
            "validation_loss": []
        }
    #------------------------------------------------ Training --------------------------------------------------#
    def training(self):
        train_loader, validation_loader = self.loader.load_data()
        self.model.train()

        if self.args.resume_training:
            ask_best_model_path = filedialog.askopenfile(title="Select the weights of the best model to resume the training!")
            best_model_path = ask_best_model_path.name
            self.resume_training(best_model_path)
            print(f"\nResuming training from epoch {self.start_epoch}...\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            training_loss = 0.0

            for batch_idx, batch_imgs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")):
                batch_imgs = batch_imgs.to(self.device)
                outputs = self.model(batch_imgs)
                loss = self.criterion(outputs, batch_imgs)

                if self.args.dataloader_verbose == 2:  
                    print(f"\n{'='*100}\nTraining verbose:\n")
                    print(f"\ninput imgs (1st img, 10 pxls): {batch_imgs[0][:10]}")
                    print(f"\nreconstructed imgs (1st img, 10 pxls): {outputs[0][:10]}")
                    print("\nInput min/max:", batch_imgs.min().item(), batch_imgs.max().item())
                    print("Output min/max:", outputs.min().item(), outputs.max().item())
                    
                    # to visualize the reconstructed imgs during training.
                    pl.imgs_visualization(batch_imgs,
                                          outputs,
                                          batch_idx,
                                          self.args.epochs,
                                          epoch,
                                          self.args.img_resize,
                                          len(train_loader),
                                          tanh=False)
                    
                    if self.args.debug:
                        raise SystemExit(f'\nExecution stopped intentionally\n{"="*100}\n')

                self.optimizer.zero_grad()  
                loss.backward() 
                self.optimizer.step()
                training_loss += loss.item()

            train_epoch_loss = training_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Trainig loss: {train_epoch_loss:.6f}")
            self.history["training_loss"].append(train_epoch_loss)

            self.validation(validation_loader, epoch)
            
            if self.early_stop_counter >= self.args.early_stop:
                    print(f'Early stopping in epoch: {epoch + 1}')
                    break
        
        print(f"\n{'-' * 90}\n")

    #---------------------------------------------- Validation --------------------------------------------------#
    def validation(self, validation_loader, epoch):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch_imgs in enumerate(tqdm(validation_loader, desc="Validating")):
                batch_imgs = batch_imgs.to(self.device)
                outputs = self.model(batch_imgs)
                loss = self.criterion(outputs, batch_imgs)
                val_loss += loss.item()

                if self.args.dataloader_verbose == 4:  
                    print(f"\n{'='*100}\nValidation verbose:\n")
                    print(f"\ninput imgs (1st img, 10 pxls): {batch_imgs[0][:10]}")
                    print(f"\nreconstructed imgs (1st img, 10 pxls): {outputs[0][:10]}")
                    print("\nInput min/max:", batch_imgs.min().item(), batch_imgs.max().item())
                    print("Output min/max:", outputs.min().item(), outputs.max().item())
                    
                    # to visualize the reconstructed imgs during training.
                    pl.imgs_visualization(batch_imgs,
                                          outputs,
                                          batch_idx,
                                          self.args.epochs,
                                          epoch,
                                          self.args.img_resize,
                                          len(validation_loader),
                                          tanh=False)
                    
                    if self.args.debug:
                        raise SystemExit(f'\nExecution stopped intentionally\n{"="*100}\n')
        
            val_epoch_loss = val_loss/len(validation_loader)
            self.history["validation_loss"].append(val_epoch_loss)

            checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_loss': self.best_loss,
                    'history':self.history,
                    'early_stop_counter': self.early_stop_counter,
                    'hyperparams': {
                        'input_dim': self.input_dim,
                        'hidden_layers': self.args.hidden_layers,
                        'latent_dim': self.args.latent_dim,
                        'out_activation': getattr(self.model, 'out_activation', 'sigmoid'), # (obj, attr_name, default)
                        'use_bn': getattr(self.model, 'use_bn', True),
                    }
                }
            
            last_file_name = "last_checkpoint.pth"
            last_log_name = "last_checkpoint_info.json"
            torch.save(self.model.state_dict(), os.path.join(self.args.save_best_model, last_file_name))
            torch.save(checkpoint, os.path.join(self.args.save_best_model, last_file_name))
            
            checkpoint_log = checkpoint.copy()
            checkpoint_log.pop('model_state_dict', None)
            checkpoint_log.pop('optimizer_state_dict', None)
            with open(os.path.join(self.args.save_best_model, last_log_name), "w") as log_file:
                json.dump(checkpoint_log, log_file, indent=4)

            if val_epoch_loss < self.best_loss:
                print(f'\n{"=" * 80}\nNew best model found in epoch: {epoch + 1}\n{"=" * 80}\n')
                self.best_loss = val_epoch_loss
                # file_name = f'Best loss_{self.best_loss:.4f}, founded in epoch_{epoch + 1}.pth'
                best_file_name = "best_checkpoint.pth"
                best_log_name = "best_checkpoint_info.json"
                torch.save(self.model.state_dict(), os.path.join(self.args.save_best_model, best_file_name))
                torch.save(checkpoint, os.path.join(self.args.save_best_model, best_file_name))
                
                with open(os.path.join(self.args.save_best_model, best_log_name), "w") as log_file:
                    json.dump(checkpoint_log, log_file, indent=4)

                self.best_epoch = epoch + 1

                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        
        print(f"Validation loss: {val_epoch_loss:.4f}\n\n{'-' * 90}\n\n")

    #------------------------------------------------ Test --------------------------------------------------#
    def test(self):
        # ** apply the same transform applied for training. ** #
        test_transforms = transforms.Compose([
            transforms.Resize(self.args.img_resize),
            transforms.ToTensor(), # Rearranges shape from [H, W, C] (height, width, channels) to [C, H, W].
            # transforms.Normalize([0.5]*3, [0.5]*3),  # activate this if you want to standardize [-1, 1] for RGB
            transforms.Lambda(lambda x: x.view(-1)) # to flatten step so the data is [channels * height * width]
        ])
     
        try:
            ask_best_model_path = filedialog.askopenfile(title="Select the weights of the best model to start the test!")
            best_model_path = ask_best_model_path.name
            checkpoint = torch.load(best_model_path, weights_only=True)
            hparams = checkpoint["hyperparams"]
            model = MyAutoencoder(**hparams).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model = model
            print(f"\nbest model weights loaded succesfully!...")
        except FileNotFoundError:
            print(f"\nError: Could not find the best model file\n{self.args.load_best_model}")
            return  
        
        self.model.eval()

        valid_ext = (".jpg", ".jpeg", ".png")
        try:
            test_folder = filedialog.askdirectory(title="select a folder of imgs for testing the autoencoder")
            print(f"\nSelected test folder: {test_folder}")
            if not test_folder:
                print("\nNo test folder selected!, exiting...")
                return
        except FileNotFoundError:
            print(f"\nError: could not find the images folder\n")
            return
         
        # ** import GTs imgs, transform and store them. ** #
        print(f"\nProcessing images!...")
        test_imgs = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(valid_ext)])
        imgs_path = [os.path.join(test_folder, img) for img in test_imgs]
        imgs_tensors = [test_transforms(Image.open(img).convert("RGB")) for img in imgs_path]
        reconstructed = []
        results = []

        inputs = {}
        for img in imgs_path:
            inputs[os.path.basename(img)] = test_transforms(Image.open(img).convert("RGB")) 
        
        with torch.no_grad():
            for img_name, img_tensor in inputs.items():
                img_batch = img_tensor.unsqueeze(0).to(self.device)
                out_batch = self.model(img_batch)
                recon = out_batch.squeeze(0).cpu()
                reconstructed.append(recon)
                score = F.mse_loss(recon, img_tensor, reduction="mean").item()
                # scores.append(score)
                pred_label = "editada" if score > 0.04 else "real"
                results.append({
                    "image": img_name,
                    "prediction": pred_label,
                    "score": score  
                })
        
        results_df = pd.DataFrame(results)
    
        if self.args.write_results:
            try:
                gt_df_path = filedialog.askopenfile(title="Select the CSV file with the GT labels.")
                gt_df_path = gt_df_path.name
                if not gt_df_path:
                    print("\nNo csv file was selected!, exiting...")
                    return
                print(f"\nGTs dataset: {os.path.basename(gt_df_path)} opened succesfully!")
            except FileNotFoundError:
                print(f"\nCould not found the GT labels csv file!")

            gt_labels_df = pd.read_csv(gt_df_path)

            if gt_labels_df.empty:
                print("\nCouldn't load the csv file!")
                return
            
            # ** merge GTs & results datasets ** #
            merged_df = gt_labels_df[['image', 'label']].merge(
                        results_df[['image', 'prediction', 'score']],
                        on="image",
                        how="left"
                    )
           
            print(f"\n{'-' * 50}")
            print("\n              === RESULTS ===\n")
            print(f"n{merged_df.head(10)}\n{'-' * 50}")
            merged_df.to_csv(gt_df_path, index=False)
            print(f"\nDataset: {os.path.basename(gt_df_path)} exported succesfully in:\n{os.path.dirname(gt_df_path)}")
            pl.plot_distributions(gt_df_path)
        
        #** Plot GT vs reconstructed imgs ** #
        if self.args.show_reconstructed:
            batches = len(test_imgs) // self.args.show_n_imgs # count the batches for plotting.
            leftover = len(test_imgs) % self.args.show_n_imgs # to calculate the images in the last batch.
        
            imgs_batches = {}
            if batches > 0:
                for b in range(batches):
                    start = self.args.show_n_imgs * b
                    end = start + self.args.show_n_imgs 
                    imgs_batches[f"B{b + 1}"] = {"Inputs":imgs_tensors[start : end], 
                                                 "Outputs": reconstructed[start : end]} 
            if leftover > 0:
                start = batches * self.args.show_n_imgs
                imgs_batches[f"B{batches + 1}"] = {"Inputs" : imgs_tensors[start:],
                                                   "Outputs": reconstructed[start:]}
                
            for batch_name, v in imgs_batches.items():
                n_in = len(v["Inputs"])
                n_out = len(v["Outputs"])
                print(f"{batch_name} --> inputs: {n_in} imágenes, outputs: {n_out} imágenes")

            if not test_imgs:
                print("\nNo valid images found in the selcted folder!")
                return
        
            pl.visualize_test_imgs(imgs_batches,
                                   imgs_n_imgs=self.args.show_n_imgs,
                                   img_resize=self.args.img_resize)
            
    def resume_training(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        hparams = checkpoint["hyperparams"]
        self.model = MyAutoencoder(**hparams).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.start_epoch = checkpoint.get("epoch", 0)
        self.early_stop_counter = checkpoint.get("early_stop_counter", 0)
        self.history = checkpoint.get("history", {
                                                    "training_loss": [],
                                                    "validation_loss": []
                                                })
    #------------------------------------------------ Main --------------------------------------------------#
    def main(self):
        if self.args.train:
            self.training()
            # to the last epoch.
            pl.plot_train_val_losses(history=self.history,
                                     model_name="My_autoencoder_last_checkpoint",
                                     path=self.args.stats_path,
                                     show=True)
            
            history_best = {
                "training_loss": self.history["training_loss"][:self.best_epoch],
                "validation_loss": self.history["validation_loss"][:self.best_epoch]
            }
            # to the best epoch
            pl.plot_train_val_losses(history=history_best,
                                     model_name="My_autoencoder_best_checkpoint",
                                     path=self.args.stats_path,
                                     show=False)
        else:
            print(f"\n{'*' * 90}\nAutoencoder test mode activated!")
            self.test()
            print(f"\nTest finished! exiting...\n{'*' * 90}\n")

if __name__=='__main__':
#------------------------------------------ HYPERPARAMETERS ------------------------------------------ #
    def autoencoders_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Autoencoders arguments.")
        parser.add_argument('--train', type=bool, default=True, help='Decide between Train or Test the model.')
        parser.add_argument('--resume_training', type=bool, default=False, help='Decide between Train or Test the model.')
        parser.add_argument('--plot_results', type=bool, default=True, help='plot metrics results after test.')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
        parser.add_argument('--validation_size', type=float, default=0.10, help='Portion of data samples used for validation.')
        parser.add_argument('--img_width', type=int, default=1280, help='Image size (width).')
        parser.add_argument('--img_height', type=int, default=720, help='Image size (height).')
        parser.add_argument('--img_resize', type=tuple, default=(128, 128), help='Image dimensions (width and height) for RGB.')
        parser.add_argument('--hidden_layers', type=list, default=[2048,1024, 512, 256, 128], help='dimensions in the hidden layers.')
        parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of bottleneck.')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training.')
        parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
        parser.add_argument("--early_stop", type=int, default=400,help="Number of epochs to wait for improvement before stopping training")
        
        #----- Input datasets path -----#
        parser.add_argument('--train_path', type=str,
                            default='./datasets/train_and_test/autoencoder/normal/',
                            help='Directory containing the images.')

        #----- Save Best models path -----#
        parser.add_argument('--save_best_model', type=str,
                            default='./best_models/my_autoencoder/',
                            help='Path to load the generator model with the best loss obtained during training.')
        
        #----- Load Best models path -----#
        parser.add_argument('--load_best_model', type=str,
                            default='./best_models/my_autoencoder/Best loss_0.0349, founded in epoch_5.pth',
                            help='Path to load the generator model with the best loss obtained during training.')
    
        #----- Stats path -----#
        parser.add_argument('--stats_path', type=str,
                            default='./stats/my_autoencoder/',
                            help='Path to save the Discriminator loss graph of fake and real loss obtained during training.')
        
        #----- Visualization and Debbug -----#
        parser.add_argument('--dataloader_verbose',
                            type=int,
                            default=None,
                            help="""Visualization of the code process depending on the choice:
                                    
                                    dataloader_verbose = 1 ------> Iput data description in load_data function.
                                    dataloader_verbose = 2 ------> Iput data description in load_data function.

                                 
                                 """)
        parser.add_argument('--debug', type=bool, default=False, help='Stop the run of the code for visualization and debug.')
        parser.add_argument('--write_results', type=bool, default=False, help='If True the scores and anomaly results will be written in the GTs dataset.')
        parser.add_argument('--show_reconstructed', type=bool, default=True, help='If True imgs_visualization function will be activated to show reconstructed imgs.')
        parser.add_argument('--show_n_imgs', type=int, default=6, help='number of imgs to show with imgs_visualization function.')
        args = parser.parse_args([])
        return args
    
    args = autoencoders_args()

    my_autoencoder = AutoencoderOps(args=args)
    my_autoencoder.main()


 
