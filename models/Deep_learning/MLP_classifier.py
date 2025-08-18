import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import pygad
from tkinter import Tk, filedialog
from tqdm import tqdm
import json
import os
import argparse

# ===== my modules ====== #
import utils as ut
import dl_plots as dlpl

device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing GPU: {torch.cuda.get_device_name(device)}\n") if torch.cuda.is_available() else print("\nUsing CPU\n")

# ============= MLP model structure ============
class MLPBinaryClassifier(nn.Module):
    def __init__(self,
                 args: argparse.Namespace):
        super().__init__()
        self.args = args
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leakyrelu": lambda: nn.LeakyReLU(0.1),
            "tanh": nn.Tanh
        }
        self.act_name = self.args.hidden_activation.lower()
        act = acts[self.act_name]
        layers = []
        prev = self.args.input_dim
        
        for h in self.args.hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if self.args.dropout and self.args.dropout > 0.0:
                layers.append(nn.Dropout(self.args.dropout))
            prev = h
        
        self.last_linear = nn.Linear(prev, self.args.output_dim)
        layers.append(self.last_linear)
        self.network = nn.Sequential(*layers)
        self.network.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m is self.last_linear:
                if self.args.last_init == "xavier":
                    init.xavier_uniform_(m.weight)
                else:
                    init.kaiming_normal_(m.weight)
            else:
                if self.act_name in ("relu", "leakyrelu", "gelu"):
                    negative_slope = 0.1 if self.act_name == "leakyrelu" else 0.0
                    init.kaiming_normal_(m.weight, a=negative_slope, nonlinearity="relu")
                elif self.act_name == "tanh":
                    init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                else:
                    init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        
        elif isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)
    
# ============= MLP Genetic Algorithms ============
class HypmOptGa():
    def __init__(self, args):
        self.args = args

    def ga_fitness_func(self,
                        ga_instance,
                        solution,
                        solution_idx):
        """
        solution: chromosome or array with hyperparameters 
        that we want to optimize.
        """
        
        # --- assing genes to variables ---
        hidden_units_0 = int(solution[0])
        hidden_units_1 = int(solution[1])
        hidden_units_2 = int(solution[2])
        hidden_units_3 = int(solution[3])
        dropout = float(solution[4])
        lr = float(solution[5])

        # --- encode the hidden acts, pygad.GA only accepts float, int and None ---
        activation_map = {0: "relu", 1: "gelu", 2: "leakyrelu"}
        activation_idx = solution[6]
        hidden_activation = activation_map[activation_idx]

        # --- encode the last initialization, pygad.GA only accepts float, int and None ---
        last_init_map = {0: "xavier", 1: "kaiming"}
        last_init_idx = solution[7]
        last_init = last_init_map[last_init_idx]

        # --- loader args ---
        validation_size = float(solution[8])
        test_size = float(solution[9])
        batch_size = int(solution[10])

        # --- create the args of the model ---
        temp_args = argparse.Namespace(**vars(self.args))
        temp_args.hidden_units = (hidden_units_0, hidden_units_1, hidden_units_2, hidden_units_3)
        temp_args.dropout = dropout
        temp_args.lr = lr
        temp_args.epochs = 30
        temp_args.train = True
        temp_args.verbose = False
        temp_args.last_init = last_init
        temp_args.hidden_activation = hidden_activation
        # --- loader args ---
        temp_args.validation_size = validation_size
        temp_args.test_size = test_size
        temp_args.batch_size = batch_size

        # --- train the model here ---
        model_ops = ModelOps(temp_args)
        model_ops.training()

        val_loss = min(model_ops.history["validation_losses"])
        fitness = 1.0 / (val_loss + 1e-6)

        return fitness
    
    def run_ga_optimization(self):
        print(f"{'=' * 80}\nGA optimization training activated!")
        gene_space = [
            range(64, 2049, 64),   # 0. hidden layer 1 (64, 128, 192...2048)
            range(64, 1025, 64),   # 1. hidden layer 1 (64, 128, 192...1024)
            range(32, 513, 32),    # 2. hidden layer 2 (32, 64...512)
            range(16, 257, 16),    # 3. hidden layer 3 (16, 32...256)
            np.arange(0.0, 0.6, 0.1), # 4. dropout (0.0, 0.1 ...0.6)
            np.logspace(-4, -2, num=5), # 5. lr
            range(0, 3),           # 6. hidden activations (relu, gelu, leakrelu)
            range(0, 2),           # 7. last layer init (xavier, kaiming)
            # --- loader args ---
            np.arange(0.15, 0.26, 0.05), # 8. validation_size (0.15, 0.20, 0.25) 
            np.arange(0.10, 0.21, 0.05), # 9. test_size (0.10, 0.15, 0.20)
            range(32, 257, 32),          # 10. batch_size (32, 64...256)
        ]

        ga_instance = pygad.GA(
            num_generations=30,
            num_parents_mating=10,
            fitness_func=self.ga_fitness_func,
            sol_per_pop=70,
            num_genes=len(gene_space),
            gene_space=gene_space,
            parent_selection_type="sss", # sss --> steady-state-selection
            keep_parents=4,
            crossover_type="uniform",
            mutation_percent_genes=20
        )

        ga_instance.run()
        gen = ga_instance.generations_completed
        solution, solution_fitness, _ = ga_instance.best_solution()
        print(f"\n{'=' * 80}")
        print(f"\nGA Optimization report:")
        print(f"\nGeneration {gen} | Best hyperparameters: {solution}")
        print(f"Fitness: {solution_fitness}")

        # --- train once again using the optimized hyperparameters ----
        retrain = input(f"""\nATTENTION: Do you want to train the model again
                             using the optimized Hyperparameters? y/n?""")
        
        if retrain.lower() == "y":
            best_args = argparse.Namespace(**vars(self.args))
            best_args.hidden_units = (int(solution[0]), int(solution[1]), int(solution[2]), int(solution[3]))
            best_args.dropout = float(solution[4])
            best_args.lr = float(solution[5])
            best_args.epochs = 300
            best_args.train = True
            best_args.GA_optimization = False

            # --- decode the activations ---
            activation_map = {0: "relu", 1: "gelu", 2: "leakyrelu"}
            activation_idx = solution[6]
            best_args.hidden_activation = activation_map[activation_idx]

            # --- decode the last initialization ---
            last_init_map = {0: "xavier", 1: "kaiming"}
            last_init_idx = solution[7]
            best_args.last_init = last_init_map[last_init_idx]
            # --- loader args ---
            best_args.validation_size = float(solution[8])
            best_args.test_size = float(solution[9])
            best_args.batch_size = int(solution[10])

            model_ops = ModelOps(best_args)
            model_ops.training()
            dlpl.plot_train_val_losses(history=model_ops.history,
                                       model_name="MlpClassifierGA",
                                       path=args.stats_dir,
                                       show=True)
 
# ============= MLP model operations ============
class ModelOps:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_activation = nn.Sigmoid() if self.args.out_activation == "sigmoid" else nn.Tanh() # tanh just in case i wanna change the network later.
        self.criterion = nn.BCEWithLogitsLoss()
        self.mlp_classifier = MLPBinaryClassifier(self.args).to(device)
        self.optimizer = torch.optim.Adam(self.mlp_classifier.parameters(), lr = self.args.lr)
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.history = {
            "training_losses":[],
            "validation_losses": []
        }

# =========== Training ===========  
    def training(self):
        # 1. load input data loaders.
        self.train_loader, self.val_loader, _ = ut.load_data(purpose="Train & Validation",
                                                   validation_size=self.args.validation_size,
                                                   test_size=self.args.test_size,
                                                   batch_size=self.args.batch_size,
                                                   df_path=self.args.input_dataset_path,
                                                   target_col=self.args.target_col,
                                                   scaler_method=self.args.scaler_method,
                                                   scaler_dir=self.args.scaler_dir,
                                                   model_name=self.args.model_name)
        
        if self.args.resume_training:
            checkpoint_path = os.path.join(self.args.save_best_model, "last_checkpoint.pth")
            self.resume_training(checkpoint_path=checkpoint_path)
            
        # 2. Iterate through epochs calculate oututs, loss and update network weights.
        start = getattr(self, "start_epoch", 0)
        for self.epoch in range(start, self.args.epochs):
            self.mlp_classifier.train()
            train_loss = 0.0

            for X_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.args.epochs}"):
                X_batch, y_batch = X_batch.to(device) , y_batch.to(device)
                self.optimizer.zero_grad()
                logits = self.mlp_classifier(X_batch)
                logits = logits.view(-1)
                y_batch = y_batch.view(-1)
                
                if self.args.verbose:
                    probs = self.out_activation(logits)
                    preds = (probs >= 0.5).float()
                    print(f"\n{'='*90}")
                    print(f"\nTraining info:")
                    print(f"\ninputs shape: {y_batch.shape} | predictions shape: {preds.shape}")
                    print(f"\ninputs: {y_batch}\n\noutputs : {preds}")
                    print(f"\n{'='*90}")
                
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_epoch_loss = train_loss / len(self.train_loader)
            print(f"\nTraining loss (epoch): {train_epoch_loss:.6f}")
            self.history["training_losses"].append(train_epoch_loss)
            self.validation()

            if self.early_stop_counter >= self.args.early_stop:
                print(f'Early stopping in epoch: {self.epoch + 1}')
                break
           
# =========== Validation =========== 
    def validation(self):
        self.mlp_classifier.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.val_loader, desc=f'Validation Epoch {self.epoch + 1}'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = self.mlp_classifier(X_batch)
                logits = logits.view(-1)
                y_batch = y_batch.view(-1)

                if self.args.verbose:
                    probs = self.out_activation(logits)
                    preds = (probs >= 0.5).float()
                    print(f"\n{'='*90}")
                    print(f"\nValidation info:")
                    print(f"\ninputs shape: {y_batch.shape} | predictions shape: {preds.shape}")
                    print(f"\ninputs: {y_batch}\n\noutputs : {preds}")
                    print(f"\n{'='*90}")

                loss = self.criterion(logits, y_batch)
                val_loss += loss.item()
        
        val_epoch_loss = val_loss / len(self.val_loader)
        self.history["validation_losses"].append(val_epoch_loss)
        print(f"Validation loss Avg (epoch): {val_epoch_loss:.6f}")
        
        # create checkpoint.
        checkpoint = {
            "model_state_dict": self.mlp_classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # --- for informative reasons only, can be loaded though---
            "loader_args": {
                "target_col": self.args.target_col,
                "model_name": self.args.model_name,
                "validation_size": self.args.validation_size,
                "test_size": self.args.test_size,
                "batch_size": self.args.batch_size,
                "scaler_method": self.args.scaler_method
            },
            "epoch": self.epoch + 1,
            "best_loss": min(self.history["validation_losses"]),
            "history": self.history,
            "early_stop_counter" : self.early_stop_counter,
            "hyperparams": {
                "hidden_units": self.args.hidden_units,
                "input_dim": self.args.input_dim,
                "output_dim": self.args.output_dim,
                "dropout": self.args.dropout,
                "lr": self.args.lr,
                "activation": self.args.hidden_activation,
                "out_activation": self.args.out_activation,
                "last_init": self.args.last_init
            }
        }

        # save the last checkpoint each epoch.
        last_checkpoint_log = checkpoint.copy()
        last_file_name = "last_checkpoint.pth"
        last_log_name = "last_checkpoint_info.json"
        os.makedirs(self.args.save_best_model, exist_ok=True)
        torch.save(last_checkpoint_log, os.path.join(self.args.save_best_model, last_file_name))
        
        # pop the model and optimizer state to save the info in a json file.
        last_checkpoint_log.pop('model_state_dict', None)
        last_checkpoint_log.pop('optimizer_state_dict', None)
        
        with open(os.path.join(self.args.save_best_model, last_log_name), "w") as log_file:
            json.dump(last_checkpoint_log, log_file, indent=4)

        if val_epoch_loss < self.best_loss:
            print(f'\n{"=" * 80}\nNew best model found in epoch: {self.epoch + 1}\n{"=" * 80}\n')
            self.best_loss = val_epoch_loss

            # save the best checkpoint only if we get a better loss.
            best_checkpoint_log = checkpoint.copy()
            best_file_name = "best_checkpoint.pth"
            best_log_name = "best_checkpoint_info.json"
            torch.save(best_checkpoint_log, os.path.join(self.args.save_best_model, best_file_name))
            
            best_checkpoint_log.pop('model_state_dict', None)
            best_checkpoint_log.pop('optimizer_state_dict', None)
            with open(os.path.join(self.args.save_best_model, best_log_name), "w") as log_file:
                json.dump(best_checkpoint_log, log_file, indent=4)

            self.best_epoch = self.epoch + 1

            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
    
    print(f"\n{'-' * 90}\n")

# =========== Test =========== 
    def test(self,
             checkpoint_path: str | None = None):
        print(" *** Test mode activated! ***")
        try:
            if checkpoint_path is None:
                checkpoint_path = filedialog.askopenfilename(
                title="Select a checkpoint (.pth) to start a test!",
                filetypes=[("PyTorch checkpoint", "*.pth")])
            
            if not checkpoint_path:
                print(f"\nNo checkpoint file selected! , exiting ...")
        
        except FileNotFoundError:
            print(f"\nError: Could not find the checkpoint file! ...")
            return
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # --- update args using the hyperparameters of the checkpoint ---
        hp = checkpoint.get("hyperparams", {})
        print("=================\nCheckpoint (Network weights) hyperparameters:\n")
        for k,v in hp.items():
            print(f"{k}: {v}")

        if "hidden_units" in hp: self.args.hidden_units = tuple(hp["hidden_units"]) 
        if "input_dim"     in hp: self.args.input_dim         = int(hp["input_dim"])
        if "output_dim"    in hp: self.args.output_dim        = int(hp["output_dim"])
        if "dropout"       in hp: self.args.dropout           = float(hp["dropout"])
        if "lr"            in hp: self.args.lr                = float(hp["lr"])
        if "activation"    in hp: self.args.hidden_activation = hp["activation"]
        if "out_activation"in hp: self.args.out_activation    = hp["out_activation"]
        if "last_init"     in hp: self.args.last_init         = hp["last_init"]

        # --- run the model and load states ---
        self.mlp_classifier = MLPBinaryClassifier(self.args).to(device)
        self.mlp_classifier.load_state_dict(checkpoint["model_state_dict"])
        print(f"\nCheckpoint weights loaded succesfully!...")

        # --- load the dataloader args and the input dataset ---
        loargs = checkpoint.get("loader_args", {})
        print("=================\nCheckpoint loader_args:\n")
        for k,v in loargs.items():
            print(f"{k}: {v}")

        if "target_col" in loargs: self.args.target_col = loargs["target_col"]
        if "model_name" in loargs: self.args.model_name = loargs["model_name"]
        if "validation_size" in loargs: self.args.validation_size = float(loargs["validation_size"])
        if "test_size" in loargs: self.args.test_size = float(loargs["test_size"])
        if "batch_size" in loargs: self.args.batch_size = int(loargs["batch_size"])
        if "scaler_method" in loargs: self.args.scaler_method = loargs["scaler_method"]

        test_loader = ut.load_data(purpose="Test",
                                   validation_size=self.args.validation_size,
                                   test_size=self.args.test_size,
                                   batch_size=self.args.batch_size,
                                   df_path=self.args.input_dataset_path,
                                   target_col=self.args.target_col,
                                   test_split=True,
                                   test_scaler_path= os.path.join(self.args.scaler_dir,"MlpClassifier_MinMaxScaler.pkl"),
                                   scaler_method=self.args.scaler_method,
                                   scaler_dir=self.args.scaler_dir,
                                   model_name=self.args.model_name)
        
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                logits = self.mlp_classifier(X_test)
                logits = logits.view(-1)
                y_test = y_test.view(-1)
                probs = self.out_activation(logits).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(np.float32)
                print(f"\n{'='*90}")
                print(f"\nTest info:")
                print(f"\ninputs shape: {y_test.shape} | predictions shape: {preds.shape}")
                print(f"\nGTs: {y_test}\n\nPredictions : {preds}")

            loss = self.criterion(logits, y_test)
            y_test = y_test.detach().cpu().numpy()
    
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0)
            }

            results = {
                "auroc": roc_auc_score(y_test, probs) if len(np.unique(y_test)) == 2 else float("nan"),
                "report": classification_report(y_test, preds, digits=4),
                "confusion_matrix": confusion_matrix(y_test, preds),
            }

            print(f"\nloss: {loss}")
            print(f"acc={metrics['accuracy']:.4f} | prec={metrics['precision']:.4f} | "
                f"rec={metrics['recall']:.4f} | f1={metrics['f1']:.4f} | auroc={results['auroc']:.4f}")
            print("\nClassification report:\n", results["report"])
            print("\nConfusion matrix:\n", results["confusion_matrix"])

            dlpl.plot_evaluation_metrics(metrics,
                                         model_name="MlpClassifier",
                                         save_dir=self.args.stats_dir,
                                         show=True)

        print(f"\n{'='*90}")

# =========== Resume training ===========
    def resume_training(self,
                        checkpoint_path: str | None = None,
                        loader_args: bool=False):
        # --- open the checkpoint ---
        try:
            if checkpoint_path is None:
                checkpoint_path = filedialog.askopenfilename(
                title="Select a checkpoint (.pth)",
                filetypes=[("PyTorch checkpoint", "*.pth")])
            
            if not checkpoint_path:
                print(f"\nNo checkpoint file selected! , exiting ...")
                return
            
        except FileNotFoundError:
            print(f"\nError: Could not find the checkpoint file! ...")
            return

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # --- in case you resume the trainig with saved dataloader args ---
        if loader_args:
            loargs = checkpoint.get("loader_args", {})
            if "target_col" in loargs: self.args.target_col = loargs["target_col"]
            if "model_name" in loargs: self.args.model_name = loargs["model_name"]
            if "validation_size" in loargs: self.args.validation_size = float(loargs["validation_size"])
            if "test_size" in loargs: self.args.test_size = float(loargs["test_size"])
            if "batch_size" in loargs: self.args.batch_size = int(loargs["batch_size"])
            if "scaler_method" in loargs: self.args.scaler_method = loargs["scaler_method"]
        
        # --- update args using the hyperparameters of the checkpoint ---
        hp = checkpoint.get("hyperparams", {})
        print("=================\nResume training hyperparameters:\n")
        for k,v in hp.items():
            print(f"{k}: {v}")

        if "hidden_units"  in hp: self.args.hidden_units = tuple(hp["hidden_units"]) 
        if "input_dim"     in hp: self.args.input_dim         = int(hp["input_dim"])
        if "output_dim"    in hp: self.args.output_dim        = int(hp["output_dim"])
        if "dropout"       in hp: self.args.dropout           = float(hp["dropout"])
        if "lr"            in hp: self.args.lr                = float(hp["lr"])
        if "activation"    in hp: self.args.hidden_activation = hp["activation"]
        if "out_activation"in hp: self.args.out_activation    = hp["out_activation"]
        if "last_init"     in hp: self.args.last_init         = hp["last_init"]

        # --- run the model and load states ---
        self.mlp_classifier = MLPBinaryClassifier(self.args).to(device)
        self.mlp_classifier.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = torch.optim.Adam(self.mlp_classifier.parameters(), lr=self.args.lr)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.start_epoch = checkpoint.get("epoch", 0)
        # self.early_stop_counter = checkpoint.get("early_stop_counter", 0)
        self.history = checkpoint.get("history", {
                                                    "training_losses": [],
                                                    "validation_losses": []
                                                })
        
        print(f"\nResumed from epoch {self.start_epoch} | best_loss={self.best_loss:.6f}")

# =========== main =========== 
def main(args: argparse.Namespace):
    model_ops = ModelOps(args)
   
    if args.train:
        # --- GA optimization ---
        if args.GA_optimization:
            GA = HypmOptGa(args)
            GA.run_ga_optimization()
        # -- normal training and validation --
        else:
            model_ops.training()
            dlpl.plot_train_val_losses(history=model_ops.history,
                                       model_name="MlpClassifier",
                                       path=args.stats_dir,
                                       show=True)
    # --- Test --- 
    else:
        checkpoint_path = os.path.join(args.save_best_model, "best_checkpoint.pth")
        model_ops.test(checkpoint_path)

if __name__ == "__main__":
    def mlp_classifier_args() ->argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Mlp")

        # ===== Dataset loader arguments =====
        parser.add_argument("--target_col", type=str, default="飆股",
                            help="Target column name in the dataset")
        parser.add_argument("--model_name", type=str, default="MlpClassifier",
                            help="Target column name in the dataset")
        parser.add_argument('--validation_size', type=float, default=0.2,
                            help='Percentage of data from the dataset that is going to be used for training')
        parser.add_argument('--test_size', type=float, default=0.1,
                            help='Percentage of data from the dataset that is going to be used for training')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Number of samples per batch to load')
        parser.add_argument('--scaler_method', type=str, default="minmax",
                            help='to normalize input data use "minmax"  to standardize use "standard"')
        
        # ===== MLP args =====
        parser.add_argument('--hidden_units', type=tuple, default=(1024, 512, 256, 128, 64),
                            help='dimensions in the hidden layers.')
        parser.add_argument('--input_dim', type=int, default=27,
                            help='Number of features in the input data')
        parser.add_argument('--output_dim', type=int, default=1,
                            help='Dimension of the Output')
        parser.add_argument('--dropout', type=float , default=0.2,
                            help='Neural networts to be disregarded to prevent overfitting' )
        parser.add_argument('--last_init', type=str, default="xavier",
                             help='if xavier apply xavier init else kaiming in the last layer.')
        parser.add_argument('--hidden_activation', type=str, default="relu",
                            help='The activation function in the hidden layer')
        parser.add_argument('--out_activation', type=str, default="sigmoid",
                             help='The activation function in the output layer')
        
        # === Required training arguments ===
        parser.add_argument('--GA_optimization', type=bool, default=False,
                            help='If True use GA algorithms to optimize hyperparameters')
        parser.add_argument('--train', type=bool, default=False,
                            help='Decide between train the model or make predictions')
        parser.add_argument('--resume_training', type=bool, default=False,
                            help='If True restart the training from the last checkpoint')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs to train the model')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate for the optimizer')
        parser.add_argument('--early_stop', type=int, default=100,
                            help='Number of epochs max for early stoping in case the error does not improve')
        parser.add_argument('--verbose', type=bool, default=False,
                            help='if True you can visualize tran, validation y test special info.')
        
        # ===== Paths ===== #
        parser.add_argument('--input_dataset_path', type=str,
                            default='./datasets/training_balanced_technical_analysis_dropped.csv',
                            help='Path to load the input dataset.')
        parser.add_argument('--stats_dir', type=str,
                            default='./stats/Deep_learning/mlp_classifier/',
                            help='Path of the dir to save the stats of training and validation')
        parser.add_argument('--scaler_dir', type=str,
                            default='./stats/Deep_learning/mlp_classifier/scaler_files/',
                            help='Path of the dir to save the scalers')
        parser.add_argument('--save_best_model', type=str,
                            default='./best_weights/Deep_learning/mlp_classifier/',
                            help='Path to load the generator model with the best loss obtained during training.')
        parser.add_argument('--load_best_model', type=str,
                            default='./best_weights//Deep_learning/mlp_classifier/',
                            help='Path to load the generator model with the best loss obtained during training.')
    
        args = parser.parse_args([])
        return args 
    
    args = mlp_classifier_args()

main(args)