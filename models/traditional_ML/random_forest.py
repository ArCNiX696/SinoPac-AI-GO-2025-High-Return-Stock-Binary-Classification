import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint
from typing import Optional, Union
import os
import numpy as np
from tqdm import tqdm
import argparse
import joblib
# *** my modules ***#
import utils as ut
import ml_plots as mlpl
   
# ====== Dataloader class ====== #
class Dataloader():
    def __init__(self):
        pass
    
    @staticmethod
    def split_data(df: pd.DataFrame,
                   validation_size : float,
                   test_size: float,
                   target_col: Optional[str] = None,
                   ):
        if target_col is not None and target_col not in df.columns:
            raise KeyError(f"Target column: {target_col} is not in the dataset!")
        
        if target_col is None:
            target_col = df.columns[-1] # if not target col the default is the last col of df.
            print(f"\nNo target column specified. Using '{target_col}' as default target.")

        if test_size + validation_size >= 1.0:
            raise ValueError("The sum of test_size and validation_size must be less than 1.0")
  
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        val_ratio = validation_size / (1.0 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )

        print(f"\nX_train: {len(X_train)} | y_train: {len(y_train)}")
        print(f"X_val: {len(X_val)} | y_val: {len(y_val)}, (this is {val_ratio} of validation ratio!)")
        print(f"X_test: {len(X_test)} | y_test: {len(y_test)}")

        # Distribution of the clases
        for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
            labels, counts = np.unique(arr, return_counts=True)
            print(f"\nDistribution in {name}:")
            for label, count in zip(labels, counts):
                print(f"Label {label}: {count} Samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
# ====== Cross Validation ====== #
class CrossValidation():
    def __init__(self,
                 args: argparse.Namespace):
        self.args = args
        
    def HalvingRandom_search_cv(self) -> None:
        df = ut.open_dataset(purpose="Cross Validation Train (RandomForestClassifier)")
        
        if self.args.target_col is not None and self.args.target_col not in df.columns:
            raise KeyError(f"Target column: {self.args.target_col} is not in the dataset!")
        
        if self.args.target_col is None:
            self.args.target_col = df.columns[-1] # if not target col the default is the last col of df.
            print(f"\nNo target column specified. Using '{self.args.target_col}' as default target.")

        X = df.drop(columns=[self.args.target_col]).values
        y = df[self.args.target_col].values

        X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=self.args.test_size, stratify=y, random_state=42
            )
        print(f"Train + Validation: {len(y_trainval)} - Test: {len(y_test)}")
        
        param_dist = {
            "max_depth":        randint(3, 40),
            "min_samples_split":randint(2, 50),
            "min_samples_leaf": randint(1, 40),
            "max_features":     ["sqrt", "log2", None],
            "criterion":        ["gini", "entropy"]
        }

        halving_search = HalvingRandomSearchCV(
            estimator=RandomForestClassifier(random_state=42, oob_score=True),
            param_distributions=param_dist,
            n_candidates="exhaust", # number of iterations, like this it will try all the combinations.
            factor=3,               # reduce candidates to 1/factor per round.
            resource="n_estimators",# 
            max_resources=300,      # max of trees per round.
            cv=self.args.k_folds,
            scoring=self.args.scoring,
            random_state=42,
            verbose=2, 
            n_jobs=1
        )
        halving_search.fit(X_trainval, y_trainval)

        print("\n=== Best hiperparameters founded ===")
        dirpath = os.path.dirname(self.args.cv_best_parms_path)
        os.makedirs(dirpath, exist_ok=True)

        with open(self.args.cv_best_parms_path, "w") as log_file:
            log_file.write(f"Best hyperparameters (scoring={self.args.scoring}):\n")
            log_file.write(f"\nk_folds: {self.args.k_folds}")
            log_file.write(f"\nn_iter: {self.args.n_iter}\n")
            for k, v in halving_search.best_params_.items():
                line = f"  - {k}: {v}\n"
                print(line, end="")
                log_file.write(line)
            summary = f"Mean CV {self.args.scoring}: {halving_search.best_score_:.4f}\n"
            print(summary, end="")
            log_file.write(summary)

        # ** Evaluate best stimator ** #
        best_model = halving_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuaracy in Cross validation test: {acc:.4f}")
        print(f"\nClassification Cross validation report in test:")
        print(classification_report(y_test, y_pred))

    def cross_validation_train(self):
        if self.args.HalvingRandomSearchCV:
            print(f"\nCross Validation and Optimization method --> HalvingRandomSearchCV")
            self.HalvingRandom_search_cv()

# ====== Decision Tree ====== #
class RandomForest():
    def __init__(self, 
                 args:argparse.Namespace):
        self.args = args
        self.rf_model = RandomForestClassifier(
            n_estimators=self.args.n_estimators,
            max_depth=self.args.max_depth,
            oob_score=self.args.oob_score,
            random_state=self.args.random_state
        )

    def training(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray) -> None:
        
        print("\nTraining Random Forest Classifier model...")
        self.rf_model.fit(X=X_train, y=y_train)
        print(f"(OOB) out-of-bag score: {self.rf_model.oob_score_:.4f}")

        checkpoint= {
            "model": self.rf_model,
            "hyperparams": {
                "n_estimators": self.args.n_estimators,
                "max_depth": self.args.max_depth,
                "oob_score": self.args.oob_score,
                "random_state": self.args.random_state
            }
        }

        dirpath = os.path.dirname(self.args.checkpoint_path)
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(checkpoint, self.args.checkpoint_path)
        print(f"\nCheckpoint saved to --> {dirpath}")

    def evaluate_model(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       set_name: str,
                       features_names: list[str]) -> list:
        print(f"\nEvaluating {set_name} set...")
        
        # ** 1.call load_checkpoint from utils module then make predictions** #
        model, _ = ut.load_checkpoint(path=self.args.checkpoint_path)
        
        preds = []

        for x in tqdm(X, desc=f"Prediction on {set_name} set", unit="Samples"):
            preds.append(model.predict(x.reshape(1, -1))[0]) 

        # ** 2.calculate metrics and plot results ** #
        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)
        print(f"\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nf1 score: {f1:.4f}")
        print(f"\nClassification report {set_name} set:\n{classification_report(y, preds)}")

        # plot feature importance
        mlpl.plot_ft_importance(model.feature_importances_,
                                feature_names=features_names,
                                top_features=15,
                                save_dir=self.args.stats_path,
                                model_name=f"Random Forest {set_name}")

        # plot feature metrics results
        metrics = {
            "accuaracy":acc,
            "precision":precision,
            "recall":recall,
            "f1 score":f1
        }
        
        mlpl.plot_evaluation_metrics(metrics,
                                     model_name=f"Random Forest {set_name}",
                                     save_dir=self.args.stats_path)  
        return preds

def main(args):
    # *** Cross Validation or hold-out validation *** #
    if args.CrossValidation:
        print(f"\nCross validation training activated!")
        CV = CrossValidation(args)
        CV.cross_validation_train()

    else:
        df = ut.open_dataset(purpose="Train" if args.train else "Test")
        df_copy = df.copy()
        X = df_copy.drop(columns=[args.target_col])
        features_names = X.columns

        X_train, X_val, X_test, y_train, y_val, y_test = Dataloader.split_data(
                                                            df=df,
                                                            validation_size=args.validation_size,
                                                            test_size=args.test_size,
                                                            target_col=args.target_col
                                                            )
        
        random_forest = RandomForest(args)
        
        # *** Train & Validation *** #
        if args.train:
            print(f"\nhold-out validation training activated!")
            random_forest.training(X_train, y_train)
            random_forest.evaluate_model(X_val,
                                        y_val,
                                        set_name="Validation",
                                        features_names=features_names)

        # *** Test *** #
        else:
            print(f"\ntest mode activated!")
            random_forest.evaluate_model(X=X_test,
                                        y=y_test,
                                        set_name="Test",
                                        features_names=features_names) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree with train/val/test and tqdm")
    # *** Trainig and validation Args *** #
    parser.add_argument("--target_col", type=str, default="飆股", help="Target column name")
    parser.add_argument("--train", type=bool, default=False, help="If True, train and validate the model")
    parser.add_argument("--validation_size", type=float, default=0.2, help="Validation set proportion")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    
    # *** Model Args *** #
    parser.add_argument("--n_estimators", type=int, default=243, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=34, help="Maximum tree depth")
    parser.add_argument("--oob_score", type=bool, default=True, help="If True,use out-og-bag to estimate generalization.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--min_samples_leaf", type=int, default=9, help="Minimum samples in a leaf")
    parser.add_argument("--min_samples_split", type=int, default=42, help="Minimum samples to split a node")
    parser.add_argument("--criterion", type=str, default="gini", help="Split criterion: gini or entropy")
    
    # *** Cross Validtion Args *** #
    parser.add_argument("--CrossValidation", type=bool, default=False, help="If True, activate cross validation.")
    parser.add_argument("--HalvingRandomSearchCV", type=bool, default=True, help="If True, train using HalvingRandomSearchCV and cross validation.")
    parser.add_argument("--k_folds", type=int, default=5, help="Maximum tree depth")
    parser.add_argument("--n_iter", type=int, default=100, help="Maximum of iteration for cross validation.")
    parser.add_argument("--scoring", type=str, default="f1", help="Metric for cross validation score.")

    # *** Paths *** #
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default=os.path.join("./best_weights/traditional_ML/random_forest/", "best_model_dt.joblib"),
                        help="Path to save the checkpoint obtained during training")
    
    parser.add_argument("--cv_best_parms_path",
                        type=str,
                        default=os.path.join("./best_weights/traditional_ML/random_forest/", "cv_best_parameters.txt"),
                        help="Path to save cross validation best hyperparameters")
    
    parser.add_argument("--stats_path",
                        type=str,
                        default="./stats/traditional_ML/random_forest/",
                        help="Path to save stats")

    args = parser.parse_args()

    main(args=args)