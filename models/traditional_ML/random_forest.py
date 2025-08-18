import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint
import json
from typing import Optional
import os
import numpy as np
import argparse
import joblib
# *** my modules ***#
import utils as ut
import ml_plots as mlpl
       
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

# ====== Random Forest ====== #
class RandomForest():
    def __init__(self, 
                 args:argparse.Namespace):
        self.args = args
        self.rf_model = RandomForestClassifier(
            n_estimators=self.args.n_estimators,
            max_depth=self.args.max_depth,
            oob_score=self.args.oob_score,
            random_state=self.args.random_state,
            min_samples_leaf=self.args.min_samples_leaf,
            min_samples_split=self.args.min_samples_split,
            criterion=self.args.criterion,
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
                       feature_names: list[str]) -> list:
        print(f"\nEvaluating {set_name} set...")
        
        # ** 1.call load_checkpoint from utils module then make predictions** #
        model, _ = ut.load_checkpoint(path=self.args.checkpoint_path)
        preds = model.predict(X)
        n_pred = 50
        print(f"\n========= Random Forest predictions vs GTs: =========")
        print(f"\nPredictions ({n_pred}) values:\n{preds[:50]}\n")
        print(f"\nGTs ({n_pred}) values:\n{preds[:50]}")
        print(f"\n=====================================================\n")

        # ** 2.calculate metrics and plot results ** #
        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)
        print(f"\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nf1 score: {f1:.4f}")
        print(f"\nClassification report {set_name} set:\n{classification_report(y, preds)}")

        # plot feature metrics results
        metrics = {
            "accuaracy":acc,
            "precision":precision,
            "recall":recall,
            "f1 score":f1
        }
        
        mlpl.plot_evaluation_metrics(metrics,
                                     model_name=f"Random Forest {set_name}",
                                     save_dir=self.args.stats_path,
                                     show=self.args.show)  
        
        # plot feature importance
        mlpl.plot_ft_importance(model.feature_importances_,
                                feature_names=feature_names,
                                top_features=15,
                                save_dir=self.args.stats_path,
                                model_name=f"Random Forest {set_name}",
                                show=self.args.show)

        # ** 5.return the best features ** #
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances_sorted = importances.sort_values(ascending=False)
        best_importances = int(len(importances_sorted) * self.args.best_features_rate)
        best_features = importances_sorted.head(best_importances)
        best_features_dict = best_features.to_dict()
        
        print(f"\n========= Best {best_importances} features: =========")
        for v, k in best_features_dict.items():
            print(f"{v}: {k}")
        print(f"\n=====================================================\n")

        with open(self.args.best_features_path, "w", encoding="utf-8") as f:
            json.dump(best_features_dict, f, ensure_ascii=False, indent=4)
        print(f"\nBest features json file saved in --> {os.path.dirname(self.args.best_features_path)}")
        
        return preds

def main(args):
    if args.CrossValidation and args.train:
        raise ValueError("Choose either CrossValidation or hold-out training, not both.")
    
    # *** Cross Validation or hold-out validation *** #
    if args.CrossValidation:
        print(f"\nCross validation training activated!")
        CV = CrossValidation(args)
        CV.cross_validation_train()

    else:
        df = ut.open_dataset(purpose="Train" if args.train else "Test")
        df_copy = df.copy()
        X = df_copy.drop(columns=[args.target_col])
        feature_names = X.columns
        X_train, X_val, X_test, y_train, y_val, y_test = ut.split_data(
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
                                        feature_name=feature_names)

        # *** Test *** #
        else:
            print(f"\ntest mode activated!")
            random_forest.evaluate_model(X=X_test,
                                        y=y_test,
                                        set_name="Test",
                                        feature_names=feature_names) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest with train/val/test and feature selection")

    # ========== Dataset Config ========== #
    parser.add_argument("--target_col", type=str, default="飆股",
                        help="Name of the target column to predict")
    parser.add_argument("--validation_size", type=float, default=0.2,
                        help="Proportion of data used for validation set (only in hold-out mode)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data used for test set")

    # ========== Training Strategy ========== #
    parser.add_argument("--train", type=bool, default=False,
                        help="Train the model using hold-out validation")
    parser.add_argument("--CrossValidation", type=bool, default=False,
                        help="Enable cross-validation mode")
    parser.add_argument("--HalvingRandomSearchCV", type=bool, default=True,
                        help="Use HalvingRandomSearchCV for hyperparameter tuning")
    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds for cross-validation")
    # parser.add_argument("--n_iter", type=int, default=100,
    #                     help="(Not used) Number of iterations for RandomizedSearchCV")

    # ========== Model Hyperparameters ========== #
    parser.add_argument("--n_estimators", type=int, default=243,
                        help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=34,
                        help="Maximum depth of each tree")
    parser.add_argument("--min_samples_leaf", type=int, default=9,
                        help="Minimum number of samples required at a leaf node")
    parser.add_argument("--min_samples_split", type=int, default=42,
                        help="Minimum number of samples required to split an internal node")
    parser.add_argument("--criterion", type=str, default="gini",
                        help="Function to measure the quality of a split: 'gini' or 'entropy'")
    parser.add_argument("--oob_score", type=bool, default=True,
                        help="Use out-of-bag samples to estimate the generalization accuracy")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed used by the random number generator")

    # ========== Feature Selection ========== #
    parser.add_argument("--best_features_rate", type=float, default=0.8,
                        help="Proportion of best features to retain based on importance scores")

    # ========== Evaluation & Paths ========== #
    parser.add_argument("--scoring", type=str, default="f1",
                        help="Metric used for cross-validation scoring")

    parser.add_argument("--checkpoint_path", type=str,
                        default=os.path.join("./best_weights/traditional_ML/random_forest/", "best_model_rf.joblib"),
                        help="Path to save the trained model checkpoint")

    parser.add_argument("--cv_best_parms_path", type=str,
                        default=os.path.join("./best_weights/traditional_ML/random_forest/", "cv_best_parameters.txt"),
                        help="Path to save best hyperparameters found during cross-validation")

    parser.add_argument("--best_features_path", type=str,
                        default=os.path.join("./best_weights/traditional_ML/random_forest/", "best_features_rf.json"),
                        help="Path to save the selected best features as JSON")

    parser.add_argument("--stats_path", type=str,
                        default="./stats/traditional_ML/random_forest/",
                        help="Path to save evaluation stats and plots")

    parser.add_argument("--show", type=bool, default=False,
                        help="If True, show the plots")

    args = parser.parse_args()
    main(args=args)
