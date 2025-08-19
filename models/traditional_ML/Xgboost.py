from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint, uniform
import pandas as pd
from typing import Optional, Union
import json
import os
import numpy as np
import argparse
import joblib
# *** my modules ***#
import utils as ut
import ml_plots as mlpl
     
# ====== Cross Validation ====== #
class CrossValidation():
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def HalvingRandom_search_cv(self) -> None:
        df = ut.open_dataset(purpose="Cross Validation Train (XGBoostClassifier)")

        if self.args.target_col is not None and self.args.target_col not in df.columns:
            raise KeyError(f"Target column: {self.args.target_col} is not in the dataset!")

        if self.args.target_col is None:
            self.args.target_col = df.columns[-1]
            print(f"\nNo target column specified. Using '{self.args.target_col}' as default target.")

        X = df.drop(columns=[self.args.target_col]).values
        y = df[self.args.target_col].values

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.args.test_size, stratify=y, random_state=42
        )

        print(f"Train + Validation: {len(y_trainval)} - Test: {len(y_test)}")

        # XGBoost parameter distribution
        param_dist = {
            "max_depth": randint(3, 50),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.5, 0.5),         # range [0.5, 1.0]
            "colsample_bytree": uniform(0.5, 0.5),   # range [0.5, 1.0]
            "gamma": uniform(0, 5),
            "reg_alpha": uniform(0, 1),              # L1
            "reg_lambda": uniform(0, 1),             # L2
        }

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        halving_search = HalvingRandomSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_candidates="exhaust",
            factor=3,
            resource="n_estimators",
            max_resources=300,
            cv=self.args.k_folds,
            scoring=self.args.scoring,
            random_state=42,
            verbose=2,
            n_jobs=-1
        )

        halving_search.fit(X_trainval, y_trainval)

        print("\n=== Best hyperparameters found ===")
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

        # Save model
        best_model = halving_search.best_estimator_
        joblib.dump(best_model, os.path.join(dirpath, "best_model_xgb_cv.joblib"))
        print(f"\nBest XGBoost model saved to --> {dirpath}")

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy on test set: {acc:.4f}")
        print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")

    def cross_validation_train(self):
        if self.args.HalvingRandomSearchCV:
            print(f"\nCross Validation and Optimization method --> HalvingRandomSearchCV (XGBoost)")
            self.HalvingRandom_search_cv()

# ====== Decision Tree ====== #
class XGBoostModel():
    def __init__(self, 
                 args:argparse.Namespace):
        self.args = args
        self.xgb_model = XGBClassifier(
            learning_rate=self.args.learning_rate,
            n_estimators=self.args.n_estimators,
            max_depth=self.args.max_depth,
            subsample=self.args.subsample,
            colsample_bytree=self.args.colsample_bytree,
            gamma=self.args.gamma,
            reg_alpha=self.args.reg_alpha,
            reg_lambda=self.args.reg_lambda,
            use_label_encoder=self.args.use_label_encoder,
            eval_metric=self.args.eval_metric,
            random_state=self.args.random_state
        )

    def training(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray) -> None:
        
        print("\nTraining XGBoost Classifier model...")
        self.xgb_model.fit(X=X_train, y=y_train)
        
        checkpoint= {
            "model": self.xgb_model,
            "hyperparams": {
                "learning_rate": self.args.learning_rate,
                "n_estimators": self.args.n_estimators,
                "max_depth": self.args.max_depth,
                "subsample": self.args.subsample,
                "colsample_bytree": self.args.colsample_bytree,
                "gamma": self.args.gamma,
                "reg_alpha": self.args.reg_alpha,
                "reg_lambda": self.args.reg_lambda,
                "use_label_encoder": self.args.use_label_encoder,
                "eval_metric": self.args.eval_metric,
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
                       set_name: str) -> list:
        print(f"\nEvaluating {set_name} set...")
        
        # ** 1.call load_checkpoint from utils module then make predictions** #
        model, _ = ut.load_checkpoint(path=self.args.checkpoint_path)
        preds = model.predict(X)
        n_pred = 50
        print(f"\n========= XGBoost predictions vs GTs: =========")
        print(f"\nPredictions ({n_pred}) values:\n{preds[:50]}\n")
        print(f"\nGTs ({n_pred}) values:\n{preds[:50]}")
        print(f"\n===============================================\n")

        # ** 2.calculate metrics and plot results ** #
        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)
        print(f"\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nf1 score: {f1:.4f}")
        print(f"\nClassification report {set_name} set:\n{classification_report(y, preds)}")
        
        # ** 3.plot feature metrics results ** #
        metrics = {
            "accuaracy":acc,
            "precision":precision,
            "recall":recall,
            "f1 score":f1
        }
        
        mlpl.plot_evaluation_metrics(metrics,
                                     model_name=f"XGBoostClassifier (cleaned) {set_name}",
                                     save_dir=self.args.stats_path,
                                     show=self.args.show)  
        
        # ** 4. Plot feature importances ** #
        mlpl.plot_importance_xgb(model,
                                 importance_type=self.args.importance_type,
                                 max_num_features=20,
                                 title=f"Top Features by {self.args.importance_type}",
                                 show_values=True,
                                 model_name="XGBoostClassifier (cleaned)",
                                 save_dir=self.args.stats_path,
                                 show=self.args.show)
        
        # ** 5.return the best features ** #
        feature_names = X.columns
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
        
        return X, preds

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
        X_train, X_val, X_test, y_train, y_val, y_test = ut.split_data(
                                                            df=df,
                                                            validation_size=args.validation_size,
                                                            test_size=args.test_size,
                                                            target_col=args.target_col
                                                            )
        
        xgboost = XGBoostModel(args)
        
        # *** Train & Validation *** #
        if args.train:
            print(f"\nhold-out validation training activated!")
            xgboost.training(X_train, y_train)
            xgboost.evaluate_model(X_val,
                                   y_val,
                                   set_name="Validation")

        # *** Test *** #
        else:
            print(f"\ntest mode activated!")
            GTs, preds = xgboost.evaluate_model(X=X_test,
                                                y=y_test,
                                                set_name="Test") 
            results = pd.DataFrame({
                "ID" : X_test.index,
                "GTs" : GTs,
                "Prediction": preds
            })

            print(f"\n{'='*90}\nResults:\n{results}\n{'='*90}\n")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize an XGBoostClassifier with train/val/test and optional cross-validation")

    # === Required training arguments ===
    parser.add_argument("--target_col", type=str, default="飆股",
                        help="Target column name in the dataset")

    # === Model hyperparameters for XGBoost ===
    parser.add_argument("--n_estimators", type=int, default=243,
                        help="Number of boosting rounds (trees)")
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Maximum depth of a tree")
    parser.add_argument("--learning_rate", type=float, default=0.2625,
                        help="Step size shrinkage used in update to prevent overfitting")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Subsample ratio of the training instances (row sampling)")
    parser.add_argument("--colsample_bytree", type=float, default=0.87056,
                        help="Subsample ratio of columns when constructing each tree (feature sampling)")
    parser.add_argument("--gamma", type=float, default=2.8723,
                        help="Minimum loss reduction required to make a further partition on a leaf node")
    parser.add_argument("--reg_alpha", type=float, default=0.8279,
                        help="L1 regularization term on weights (encourages sparsity)")
    parser.add_argument("--reg_lambda", type=float, default=0.75050,
                        help="L2 regularization term on weights (prevents overfitting)")
    parser.add_argument("--eval_metric", type=str, default="logloss",
                        help="Evaluation metric used by XGBoost during training")
    parser.add_argument("--use_label_encoder", type=bool, default=False,
                        help="Disable the label encoder for compatibility with newer sklearn versions")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed for reproducibility")
    parser.add_argument("--importance_type", type=str, default="gain",
                    help="Type of feature importance metric to display (weight, gain, cover, total_gain, total_cover)")

    # === Return best features names ===
    parser.add_argument("--best_features_rate", type=float, default=0.8,
                        help="to return only a portion of the best features founded by the model.")
    
    # === Optional training flags ===
    parser.add_argument("--train", type=bool, default=False,
                        help="If True, train and validate the model using hold-out strategy")
    parser.add_argument("--validation_size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the validation split")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split")

    # === Cross-validation arguments ===
    parser.add_argument("--CrossValidation", type=bool, default=True,
                        help="If True, perform cross-validation instead of hold-out training")
    parser.add_argument("--HalvingRandomSearchCV", type=bool, default=True,
                        help="If True, use HalvingRandomSearchCV for hyperparameter optimization")
    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds in cross-validation")
    parser.add_argument("--scoring", type=str, default="f1",
                        help="Metric used for model selection during cross-validation")

    # === Paths  and plots ===
    parser.add_argument("--checkpoint_path", type=str,
                        default=os.path.join("./best_weights/traditional_ML/XGBoost/", "best_model_xgb_(cleaned).joblib"),
                        help="Path to save the trained XGBoost model")
    parser.add_argument("--cv_best_parms_path", type=str,
                        default=os.path.join("./best_weights/traditional_ML/XGBoost/", "cv_best_parameters_(cleaned).txt"),
                        help="Path to save the best hyperparameters from cross-validation")
    parser.add_argument("--stats_path", type=str,
                        default="./stats/traditional_ML/XGBoost/",
                        help="Path to save plots and evaluation results")
    parser.add_argument("--best_features_path", type=str,
                        default=os.path.join("./best_weights/traditional_ML/XGBoost/", "best_features_xgb_(cleaned).json"),
                        help="Path to save the best features foundes by XGBoost model")
    parser.add_argument("--show", type=bool, default=False,
                        help="If True, show the plots.")

    args = parser.parse_args()
    
    main(args=args)