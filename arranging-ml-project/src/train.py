import os

import click
import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher

@click.command()
@click.option(
    "--fold",
    type=int,
    default=0,
)
@click.option(
    "--model",
    type=str,
    default="decision_tree_gini",
)
def run(
    fold,
    model    
):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # drop label column
    x_train = df_train.drop(df_train.iloc[:, 0:1], axis=1).values
    y_train = df_train.iloc[:, 0:1]

    x_valid = df_valid.drop(df_valid.iloc[:, 0:1], axis=1).values
    y_valid = df_valid.iloc[:, 0:1]

    # fit the model on training data
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    
    preds = clf.predict(x_valid)

    # calculate metrics
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"../models/dt_{fold}.bin")
    )
        

if __name__ == "__main__":
    run()