"""
Model v1 (Baseline) - LSTM-based Toxic Comment Classifier
Tracked with MLflow
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("sqlite:///C:/Users/Dell/Desktop/toxic_project/mlflow.db")
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from preprocess import load_and_clean, LABEL_COLS

MAX_FEATURES = 20000
MAX_LEN = 100
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 3
SAMPLE_SIZE = 60000

def main():
    mlflow.set_experiment("toxic-comment-classification")

    with mlflow.start_run(run_name="v1-baseline-lstm"):
        df = load_and_clean("C:/Users/Dell/Desktop/toxic_project/data/train.csv")
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

        X_text = df["clean_text"].values
        y = df[LABEL_COLS].values

        X_train_text, X_val_text, y_train, y_val = train_test_split(
            X_text, y, test_size=0.2, random_state=42
        )

        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(X_train_text)

        X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN)
        X_val = pad_sequences(tokenizer.texts_to_sequences(X_val_text), maxlen=MAX_LEN)

        mlflow.log_params({
            "max_features": MAX_FEATURES,
            "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "sample_size": SAMPLE_SIZE,
            "model_type": "Embedding+LSTM",
        })

        model = Sequential([
            Embedding(MAX_FEATURES, EMBED_DIM, input_length=MAX_LEN),
            LSTM(64, return_sequences=True),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(len(LABEL_COLS), activation="sigmoid"),
        ])

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stop],
            verbose=2,
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        mlflow.log_metrics({
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
        })

        for epoch, (tl, ta, vl, va) in enumerate(zip(
            history.history["loss"], history.history["accuracy"],
            history.history["val_loss"], history.history["val_accuracy"]
        )):
            mlflow.log_metrics({
                "train_loss": tl, "train_accuracy": ta,
                "val_loss": vl, "val_accuracy": va,
            }, step=epoch)

        model.save("C:/Users/Dell/Desktop/toxic_project/models/v1_baseline.keras")
        with open("C:/Users/Dell/Desktop/toxic_project/models/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

        mlflow.log_artifact("C:/Users/Dell/Desktop/toxic_project/models/v1_baseline.keras")
        mlflow.log_artifact("C:/Users/Dell/Desktop/toxic_project/models/tokenizer.pkl")

        print(f"\n✅ Training complete. Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()