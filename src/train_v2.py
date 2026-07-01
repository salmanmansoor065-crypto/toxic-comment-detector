"""
Model v2 (Upgraded) - Bidirectional LSTM Toxic Comment Classifier
Improvements over v1:
- Bidirectional LSTM (captures context both directions)
- Class weights (handles imbalanced labels)
- Proper metrics: Precision, Recall, F1-score (not just accuracy)
Tracked with MLflow
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("sqlite:///C:/Users/Dell/Desktop/toxic_project/mlflow.db")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from preprocess import load_and_clean, LABEL_COLS

MAX_FEATURES = 30000
MAX_LEN = 120
EMBED_DIM = 100
BATCH_SIZE = 256
EPOCHS = 4
SAMPLE_SIZE = 80000

def main():
    mlflow.set_experiment("toxic-comment-classification")

    with mlflow.start_run(run_name="v2-bidirectional-lstm"):
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

        pos_counts = y_train.sum(axis=0)
        neg_counts = y_train.shape[0] - pos_counts
        class_weight_ratios = neg_counts / np.maximum(pos_counts, 1)
        print("Class weight ratios per label:", dict(zip(LABEL_COLS, class_weight_ratios)))

        mlflow.log_params({
            "max_features": MAX_FEATURES,
            "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "sample_size": SAMPLE_SIZE,
            "model_type": "Bidirectional-LSTM",
            "improvement": "v2: BiLSTM + more data + proper metrics",
        })

        model = Sequential([
            Embedding(MAX_FEATURES, EMBED_DIM),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(128, activation="relu"),
            Dropout(0.4),
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
        y_pred_proba = model.predict(X_val, verbose=0, batch_size=1024)
        y_pred = (y_pred_proba > 0.3).astype(int)

        f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        precision_micro = precision_score(y_val, y_pred, average="micro", zero_division=0)
        recall_micro = recall_score(y_val, y_pred, average="micro", zero_division=0)

        mlflow.log_metrics({
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
        })

        for epoch, (tl, ta, vl, va) in enumerate(zip(
            history.history["loss"], history.history["accuracy"],
            history.history["val_loss"], history.history["val_accuracy"]
        )):
            mlflow.log_metrics({
                "train_loss": tl, "train_accuracy": ta,
                "val_loss": vl, "val_accuracy": va,
            }, step=epoch)

        print("\nPer-label classification report:")
        print(classification_report(y_val, y_pred, target_names=LABEL_COLS, zero_division=0))

        model.save("C:/Users/Dell/Desktop/toxic_project/models/v2_bilstm.keras")
        with open("C:/Users/Dell/Desktop/toxic_project/models/tokenizer_v2.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

        mlflow.log_artifact("C:/Users/Dell/Desktop/toxic_project/models/v2_bilstm.keras")
        mlflow.log_artifact("C:/Users/Dell/Desktop/toxic_project/models/tokenizer_v2.pkl")

        print(f"\n✅ v2 Training complete.")
        print(f"Val Accuracy: {val_acc:.4f} | F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f}")
        print(f"Precision: {precision_micro:.4f} | Recall: {recall_micro:.4f}")

if __name__ == "__main__":
    main()