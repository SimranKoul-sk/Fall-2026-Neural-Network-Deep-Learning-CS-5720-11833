"""Question 2: Sentiment Classification Using RNN"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

#Configuration
VOCAB_SIZE = 10_000   #keep only the 10,000 most frequent words
MAX_LENGTH = 200      #truncate/pad every review to this many tokens
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", 5))
CLASS_NAMES = ["Negative", "Positive"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



#Step 1: Load the IMDB sentiment dataset
#Step 2: Preprocess the text data by tokenization and padding sequences

def load_and_preprocess():
    print("Loading the IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=VOCAB_SIZE
    )

    print(f"Training reviews: {len(x_train):,}")
    print(f"Test reviews:     {len(x_test):,}")
    print(
        f"Review length before padding: "
        f"min {min(len(r) for r in x_train)}, "
        f"max {max(len(r) for r in x_train)}, "
        f"mean {np.mean([len(r) for r in x_train]):.0f}"
    )

    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=MAX_LENGTH, padding="post", truncating="post"
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=MAX_LENGTH, padding="post", truncating="post"
    )

    print(f"\nShape after padding - train: {x_train.shape}, test: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def show_example_review(x_train, y_train):
    word_index = tf.keras.datasets.imdb.get_word_index()

    #Keras reserves indices 0-3, so every real word index is shifted by 3.
    index_to_word = {index + 3: word for word, index in word_index.items()}
    index_to_word[0] = "<PAD>"
    index_to_word[1] = "<START>"
    index_to_word[2] = "<UNK>"
    index_to_word[3] = "<UNUSED>"

    decoded = " ".join(
        index_to_word.get(index, "?") for index in x_train[0] if index != 0
    )

    print("\nExample decoded review (first 300 characters):")
    print("-" * 60)
    print(decoded[:300] + "...")
    print("-" * 60)
    print(f"Label: {CLASS_NAMES[y_train[0]]}")



#Step 3: Train an LSTM-based model to classify reviews

def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MAX_LENGTH,)),
            #mask_zero=True tells the LSTM to ignore the padding tokens.
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="IMDB_LSTM_Sentiment_Classifier",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model



#Step 4: Generate a confusion matrix and classification report

def evaluate_model(model, x_test, y_test):
    print("\n" + "=" * 70)
    print("Evaluation on the Test Set")
    print("=" * 70)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(y_test, predictions)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    print("\nConfusion Matrix:")
    print(matrix)
    print("\nRead as:")
    print(f"  True Negatives  (negative, called negative): {true_negative:,}")
    print(f"  False Positives (negative, called positive): {false_positive:,}")
    print(f"  False Negatives (positive, called negative): {false_negative:,}")
    print(f"  True Positives  (positive, called positive): {true_positive:,}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, predictions, target_names=CLASS_NAMES, digits=4
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=CLASS_NAMES
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("IMDB Sentiment Classification - Confusion Matrix")
    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, "q2_confusion_matrix.png")
    plt.savefig(output_path, dpi=120)
    print(f"Saved confusion matrix figure to: {output_path}")
    plt.show()



#Step 5: Interpret why the precision-recall tradeoff is important

def explain_precision_recall():
    print("\n" + "=" * 70)
    print("Why the Precision-Recall Tradeoff Matters in Sentiment Classification")
    print("=" * 70)
    print(
        "The two metrics answer different questions about the positive class.\n\n"
        "Precision = TP / (TP + FP): of the reviews we labelled positive, how many really were? It falls when we raise false positives.\n"
        "Recall = TP / (TP + FN): of the reviews that really were positive, how many did we find? It falls when we raise false negatives.\n\n"
        "They trade off because both are controlled by the same decision threshold on the sigmoid output. Lowering the threshold below 0.5 labels more reviews positive, which catches more true positives (recall up) but also lets more negatives through (precision down). Raising the threshold does the opposite. One model therefore gives a whole curve of precision/recall pairs, not a single operating point.\n\n"
        "Which side to favour depends on what the errors cost.\n\n"
        "Brand monitoring, where negative reviews are escalated to a support team: missing a genuinely angry customer is expensive, so recall on the negative class matters more than precision. A few false alarms only cost a moment of a human's time.\n"
        "Pulling positive quotes for marketing material: a single negative review mistakenly quoted as praise is embarrassing and public, so precision matters far more than recall. Missing good reviews is fine, since there are plenty of others.\n\n"
        "This is also why accuracy alone is not enough. IMDB happens to be balanced, but on a skewed real-world feed where 95% of reviews are positive, a model that predicts 'positive' for everything scores 95% accuracy while catching zero negative reviews. The confusion matrix and the F1-score expose that failure; accuracy hides it."
    )


def main():
    #Fixed seeds so results can be reproduced between runs.
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 70)
    print("Question 2: Sentiment Classification Using an LSTM")
    print("=" * 70)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    show_example_review(x_train, y_train)

    print("\n" + "=" * 70)
    print("Model Architecture")
    print("=" * 70)
    model = build_model()
    model.summary()

    print("\n" + "=" * 70)
    print(f"Training for up to {EPOCHS} epochs")
    print("=" * 70)

    #IMDB LSTMs overfit after about one epoch, so the best weights are restored
    #rather than keeping whichever epoch happened to be last.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
    )

    evaluate_model(model, x_test, y_test)
    explain_precision_recall()


if __name__ == "__main__":
    main()
