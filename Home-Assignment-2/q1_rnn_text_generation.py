"""Question 1: Implementing an RNN for Text Generation"""

import os
import numpy as np
import tensorflow as tf

#Configuration 
#The full Shakespeare file is about 1.1 million characters. Training on a slice keeps the runtime reasonable on a laptop CPU while still producing recognisable text.
TEXT_CHARS = int(os.environ.get("TEXT_CHARS", 300_000))
EPOCHS = int(os.environ.get("EPOCHS", 10))
SEQ_LENGTH = 100    #how many characters the model sees before predicting
STEP = 3            #stride when cutting training sequences out of the text
BATCH_SIZE = 128
EMBEDDING_DIM = 256
LSTM_UNITS = 256
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



#Step 1: Load the text dataset

def load_text():
    path = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    print(f"Full dataset length: {len(text):,} characters")
    text = text[:TEXT_CHARS]
    print(f"Training on the first {len(text):,} characters\n")
    print("Sample of the text:")
    print("-" * 60)
    print(text[:250])
    print("-" * 60)
    return text



#Step 2: Convert the text into sequences of characters

def build_vocabulary(text):
    vocab = sorted(set(text))
    char_to_id = {char: index for index, char in enumerate(vocab)}
    id_to_char = np.array(vocab)
    print(f"\nVocabulary size: {len(vocab)} unique characters")
    return vocab, char_to_id, id_to_char


def build_training_sequences(text, char_to_id):
    inputs = []
    targets = []
    for start in range(0, len(text) - SEQ_LENGTH, STEP):
        sequence = text[start : start + SEQ_LENGTH]
        next_char = text[start + SEQ_LENGTH]
        inputs.append([char_to_id[char] for char in sequence])
        targets.append(char_to_id[next_char])
    x = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    print(f"Training sequences: {len(x):,}")
    print(f"Input shape:  {x.shape}  (sequences x characters per sequence)")
    print(f"Target shape: {y.shape}  (one next-character per sequence)")
    return x, y



#Step 3: Define the LSTM model

def build_model(vocab_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQ_LENGTH,)),
            tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(vocab_size),
        ],
        name="Char_LSTM_TextGenerator",
    )
    model.compile(
        optimizer="adam",
        #sparse_* is used because the targets are integer indices, not one-hot.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model



#Step 4: Generate text by sampling one character at a time

def sample_with_temperature(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    exponentiated = np.exp(scaled - np.max(scaled))
    probabilities = exponentiated / np.sum(exponentiated)
    return int(np.random.choice(len(probabilities), p=probabilities))

def generate_text(model, seed_text, char_to_id, id_to_char, temperature,
                  num_chars=400):
    context = seed_text[-SEQ_LENGTH:]
    generated = []
    for _ in range(num_chars):
        encoded = np.array(
            [[char_to_id[char] for char in context]], dtype=np.int32
        )
        logits = model.predict(encoded, verbose=0)[0]
        next_index = sample_with_temperature(logits, temperature)
        next_char = id_to_char[next_index]
        generated.append(next_char)
        context = context[1:] + next_char

    return "".join(generated)

def main():
    #Fixed seeds so results can be reproduced between runs.
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 70)
    print("Question 1: Character-Level RNN for Text Generation")
    print("=" * 70)

    #Steps 1 and 2: load the data and turn it into training sequences.
    text = load_text()
    vocab, char_to_id, id_to_char = build_vocabulary(text)
    x, y = build_training_sequences(text, char_to_id)

    #Step 3: build the model.
    print("\n" + "=" * 70)
    print("Model Architecture")
    print("=" * 70)
    model = build_model(len(vocab))
    model.summary()

    #Step 4: train.
    print("\n" + "=" * 70)
    print(f"Training for {EPOCHS} epochs")
    print("=" * 70)
    history = model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1,
    )

    final_loss = history.history["loss"][-1]
    final_accuracy = history.history["accuracy"][-1]
    print(f"\nFinal training loss: {final_loss:.4f}")
    print(f"Final training accuracy: {final_accuracy:.4f}")

    #Step 5: generate text at several temperatures to show the effect.
    seed_text = text[:SEQ_LENGTH]

    print("\n" + "=" * 70)
    print("Generated Text at Different Temperatures")
    print("=" * 70)
    print(f"\nSeed text:\n{seed_text!r}")

    for temperature in [0.2, 0.5, 1.0, 1.5]:
        print("\n" + "-" * 70)
        print(f"Temperature = {temperature}")
        print("-" * 70)
        generated = generate_text(
            model, seed_text, char_to_id, id_to_char, temperature
        )
        print(generated)

    model_path = os.path.join(SCRIPT_DIR, "q1_char_lstm.keras")
    model.save(model_path)
    print(f"\n\nSaved trained model to: {model_path}")

    #Step 5 (written part): explanation of temperature scaling.
    print("\n" + "=" * 70)
    print("The Role of Temperature Scaling in Text Generation")
    print("=" * 70)
    print(
        "At each step the model outputs one score (logit) per character. Those scores go through a softmax to become probabilities, and temperature T divides the logits before that softmax: P(character) = softmax(logits / T). Temperature controls how much the model is allowed to take risks, without changing anything it learned.\n\n"
        "Low T (0.2): dividing by a small number spreads the logits further apart, so the softmax becomes sharply peaked on the single most likely character. The text is repetitive and conservative, often looping on common words, but spelling and structure are usually correct.\n"
        "T = 1.0: the logits are used exactly as the model produced them, so sampling matches the learned distribution. This is the usual balance of variety and correctness.\n"
        "High T (1.5): dividing by a larger number pulls the logits together, flattening the distribution toward uniform. Rare characters get chosen far more often, so the text is creative and surprising but drifts into misspellings and nonsense words.\n\n"
        "In short, temperature trades off coherence against diversity. As T approaches 0 the sampling becomes greedy, always taking the top character; as T grows large it approaches uniform random character choice."
    )


if __name__ == "__main__":
    main()
