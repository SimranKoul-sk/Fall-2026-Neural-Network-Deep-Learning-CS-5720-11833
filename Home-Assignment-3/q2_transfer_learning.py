"""Question 2: Transfer Learning, Freeze vs Fine-Tune"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#Configuration
#flower_photos is a 5-class image dataset (daisy, dandelion, roses,
#sunflowers, tulips) of 3670 photos, downloaded and cached on first run.
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", 5))

#Experiment A trains only a fresh classifier head, so it can take a normal
#learning rate. Experiment B updates pretrained convolution weights, which
#needs a much smaller rate or the useful features get destroyed.
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



#Step 1: Load the image-classification dataset

#Descend through single-child folders until the directory holding the class subfolders is found.
def find_class_root(path):
    #Different Keras versions return either the archive or the extraction
    #directory, and the archive unpacks to flower_photos_extracted/flower_photos.
    #Walking down while there is exactly one subdirectory lands on the folder
    #whose children are the classes, whichever layout was produced.
    if not os.path.isdir(path):
        path = os.path.splitext(path)[0]

    for _ in range(4):
        subdirs = sorted(e.path for e in os.scandir(path) if e.is_dir())
        if len(subdirs) == 1:
            path = subdirs[0]
        else:
            break

    class_count = sum(1 for e in os.scandir(path) if e.is_dir())
    if class_count < 2:
        raise RuntimeError(
            f"Expected at least two class folders under {path}, found {class_count}"
        )

    return path


#Download flower_photos and return the training and validation datasets plus class names.
def load_datasets():
    downloaded = tf.keras.utils.get_file("flower_photos.tgz", DATA_URL, extract=True)
    root = find_class_root(downloaded)
    print(f"Dataset root: {root}")

    common = dict(
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root, subset="training", **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root, subset="validation", **common
    )

    class_names = train_ds.class_names
    print(f"\nClasses ({len(class_names)}): {class_names}")

    #ResNet50 expects its own channel scaling, not plain 0-1 normalisation.
    preprocess = tf.keras.applications.resnet50.preprocess_input
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y)).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(autotune)

    return train_ds, val_ds, class_names



#Step 2: Build the two models

#Attach a fresh classification head to a ResNet50 base.
def attach_head(base, num_classes, name):
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name=name,
    )
    return model


#Experiment A: load ResNet50, freeze the whole base, train only the new classifier.
def build_feature_extractor(num_classes):
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)
    )

    #Freezing the base means none of the pretrained convolution weights are
    #updated. The base acts purely as a fixed feature extractor.
    base.trainable = False

    model = attach_head(base, num_classes, "Frozen_Feature_Extractor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FROZEN),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


#Experiment B: load ResNet50 and unfreeze the last convolutional block (conv5) plus the classifier.
def build_fine_tuned(num_classes):
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)
    )
    base.trainable = True

    #Unfreeze only conv5, the last residual block. Everything earlier stays
    #frozen, so the general edge and texture filters are preserved.
    for layer in base.layers:
        layer.trainable = layer.name.startswith("conv5_")

        #BatchNorm layers are kept frozen even inside conv5. Updating their
        #running statistics on a small dataset is a well known way to wreck
        #a pretrained network, so they stay in inference mode.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model = attach_head(base, num_classes, "Fine_Tuned_Network")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



#Step 3: Train each model and record parameters, time and accuracy

#Return the number of trainable and non-trainable weights in a model.
def count_parameters(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return trainable, non_trainable


#Train one model, timing only the fit call, and return its results.
def train_and_time(model, label, train_ds, val_ds):
    trainable, non_trainable = count_parameters(model)

    print("\n" + "=" * 70)
    print(f"{label}")
    print("=" * 70)
    print(f"Trainable parameters:     {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Training for {EPOCHS} epochs\n")

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
    elapsed = time.time() - start

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

    print(f"\nTraining time:       {elapsed:.1f} s")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss:     {val_loss:.4f}")

    return {
        "label": label,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "seconds": elapsed,
        "accuracy": val_accuracy,
        "loss": val_loss,
        "history": history.history["loss"],
    }



#Step 4: Plot the training loss for both experiments

#Plot both training loss curves on one figure and save it as a PNG.
def plot_training_loss(results):
    plt.figure(figsize=(8, 5))
    for result in results:
        epochs = range(1, len(result["history"]) + 1)
        plt.plot(epochs, result["history"], marker="o", label=result["label"])

    plt.title("Training Loss: Frozen Feature Extractor vs Fine-Tuned Network")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.xticks(list(range(1, EPOCHS + 1)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, "q2_training_loss.png")
    plt.savefig(output_path, dpi=120)
    print(f"\nSaved training-loss plot to: {output_path}")
    plt.show()



#Step 5: Compare the two approaches in a table

#Print the comparison table required by part (f).
def print_comparison_table(results):
    print("\n" + "=" * 78)
    print("Comparison")
    print("=" * 78)

    header = f"{'Method':<28}{'Trainable Params':>18}{'Training Time':>16}{'Accuracy':>12}"
    print(header)
    print("-" * 78)
    for result in results:
        print(
            f"{result['label']:<28}"
            f"{result['trainable']:>18,}"
            f"{result['seconds']:>14.1f} s"
            f"{result['accuracy']:>12.4f}"
        )
    print("-" * 78)


#Print the written discussion required by part (g).
def discuss_results(results):
    frozen, tuned = results

    print("\n" + "=" * 78)
    print("Discussion")
    print("=" * 78)
    print(
        f"The frozen feature extractor trains only {frozen['trainable']:,} parameters "
        f"while the fine-tuned network trains {tuned['trainable']:,}, a difference of "
        f"roughly {tuned['trainable'] / max(frozen['trainable'], 1):.0f}x, because the "
        "first updates a single dense layer and the second also updates the whole "
        "conv5 residual block. That gap explains most of the difference in training "
        "time: both models run the same forward pass, but the frozen model needs "
        "gradients only for the last layer, whereas the fine-tuned model has to "
        "backpropagate through the final block and hold those activations in memory. "
        "Freezing is therefore cheaper per epoch, and the saving would be larger "
        "still if more of the network were unfrozen.\n\n"
        "Accuracy moves for a different reason. ImageNet features are already close "
        "to what flower photos need, so a linear classifier on top of frozen features "
        "is a strong baseline and gets most of the way there on its own. Fine-tuning "
        "can beat it because the last block holds the most task-specific features, "
        "and letting those adapt lets the network re-tune what it considers "
        "discriminative for flowers rather than for ImageNet classes. The risk is "
        "overfitting: with far more trainable parameters and only a few thousand "
        "images, the fine-tuned model can memorise the training set, which is why it "
        "uses a learning rate 10x smaller and why the BatchNorm layers are left "
        "frozen. In short, freezing is faster and safer on small datasets, while "
        "fine-tuning costs more time and needs more care but has the higher ceiling "
        "when the new task differs from the pretraining task."
    )


#Run both experiments, then plot, tabulate and discuss the results.
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("=" * 70)
    print("Question 2: Transfer Learning, Freeze vs Fine-Tune")
    print("=" * 70)

    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    #Experiment A.
    extractor = build_feature_extractor(num_classes)
    frozen_result = train_and_time(
        extractor, "Frozen Feature Extractor", train_ds, val_ds
    )

    #Experiment B.
    fine_tuned = build_fine_tuned(num_classes)
    tuned_result = train_and_time(
        fine_tuned, "Fine-Tuned Network", train_ds, val_ds
    )

    results = [frozen_result, tuned_result]

    plot_training_loss(results)
    print_comparison_table(results)
    discuss_results(results)


if __name__ == "__main__":
    main()
