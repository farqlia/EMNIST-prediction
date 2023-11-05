import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from emnist_prediction.utils.constants import SUBMISSIONS_DIR, INPUT_DATA_DIR, CLASS_LABELS, IMG_SIZE


def id_2_label(idx):
    return CLASS_LABELS[idx]


def label_2_id(label):
    return ord(label) - ord('A')


def oneh_2_label(one_hot_id):
    return CLASS_LABELS[one_hot_id.argmax(axis=-1)]


def label_2_1hot(label):
    arr = np.zeros(len(CLASS_LABELS))
    arr[label_2_id(label)] = 1
    return arr


def get_classes_count(y_true):
    class_labels = oneh_2_label(y_true)
    return pd.Series(class_labels).value_counts()


def split_save(X, y, test_size=0.25, stratify=None, writing_dir=INPUT_DATA_DIR):
    X_subval, X_subtest, y_subval, y_subtest = train_test_split(X, y, stratify=stratify,
                                                                test_size=test_size, random_state=42)

    np.save(writing_dir / 'X_subval.npy', X_subval)
    np.save(writing_dir / 'y_subval.npy', y_subval)
    np.save(writing_dir / 'X_subtest.npy', X_subtest)
    np.save(writing_dir / 'y_subtest.npy', y_subtest)


def plot_letters(images, label_ids, n_rows=3, n_cols=3):
    plt.figure(figsize=(8, 8), constrained_layout=True)
    for i, (img, label_id) in enumerate(zip(images, label_ids)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img, cmap='Greys')
        plt.title(oneh_2_label(label_id))


def visualize_img_transform(sample_img, label, transform):
    img_tensor = torch.from_numpy(sample_img.reshape(1, *IMG_SIZE))
    plt.title(oneh_2_label(label))
    plt.imshow(transform(img_tensor).reshape(*IMG_SIZE), cmap='Greys')


def save_submission(model, filename, eval_test_loader,
                    submissions_dir=SUBMISSIONS_DIR):
    trainer = pl.Trainer()
    eval_predictions = torch.cat(trainer.predict(model, eval_test_loader)).numpy()

    predictions_df = pd.DataFrame(eval_predictions).reset_index()
    predictions_df.to_csv(submissions_dir / filename, header=['index', 'class'], index=False)
    return predictions_df
