import seaborn as sns
from matplotlib import pyplot as plt


def print_summary(report, f1_score_column, threshold_value=None):

    if threshold_value is None:
        threshold_value = report['train_class_prob'].mean()

    mean_f1_baseline = report[f1_score_column].mean()
    print(f"Mean f1 score [{f1_score_column}] = {mean_f1_baseline:.3f}")
    print(f"Threshold value = {threshold_value:.3f}")

    f1_baseline_below_mean = report[report[f1_score_column] < mean_f1_baseline]
    class_prob_below_mean = report[report['train_class_prob'] < threshold_value]

    print("Classes with f1 score BELOW mean\n", f1_baseline_below_mean[[f1_score_column, 'train_class_prob']])
    print("Classes with probability BELOW threshold\n", class_prob_below_mean[[f1_score_column, 'train_class_prob']])

    min_labels_below_mean = set(f1_baseline_below_mean.index) & set(class_prob_below_mean.index)

    print(min_labels_below_mean)
    print("Percentage of minority classes with scores < mean",
          f"{(len(min_labels_below_mean) / len(set(class_prob_below_mean.index))):.3f}", 
          ", mean f1 score in these classes = ",
          f"{report.loc[list(min_labels_below_mean), f1_score_column].mean():.3f}")

    class_prob_above_mean = report[report['train_class_prob'] >= threshold_value]

    maj_labels_above_mean = set(f1_baseline_below_mean.index) & set(class_prob_above_mean.index)

    print(maj_labels_above_mean)
    print("Percentage of majority classes with scores < mean",
          f"{(len(maj_labels_above_mean) / len(set(class_prob_above_mean.index))):.3f}", 
          ", mean f1 score in these classes = ",
          f"{report.loc[list(maj_labels_above_mean), f1_score_column].mean():.3f}")

def plot_distributions(report, *f1_score_columns):
    plt.figure(figsize=(7, 7))
    sorted_class_labels = report.sort_index().index
    sns.barplot(data=report[['train_class_prob', *f1_score_columns]].reset_index().melt(
        value_vars=['train_class_prob', *f1_score_columns], id_vars=['index']), y='value',
                x='index', order=sorted_class_labels, hue='variable')
    plt.ylabel('Value')
    plt.xlabel('Letter')
    _ = plt.title('Distribution of class prob and f1')


def plot_f1_vs_class_prob(report, *f1_score_columns):
    plt.figure(figsize=(7, 7))
    sns.relplot(
        data=report[['train_class_prob', *f1_score_columns]].melt(value_vars=f1_score_columns, id_vars=['train_class_prob']),
        y='train_class_prob',
        x='value', hue='variable')
    plt.ylabel('Class probability')
    plt.xlabel('F1 score')
    _ = plt.title('Distribution of class prob and f1')
