"""Presentation helpers for compact, reproducible teaching experiments."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import silhouette_score
from .setup import find_root


def load_course_data(name):
    paths = {'hotel': 'ho01/HotelReservations.csv', 'iris': 'ho03/Iris.csv',
             'toy': 'ho07/toy3.csv', **{f'ds{i}': f'shared/ds{i}.csv' for i in range(1, 4)}}
    root = find_root()
    if name in ('concrete', 'wholesale'):
        from .uci import load_dataset
        return load_dataset(165 if name == 'concrete' else 292, root)
    path = root/'data'/paths[name]
    if not path.is_file():
        raise FileNotFoundError(f'Course data missing: {paths[name]}. Restore the repository data folder and rerun setup.')
    return pd.read_csv(path)


def split_classification(X, y, seed=0):
    return train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)


def grid(X, size=120):
    values = np.asarray(X)
    pad = np.maximum(np.ptp(values, axis=0)*.08, .1)
    xx, yy = np.meshgrid(np.linspace(values[:, 0].min()-pad[0], values[:, 0].max()+pad[0], size),
                         np.linspace(values[:, 1].min()-pad[1], values[:, 1].max()+pad[1], size))
    points = np.c_[xx.ravel(), yy.ravel()]
    if isinstance(X, pd.DataFrame):
        points = pd.DataFrame(points, columns=X.columns)
    return xx, yy, points


def boundary(ax, model, X, y, title, threshold=None):
    xx, yy, points = grid(X)
    prediction = model.predict(points) if threshold is None else (model.predict_proba(points)[:, 1] >= threshold).astype(int)
    classes = np.unique(np.concatenate([np.asarray(y), prediction]))
    encoded = np.searchsorted(classes, prediction).reshape(xx.shape)
    levels = np.arange(len(classes)+1)-.5
    from matplotlib.colors import ListedColormap
    colors = ['#0072B2', '#D55E00', '#009E73']
    ax.contourf(xx, yy, encoded, levels=levels, cmap=ListedColormap(colors[:len(classes)]), alpha=.18)
    values = np.asarray(X)
    for i, value in enumerate(classes):
        selected = np.asarray(y) == value
        ax.scatter(values[selected, 0], values[selected, 1], s=24, c=colors[i % 3],
                   marker=['o', '^', 's'][i % 3], edgecolors='white', linewidths=.3, label=str(value))
    names = list(X.columns) if isinstance(X, pd.DataFrame) else ['Feature 1', 'Feature 2']
    ax.set(xlabel=names[0], ylabel=names[1], title=title)
    ax.legend(title='True class', fontsize=8)


def compare_boundaries(models, X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 4.8), squeeze=False, layout='constrained')
    rows = []
    for ax, (name, estimator) in zip(axes[0], models.items()):
        model = clone(estimator).fit(X_train, y_train)
        train = accuracy_score(y_train, model.predict(X_train))
        test = accuracy_score(y_test, model.predict(X_test))
        rows.append({'model': name, 'train_accuracy': train, 'test_accuracy': test})
        boundary(ax, model, X_test, y_test, f'{name}: test accuracy {test:.3f}')
    plt.show()
    return pd.DataFrame(rows).set_index('model')


def binary_metrics(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'F1': f1_score(y, pred, zero_division=0)}


def regression_metrics(y, pred):
    return {'MSE': mean_squared_error(y, pred), 'MAE': mean_absolute_error(y, pred)}


def make_imbalance_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                              n_clusters_per_class=1, weights=[.9, .1], class_sep=.65,
                              flip_y=.02, random_state=1)
    return pd.DataFrame(X, columns=['A1', 'A2']), pd.Series(y, name='class')


def histogram_grid(frame):
    columns = frame.select_dtypes(include='number').columns
    rows = int(np.ceil(len(columns)/3))
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3*rows), squeeze=False, layout='constrained')
    for ax, name in zip(axes.flat, columns):
        ax.hist(frame[name].dropna(), bins=20, color='#0072B2', edgecolor='white')
        ax.set(title=name, xlabel='Value', ylabel='Count')
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    plt.show()


def cluster_summary(X, labels):
    labels = np.asarray(labels)
    keep = labels != -1
    count = len(np.unique(labels[keep]))
    value = silhouette_score(np.asarray(X)[keep], labels[keep]) if 1 < count < keep.sum() else np.nan
    return {'clusters': count, 'noise_count': int((~keep).sum()), 'silhouette': value}


def cluster_plot(ax, X, labels, title):
    values = np.asarray(X)
    labels = np.asarray(labels)
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(values[mask, 0], values[mask, 1], s=28,
                   marker='x' if label == -1 else 'o',
                   color='grey' if label == -1 else plt.cm.tab10(int(label) % 10),
                   label='Noise' if label == -1 else f'Cluster {label}')
    names = list(X.columns) if isinstance(X, pd.DataFrame) else ['Coordinate 1', 'Coordinate 2']
    ax.set(xlabel=names[0], ylabel=names[1], title=title)
    ax.legend(fontsize=8)


def compare_clusters(models, X, projection=None):
    fig, axes = plt.subplots(1, len(models), figsize=(5.5*len(models), 4.8), squeeze=False, layout='constrained')
    rows = []
    for ax, (name, model) in zip(axes[0], models.items()):
        labels = clone(model).fit_predict(X)
        cluster_plot(ax, X if projection is None else projection, labels, name)
        rows.append({'model': name, **cluster_summary(X, labels)})
    plt.show()
    return pd.DataFrame(rows).set_index('model')


def load_task_results():
    path = find_root()/'data/ho06/results.zip'
    if not path.is_file():
        raise FileNotFoundError('Course data missing: ho06/results.zip. Restore the repository data folder and rerun setup.')
    with zipfile.ZipFile(path) as archive:
        names = sorted((n for n in archive.namelist() if n.startswith('results_ds') and n.endswith('.csv')),
                       key=lambda n: int(Path(n).stem.replace('results_ds', '')))
        return pd.DataFrame({Path(n).stem: pd.read_csv(archive.open(n)).mean() for n in names}).T


def load_meta_dataset():
    from .metalearning import load_meta_dataset as verified_load
    source = verified_load(find_root())
    names = {'Dataset': 'dataset', 'mf_n_instances': 'n_instances', 'mf_n_features_raw': 'n_features',
             'mf_n_classes': 'n_classes', 'mf_imbalance_ratio': 'class_imbalance',
             'mf_missing_rate': 'missing_rate', 'mf_numeric_ratio': 'numeric_ratio',
             'mf_class_entropy': 'class_entropy',
             'AccuracyDT': 'tree_accuracy', 'AccuracyDS': 'stump_accuracy',
             'AccuracyMajority': 'majority_accuracy',
             'AccuracyGap_DT_minus_DS': 'accuracy_gap', 'BestModelBinary': 'tree_wins'}
    return source[list(names)].rename(columns=names)
