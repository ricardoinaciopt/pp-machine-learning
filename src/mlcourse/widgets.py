"""Bounded interactive laboratories with visible controls and testable renderers."""
from dataclasses import dataclass
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .labs import boundary, binary_metrics, cluster_plot, cluster_summary, grid


@dataclass
class Laboratory:
    widget: object
    controls: dict
    render: object


def control(kind, description, **kwargs):
    kwargs.update(description=description, style={'description_width': '180px'},
                  layout=widgets.Layout(width='440px'))
    if 'Slider' in kind.__name__:
        kwargs['continuous_update'] = False
    if kind in (widgets.FloatSlider, widgets.FloatLogSlider):
        kwargs['readout_format'] = '.3f'
    return kind(**kwargs)


def laboratory(draw, controls):
    output = widgets.interactive_output(draw, controls)
    return Laboratory(widgets.VBox([*controls.values(), output]), controls, draw)


def classification_lab(factory, controls, X_train, X_test, y_train, y_test):
    def draw(**parameters):
        model = factory(**parameters).fit(X_train, y_train)
        train = accuracy_score(y_train, model.predict(X_train))
        test = accuracy_score(y_test, model.predict(X_test))
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        boundary(ax, model, X_test, y_test, f'Train accuracy {train:.3f}; test accuracy {test:.3f}')
        plt.show()
        return {'train_accuracy': train, 'test_accuracy': test}
    return laboratory(draw, controls)


def interactive_knn(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    controls = {'n_neighbors': control(widgets.IntSlider, 'n_neighbors', value=5, min=1, max=min(31, len(X_train)), step=2),
                'weights': control(widgets.Dropdown, 'weights', options=['uniform', 'distance'])}
    return classification_lab(lambda **p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
                              controls, X_train, X_test, y_train, y_test)


def interactive_tree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    controls = {'max_depth': control(widgets.IntSlider, 'max_depth', value=3, min=1, max=12),
                'min_samples_leaf': control(widgets.IntSlider, 'min_samples_leaf', value=1, min=1, max=20)}
    return classification_lab(lambda **p: DecisionTreeClassifier(**p, random_state=0),
                              controls, X_train, X_test, y_train, y_test)


def interactive_svm(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    controls = {'kernel': control(widgets.Dropdown, 'kernel', options=['linear', 'poly', 'rbf'], value='rbf'),
                'C': control(widgets.FloatLogSlider, 'C', value=1., base=10, min=-2, max=2, step=.5),
                'gamma': control(widgets.FloatLogSlider, 'gamma', value=1., base=10, min=-2, max=1, step=.5),
                'degree': control(widgets.IntSlider, 'degree', value=3, min=2, max=5)}
    def relevant(change=None):
        controls['gamma'].disabled = controls['kernel'].value == 'linear'
        controls['degree'].disabled = controls['kernel'].value != 'poly'
    controls['kernel'].observe(relevant, names='value')
    relevant()
    return classification_lab(lambda **p: make_pipeline(StandardScaler(), SVC(**p)),
                              controls, X_train, X_test, y_train, y_test)


def interactive_mlp(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    controls = {'width': control(widgets.IntSlider, 'Hidden-layer width', value=5, min=1, max=30),
                'depth': control(widgets.IntSlider, 'Hidden-layer count', value=1, min=1, max=3),
                'activation': control(widgets.Dropdown, 'activation', options=['tanh', 'relu', 'logistic', 'identity']),
                'alpha': control(widgets.FloatLogSlider, 'alpha', value=.01, base=10, min=-4, max=1, step=1),
                'random_state': control(widgets.IntSlider, 'random_state', value=0, min=0, max=10)}
    def model(width, depth, **p):
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(width,)*depth,
                             solver='lbfgs', max_iter=1500, tol=1e-3, **p))
    return classification_lab(model, controls, X_train, X_test, y_train, y_test)


def interactive_regression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from .labs import regression_metrics
    controls = {'max_depth': control(widgets.IntSlider, 'max_depth', value=3, min=1, max=12),
                'min_samples_leaf': control(widgets.IntSlider, 'min_samples_leaf', value=5, min=1, max=40)}
    def draw(max_depth, min_samples_leaf):
        models = {'Linear plane': LinearRegression(),
                  'Regression tree': DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)}
        xx, yy, points = grid(X_train, size=80)
        models = {name: model.fit(X_train, y_train) for name, model in models.items()}
        surfaces = [m.predict(points).reshape(xx.shape) for m in models.values()]
        lower = min(float(y_train.min()), *(z.min() for z in surfaces))
        upper = max(float(y_train.max()), *(z.max() for z in surfaces))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        for ax, (name, model), surface in zip(axes, models.items(), surfaces):
            picture = ax.pcolormesh(xx, yy, surface, cmap='viridis', shading='auto', vmin=lower, vmax=upper)
            ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis', vmin=lower, vmax=upper,
                       s=10, edgecolors='white', linewidths=.2)
            ax.set(title=name, xlabel=X_train.columns[0], ylabel=X_train.columns[1])
        fig.colorbar(picture, ax=axes, label='Predicted strength (MPa)')
        plt.show()
        scores = pd.DataFrame({name: regression_metrics(y_test, model.predict(X_test)) for name, model in models.items()}).T
        display(scores)
        return scores
    return laboratory(draw, controls)


def interactive_clustering(X, algorithm='K-Means'):
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    controls = {'algorithm': control(widgets.Dropdown, 'Algorithm', options=['K-Means', 'DBSCAN', 'Hierarchical'], value=algorithm),
                'n_clusters': control(widgets.IntSlider, 'n_clusters', value=3, min=2, max=8),
                'eps': control(widgets.FloatSlider, 'eps', value=.5, min=.05, max=2., step=.05),
                'min_samples': control(widgets.IntSlider, 'min_samples', value=5, min=2, max=20),
                'linkage': control(widgets.Dropdown, 'linkage', options=['complete', 'average', 'single', 'ward'])}
    def relevant(change=None):
        method = controls['algorithm'].value
        controls['n_clusters'].disabled = method == 'DBSCAN'
        controls['eps'].disabled = controls['min_samples'].disabled = method != 'DBSCAN'
        controls['linkage'].disabled = method != 'Hierarchical'
    controls['algorithm'].observe(relevant, names='value')
    relevant()
    def draw(algorithm, n_clusters, eps, min_samples, linkage):
        model = (KMeans(n_clusters=n_clusters, n_init=10, random_state=0) if algorithm == 'K-Means'
                 else DBSCAN(eps=eps, min_samples=min_samples) if algorithm == 'DBSCAN'
                 else AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage))
        labels = model.fit_predict(X)
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        cluster_plot(ax, X, labels, algorithm)
        plt.show()
        scores = pd.DataFrame([cluster_summary(X, labels)])
        display(scores)
        return scores
    return laboratory(draw, controls)


def interactive_threshold(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    controls = {'threshold': control(widgets.FloatSlider, 'Decision threshold', value=.5, min=.05, max=.95, step=.05),
                'class_weight': control(widgets.Dropdown, 'class_weight', options=[('Uniform', None), ('Balanced', 'balanced')])}
    # Fit the two fixed models once; threshold movement never retrains.
    models = {weight: make_pipeline(StandardScaler(), LogisticRegression(class_weight=weight)).fit(X_train, y_train)
              for weight in [None, 'balanced']}
    def draw(threshold, class_weight):
        model = models[class_weight]
        pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        boundary(axes[0], model, X_test, y_test, f'Threshold {threshold:.3f}', threshold=threshold)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1], ax=axes[1], colorbar=False)
        axes[1].set_title('Test predictions')
        plt.show()
        scores = pd.DataFrame([binary_metrics(y_test, pred)])
        display(scores)
        return scores
    return laboratory(draw, controls)


def interactive_meta_performance(meta_data, predictions):
    descriptors = ['n_instances', 'n_features', 'n_classes', 'class_imbalance', 'numeric_ratio', 'class_entropy']
    strategies = {
        'Full tree': 'tree_accuracy',
        'Decision stump': 'stump_accuracy',
        'Majority baseline': 'majority_accuracy',
    }
    controls = {
        'dataset': control(widgets.Dropdown, 'Dataset', options=meta_data.dataset.tolist(), value=meta_data.dataset.iloc[0]),
        'feature': control(widgets.Dropdown, 'Dataset-level feature', options=descriptors, value='class_imbalance'),
        'strategy': control(widgets.Dropdown, 'Performance to inspect', options=list(strategies), value='Full tree'),
    }

    # Use widget-native image and HTML outputs rather than an IPython Output area.
    # This avoids duplicate Matplotlib/table renders in frontends that replay
    # widget-output messages when several controls synchronise at display time.
    image = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1200px'))
    table = widgets.HTML(layout=widgets.Layout(width='100%', max_width='900px'))

    def draw(dataset, feature, strategy):
        target = strategies[strategy]
        row = meta_data.loc[meta_data.dataset.eq(dataset)].iloc[0]
        pred = predictions.loc[predictions.dataset.eq(dataset)].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        axes[0].scatter(meta_data[feature], meta_data[target], s=50)
        axes[0].scatter([row[feature]], [row[target]], s=180, marker='*')
        axes[0].set(xlabel=feature, ylabel='Measured accuracy', title=f'{strategy} across datasets')
        axes[0].annotate(dataset, (row[feature], row[target]), xytext=(8, 8), textcoords='offset points')

        labels = list(strategies)
        measured = [row[strategies[label]] for label in labels]
        predicted = [pred[f'pred_{strategies[label]}'] for label in labels]
        positions = np.arange(len(labels))
        width = .36
        axes[1].bar(positions - width/2, measured, width, label='Measured')
        axes[1].bar(positions + width/2, predicted, width, label='Predicted')
        axes[1].set_xticks(positions, labels, rotation=15, ha='right')
        axes[1].set_ylim(0, 1.05)
        axes[1].set(ylabel='Accuracy', title=f'Performance on {dataset}')
        axes[1].legend()

        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        image.value = buffer.getvalue()

        ranking = pd.DataFrame({
            'strategy': labels,
            'predicted_accuracy': predicted,
            'measured_accuracy': measured,
        })
        ranking['predicted_rank'] = ranking.predicted_accuracy.rank(method='min', ascending=False).astype(int)
        ranking['measured_rank'] = ranking.measured_accuracy.rank(method='min', ascending=False).astype(int)
        ranking = ranking.sort_values('predicted_rank').reset_index(drop=True)
        table.value = ranking.to_html(index=False, float_format=lambda value: f'{value:.3f}')
        return ranking

    # Some notebook frontends synchronise each control once when the widget is
    # first displayed. With three controls, that can trigger the observer three
    # times in addition to the explicit initial render. Cache the complete
    # control state so identical synchronisation events do not recompute the
    # laboratory. A genuine user change still produces one new state and one
    # render.
    last_state = {'value': None}

    def update(change=None):
        state = (controls['dataset'].value,
                 controls['feature'].value,
                 controls['strategy'].value)
        if state == last_state['value']:
            return
        draw(dataset=state[0], feature=state[1], strategy=state[2])
        last_state['value'] = state

    for widget in controls.values():
        widget.observe(update, names='value')
    update()

    content = widgets.VBox([image, table])
    return Laboratory(widgets.VBox([*controls.values(), content]), controls, draw)

