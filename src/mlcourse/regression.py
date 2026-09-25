import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from .uci import CONCRETE_COLUMNS


def concrete_groups(frame):
    """Exact seven-ingredient recipes, excluding age and the outcome."""
    return frame.groupby(CONCRETE_COLUMNS[:7], sort=True, dropna=False).ngroup()


def concrete_split(frame):
    """Disjoint recipe groups for development/test and training/validation."""
    groups = concrete_groups(frame)
    development, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(
            frame, groups=groups
        )
    )
    train_local, validation_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
            frame.iloc[development], groups=groups.iloc[development]
        )
    )
    return development[train_local], development[validation_local], test, groups


def plot_concrete_surfaces(X, y, max_depth=3, min_samples_leaf=5):
    """Two-feature models, not slices of an eight-feature model; training data only."""
    from matplotlib import pyplot as plt

    features = ["cement", "age"]
    data = X[features]
    cement = np.linspace(data.cement.min(), data.cement.max(), 90)
    age = np.linspace(data.age.min(), data.age.max(), 90)
    xx, yy = np.meshgrid(cement, age)
    grid = pd.DataFrame({"cement": xx.ravel(), "age": yy.ravel()})
    models = [
        LinearRegression(),
        DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42
        ),
    ]
    surfaces = [model.fit(data, y).predict(grid).reshape(xx.shape) for model in models]
    lower = min(float(y.min()), *(z.min() for z in surfaces))
    upper = max(float(y.max()), *(z.max() for z in surfaces))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, z, title in zip(
        axes,
        surfaces,
        ["Linear plane", f"Tree: depth {max_depth}, leaf ≥ {min_samples_leaf}"],
    ):
        image = ax.pcolormesh(
            xx, yy, z, cmap="viridis", shading="auto", vmin=lower, vmax=upper
        )
        ax.scatter(
            data.cement,
            data.age,
            c=y,
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            s=10,
            edgecolors="white",
            linewidths=0.2,
        )
        ax.set(xlabel="Cement (kg/m³)", ylabel="Age (days)", title=title)
    fig.colorbar(image, ax=axes, label="Predicted strength (MPa)")
    return fig


def interactive_concrete_surfaces(X, y):
    import ipywidgets as widgets
    from IPython.display import display
    from matplotlib import pyplot as plt

    depth = widgets.IntSlider(
        value=3, min=1, max=12, description="Depth", continuous_update=False
    )
    leaf = widgets.IntSlider(
        value=5, min=1, max=50, description="Min leaf", continuous_update=False
    )

    def draw(max_depth, min_samples_leaf):
        plot_concrete_surfaces(X, y, max_depth, min_samples_leaf)
        plt.show()

    output = widgets.interactive_output(
        draw, {"max_depth": depth, "min_samples_leaf": leaf}
    )
    display(widgets.VBox([depth, leaf, output]))
    return depth, leaf, output
