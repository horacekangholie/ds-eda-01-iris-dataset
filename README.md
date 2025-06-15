## 📚 EDA - Iris Dataset

### 🗒️ About the dataset
The Iris dataset is a classic multivariate dataset introduced by Ronald Fisher, widely used in pattern recognition and machine learning. It contains 150 samples of iris flowers, divided equally among three species: setosa, versicolor, and virginica. Each sample includes four numerical features: `sepal length`, `sepal width`, `petal length`, and `petal width`, all measured in centimeters. These features are used to classify the species of the iris flower. The dataset is well-balanced, clean, and small, making it ideal for learning classification techniques and exploratory data analysis.

### 🗒️ Histogram
Histograms are used in Exploratory Data Analysis (EDA) to visualize the distribution of a single numerical feature. They help you understand how values in a dataset are spread out, whether they are symmetrically distributed, skewed, or show signs of multimodal behavior (multiple peaks). Adding a KDE (Kernel Density Estimate) curve gives a smoothed version of the distribution, making patterns even clearer.

```python
sns.histplot(data=df_data, x=feature, ax=axes[0], kde=True)
```

![Histogram](/assets/histogram.png)

> 🔍 What the Histogram tells about the Iris Dataset
> * _Sepal Length_: The distribution is slightly right-skewed with most values clustered between 5 and 7 cm. It appears to follow a unimodal shape.
> * _Sepal Width_: This is closer to a normal distribution but slightly left-skewed, with a peak around 3.0 cm.
> * _Petal Length_: Shows a bimodal distribution, suggesting the presence of at least two distinct groups (likely related to different species). This is a strong signal that petal length is a useful feature for classification.
> * _Petal Width_: Also bimodal, with many samples having small values (near 0.2 cm) and another group between 1.0 to 2.0 cm. Like petal length, it likely differentiates between species. 

### 🗒️ Countplot
The countplot is used to visualize the frequency (i.e. count) of each category in a categorical feature—here, the target variable which represents the three species of iris flowers. By using hue='target', you reinforce the separation of classes visually, and the legend maps numeric class labels (0, 1, 2) to their actual names (setosa, versicolor, virginica)

```python
sns.countplot(data=df_data, x='target', hue='target', palette='tab10')
```

![Countplot](/assets/countplot.png)

> 🔍 What the Countplot tells about the Iris Dataset
> * Each class (setosa, versicolor, virginica) has exactly 50 samples.
> * The dataset is perfectly balanced across the three species—meaning no class is over- or under-represented.
> * This is important for machine learning because it reduces the risk of bias during training—models won't be skewed toward any specific class simply due to frequency.

### 🗒️ Pairplot
The pairplot is used in EDA to: 
* Visualize pairwise relationships between all numerical features in the dataset.
* Quickly spot patterns, correlations, or separation between classes.
* The hue='target' parameter colors the data points by species, making it easy to see how well different species are separated across different feature combinations.
* The diag_kind='kde' shows smoothed density plots along the diagonal to understand each feature’s distribution within each class.

```python
sns.pairplot(df_data, hue='target', diag_kind='kde', palette='tab10')
```

![pairplot](/assets/pairplot.png)

> 🔍 What the Pairplot tells about the Iris Dataset
> * Setosa (blue, class 0) is clearly separated from the other two classes in almost all combinations—especially in petal length and width. It forms tight, distinct clusters.
> * Versicolor (orange, class 1) and Virginica (green, class 2) show some overlap, particularly in sepal-based features, but are more separable in petal-based features.
> * There’s a strong positive correlation between:
>     * Petal length and petal width.
>     * Sepal length and petal length.
> * Features like sepal width show weaker correlation or less class separability.

> 🔍 This plot helps to conclude that:
> * Petal measurements are more effective for distinguishing species.
> * Setosa is easiest to classify.
> * Feature combinations like petal length vs petal width would be highly useful in classification models.

### 🗒️ Heatmap
A correlation heatmap is used to visualize the linear relationships between numerical variables. It helps:
* Identify strong positive or negative correlations.
* Detect redundant features (highly correlated with others) which may affect model performance.
* Guide feature selection or dimensionality reduction.

```python
sns.heatmap(corr, square=True, annot=True, mask=mask, cmap="RdBu_r")
```

__Pearson correlation coefficients__

| Correlation Coefficient (r) | Strength       |
| --------------------------- | -------------- | 
| **0.90 to 1.00**            | Very strong    |
| **0.70 to 0.89**            | Strong         |
| **0.40 to 0.69**            | Moderate       |
| **0.10 to 0.39**            | Weak           |
| **0**                       | No correlation |

![heatmap](/assets/heatmap.png)

> 🔍 What the Heatmap tells About the Iris Dataset
> * Petal width and petal length are very strongly correlated (0.96).
> * Both petal features are highly correlated with the target (0.95 and 0.96), indicating they are highly informative for classification.
> * Sepal width has a weak or even negative correlation with other features and with the target (-0.43), suggesting it may be less useful for separating species.
> * Sepal length is moderately correlated with both petal features and the target.

> 🧠 Key Insight: Petal-based measurements are far more predictive of species classification than sepal-based ones.

### 🗒️ Scatter Plot
* Scatter Plot for Feature Relationship: It shows how two continuous variables—in this case, petal length vs. petal width—relate to each other.
* Class Separation via Color: Using hue='target' colors each point by species, making it easy to see how well the three classes separate in this 2-D feature space.
* No Regression Assumption: With fit_reg=False, we focus purely on the point clouds without fitting a linear model, since we’re more interested in clustering than in trends.

```python
sns.lmplot(x="petal length (cm)", y="petal width (cm)", hue='target', data=df_data, fit_reg=False, legend=False)
```

![scatterplot](/assets/scatterplot.png)

> 🔍 What This Plot Tells About the Iris Dataset
> Perfect Cluster Separation
> * Setosa (blue) points form a tight cluster at the lower-left corner (petal length ≈1–1.9 cm, width ≈0.1–0.6 cm).
> * Versicolor (orange) occupy the middle region (length ≈3–5 cm, width ≈1–1.8 cm).
> * Virginica (green) are in the upper-right (length ≈4.5–7 cm, width ≈1.4–2.5 cm).
> Linear-Like Relationship
> * Within each species, petal length and width increase together, showing a roughly linear trend—longer petals are also wider.
> High Discriminative Power
> * There is virtually no overlap between the three groups, indicating that these two petal measurements alone can almost perfectly classify the species.

> 🧠 Key Insight:
> Petal measurements offer clear, near-perfect separation of the three Iris species—making them the most powerful features for building a simple, accurate classifier.