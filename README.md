### 📚 EDA - Iris Dataset

#### 🗒️ About the dataset
The Iris dataset is a classic multivariate dataset introduced by Ronald Fisher, widely used in pattern recognition and machine learning. It contains 150 samples of iris flowers, divided equally among three species: setosa, versicolor, and virginica. Each sample includes four numerical features: `sepal length`, `sepal width`, `petal length`, and `petal width`, all measured in centimeters. These features are used to classify the species of the iris flower. The dataset is well-balanced, clean, and small, making it ideal for learning classification techniques and exploratory data analysis.

#### 🗒️ Histogram
Histograms are used in Exploratory Data Analysis (EDA) to visualize the distribution of a single numerical feature. They help you understand how values in a dataset are spread out, whether they are symmetrically distributed, skewed, or show signs of multimodal behavior (multiple peaks). Adding a KDE (Kernel Density Estimate) curve gives a smoothed version of the distribution, making patterns even clearer.

```python
for i, feature in enumerate(feature_names):
    axes[i].set_title(feature)
    sns.histplot(data=df_data, x=feature, ax=axes[i], kde=True)
```

![Histogram](/assets/histogram.png)

💡 What the Histogram tells about the Iris Dataset
* _Sepal Length_: The distribution is slightly right-skewed with most values clustered between 5 and 7 cm. It appears to follow a unimodal shape.
* _Sepal Width_: This is closer to a normal distribution but slightly left-skewed, with a peak around 3.0 cm.
* _Petal Length_: Shows a bimodal distribution, suggesting the presence of at least two distinct groups (likely related to different species). This is a strong signal that petal length is a useful feature for classification.
* _Petal Width_: Also bimodal, with many samples having small values (near 0.2 cm) and another group between 1.0 to 2.0 cm. Like petal length, it likely differentiates between species.

#### 🗒️ Countplot
The countplot is used to visualize the frequency (i.e. count) of each category in a categorical feature—here, the target variable which represents the three species of iris flowers. By using hue='target', you reinforce the separation of classes visually, and the legend maps numeric class labels (0, 1, 2) to their actual names (setosa, versicolor, virginica)

```python
sns.countplot(data=df_data, x='target', hue='target', palette='tab10')
```

![Countplot](/assets/countplot.png)

💡 What the Countplot tells about the Iris Dataset
* Each class (setosa, versicolor, virginica) has exactly 50 samples.
* The dataset is perfectly balanced across the three species—meaning no class is over- or under-represented.
* This is important for machine learning because it reduces the risk of bias during training—models won't be skewed toward any specific class simply due to frequency.

#### 🗒️ Pairplot
The pairplot is used in EDA to: 
* Visualize pairwise relationships between all numerical features in the dataset.
* Quickly spot patterns, correlations, or separation between classes.
* The hue='target' parameter colors the data points by species, making it easy to see how well different species are separated across different feature combinations.
* The diag_kind='kde' shows smoothed density plots along the diagonal to understand each feature’s distribution within each class.

```python
sns.pairplot(df_data, hue='target', diag_kind='kde', palette='tab10')
```

![pairplot](/assets/pairplot.png)

💡 What the Pairplot tells about the Iris Dataset
* Setosa (blue, class 0) is clearly separated from the other two classes in almost all combinations—especially in petal length and width. It forms tight, distinct clusters.
* Versicolor (orange, class 1) and Virginica (green, class 2) show some overlap, particularly in sepal-based features, but are more separable in petal-based features.
* There’s a strong positive correlation between:
    * Petal length and petal width.
    * Sepal length and petal length.
* Features like sepal width show weaker correlation or less class separability.

💡 This plot helps to conclude that:
* Petal measurements are more effective for distinguishing species.
* Setosa is easiest to classify.
* Feature combinations like petal length vs petal width would be highly useful in classification models.