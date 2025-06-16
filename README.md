## 📚 EDA - Iris Dataset

[About the dataset](#️-about-the-dataset)
[Histogram](#️-histogram)
[Bar Chart](#️-bar-chart)

### 🗒️ About the dataset
The Iris dataset is a classic multivariate dataset introduced by Ronald Fisher, widely used in pattern recognition and machine learning. It contains 150 samples of iris flowers, divided equally among three species: setosa, versicolor, and virginica. Each sample includes four numerical features: `sepal length`, `sepal width`, `petal length`, and `petal width`, all measured in centimeters. These features are used to classify the species of the iris flower. The dataset is well-balanced, clean, and small, making it ideal for learning classification techniques and exploratory data analysis.

### 🗒️ Histogram
A histogram is a commonly used chart to show the distribution of data — it is a one-dimensional bar chart representing the frequency of data intervals. Seaborn’s `histplot()` function is used to draw histograms of the four Iris features (sepal and petal width and length).

This plot allows us to visually check how each variable is distributed. From the chart, we can observe:
* The petal length and petal width distributions show clear separations between classes.
* The sepal measurements show more overlapping.
* This helps us preliminarily identify which features may better distinguish between different species.

```python
sns.histplot(data=df_data, x=feature, ax=axes[0], kde=True)
```

* `x="sepal length (cm)"`: specifies which column to plot on the x-axis.
* `data=df_data`: specifies the dataset being used.
* `ax=axes[...]`: specifies which subplot the chart will be drawn on (left to right).
* `kde=True`: enables the Kernel Density Estimation curve to be drawn over the histogram, which helps visualize the smooth distribution of the data.

![Histogram](/assets/histogram.png)

> 🔍 What the Histogram tells about the Iris Dataset
> * _Sepal Length_: The distribution is slightly right-skewed with most values clustered between 5 and 7 cm. It appears to follow a unimodal shape.
> * _Sepal Width_: This is closer to a normal distribution but slightly left-skewed, with a peak around 3.0 cm.
> * _Petal Length_: Shows a bimodal distribution, suggesting the presence of at least two distinct groups (likely related to different species). This is a strong signal that petal length is a useful feature for classification.
> * _Petal Width_: Also bimodal, with many samples having small values (near 0.2 cm) and another group between 1.0 to 2.0 cm. Like petal length, it likely differentiates between species. 

### 🗒️ Bar Chart
The Bar Chart is used to visualize the frequency (i.e. count) of each category in a categorical feature—here, the target variable which represents the three species of iris flowers. By using hue='target', you reinforce the separation of classes visually, and the legend maps numeric class labels (0, 1, 2) to their actual names (setosa, versicolor, virginica)

```python
sns.countplot(data=df_data, x='target', hue='target', palette='tab10')
```

![Countplot](/assets/countplot.png)

> 🔍 What the Countplot tells about the Iris Dataset
> * Each class (setosa, versicolor, virginica) has exactly 50 samples.
> * The dataset is perfectly balanced across the three species—meaning no class is over- or under-represented.
> * This is important for machine learning because it reduces the risk of bias during training—models won't be skewed toward any specific class simply due to frequency.

### 🗒️ Kernel Density Estimation Plot
Kernel Density Estimation (KDE) plots are used to visualize data distributions. Diagonal plots represent KDE plots for each feature, while off-diagonal plots represent scatter plots between features. This type of plot is very effective for visualizing how features are distributed for each species and identifying which features may help distinguish between categories.
For example:
* Petal length and petal width show clear clustering among different species, suggesting good separation.
* Sepal length and sepal width show more overlap.
This plot provides an intuitive and effective way to observe the overall structure of the data.

```python
sns.pairplot(df_data, hue='target', diag_kind='kde', palette='tab10')
```
* `df_data`: the dataset to be plotted.
* `hue='target'`: sets the target class as the hue, so different species will be shown in different colors.
* `diag_kind='kde'`: sets the diagonal plots (individual variable distributions) to use kernel density estimation.
* `palette='tab10'`: uses the built-in seaborn color palette "tab10" to distinguish different species.

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

### 🗒️ Correlation Heatmap
A correlation heatmap is used to visualize explore relationships between different numerical features. By visualizing correlations using a heatmap, we can quickly identify which variables are more related.
This is useful for:
* Feature selection – removing redundant or weak features.
* Understanding which features are strongly related to the target for potential modeling.
* Reducing multicollinearity – preventing highly correlated inputs in models like linear regression.
* Early warning of feature interdependence, helping design more robust models.

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
Scatter plots visualize the relationship between two numeric features on a 2D plane. Each dot represents an observation and helps in understanding the pattern between two variables. When we have labeled data (like Iris species), we can assign different colors to each class to observe potential separations between categories. In this case, we use Seaborn's `lmplot()` to plot sepal length vs sepal width, and color each point according to its `target` class.

```python
sns.lmplot(x="petal length (cm)", y="petal width (cm)", hue='target', data=df_data, fit_reg=False, legend=False)
```

![scatterplot](/assets/scatterplot.png)

> 🔍 What This Plot Tells About the Iris Dataset

> * Perfect Cluster Separation
>   * Setosa (blue) points form a tight cluster at the lower-left corner (petal length ≈1–1.9 cm, width ≈0.1–0.6 cm).
>   * Versicolor (orange) occupy the middle region (length ≈3–5 cm, width ≈1–1.8 cm).
>   * Virginica (green) are in the upper-right (length ≈4.5–7 cm, width ≈1.4–2.5 cm).
> * Linear-Like Relationship
>   * Within each species, petal length and width increase together, showing a roughly linear trend—longer petals are also wider.
> * High Discriminative Power
>   * There is virtually no overlap between the three groups, indicating that these two petal measurements alone can almost perfectly classify the species.

> 🧠 Key Insight:
> Petal measurements offer clear, near-perfect separation of the three Iris species, 
> making them the most powerful features for building a simple, accurate classifier.

### 🗒️ Box Plot
A box plot (also called a box-and-whisker plot) is a common visualization for statistical distribution. It shows a dataset’s median, quartiles, and outliers, helping users quickly understand the data’s central tendency and spread.
Key Concepts in a Box Plot:
* `Median`: The central value that separates the higher half from the lower half of the data.
* `Quartiles` (Q1, Q3):
  * Q1: the value at the 25th percentile.
  * Q3: the value at the 75th percentile.
* `Box`: Represents the interquartile range (IQR), which covers the middle 50% of the data (from Q1 to Q3).
* `Whiskers`: The lines extending from the box to the smallest and largest values within 1.5×IQR from the box.
* `Outliers`: Data points that fall outside the whiskers (typically beyond 1.5×IQR from Q1 or Q3).

In seaborn’s boxplot(), the whiskers by default extend to `1.5 × IQR`. This means values outside this range are considered outliers and shown as individual points. 
This visualization is useful for:
* Spotting skewed distributions,
* Comparing multiple features side by side,
* Quickly identifying potential outliers.

```python
sns.boxplot(data=df_data, showmeans=True, orient="v", palette="tab10")
```
* `showmeans=True`: Display the mean as a symbol on the plot.
* `orient="v"`: Vertical orientation of the box plot.
* `palette="tab10"`: Use seaborn’s built-in color palette to differentiate each variable.

![boxplot](/assets/boxplot.png)

> 🔍 What This Plot Tells About the Iris Dataset
> * Sepal Length (cm)
>   * Median around 5.8 cm, IQR roughly 5.1–6.4 cm.
>   * Whiskers span from about 4.3 cm up to 7.9 cm.
>   * Relatively symmetrical, with few extreme values.
> * Sepal Width (cm)
>   * Narrow IQR (~2.8–3.3 cm), indicating low variability.
>   * A couple of outliers above ~4.0 cm.
>   * Slight left-skew (median closer to the top of the box).
> * Petal Length (cm)
>   * Wide spread (IQR ~1.6–5.1 cm) reflecting two distinct clusters (setosa vs. the other species).
>   * Whiskers extend from ~1.0 cm up to ~6.9 cm, highlighting more variability.
>   * Box shows that petal length varies far more than sepal width.
> * Petal Width (cm)
>   * Similar story to petal length: IQR ~0.3–1.8 cm, whiskers up to ~2.5 cm.
>   * Clear separation in spread between setosa (very low widths) and the others.
> * Target
>   * Plotting the numeric target (0–2) mainly shows that the median sits at 1 with minimal spread—less informative here. (Boxplots are best reserved for continuous variables.)

> 🧠 Key Takeaways
> * Petal features exhibit greater variability and clear grouping—reinforcing their power for class separation.
> * Sepal width is fairly consistent with few outliers.
> * Boxplots quickly reveal which features have skew, outliers, or distinct groupings that merit deeper analysis.


### 🗒️ Summary & Conclusion
In this exploratory analysis of the classic Iris dataset (150 samples, three equally represented species), first confirmed a perfectly balanced target distribution (50 samples per species) and no missing values. Univariate histograms and KDEs revealed that sepal length and width are roughly unimodal and show moderate variability, whereas petal length and width exhibit clear bimodal patterns—an early indication of strong class separation.

Pairwise scatterplots and correlation heatmaps highlighted that petal length and petal width are very strongly correlated (r≈0.96) and both correlate highly with species (r≈0.95–0.96). In contrast, sepal width even shows a moderate negative correlation with the target (r≈–0.43) and weaker correlations with other features. Boxplots reinforced these findings: petal measurements have much greater spread and distinct groupings, while sepal width remains relatively consistent across species.

Conclusion:
* Petal dimensions are the most informative features for distinguishing species, offering near-perfect linear separability—ideal predictors for a simple classifier.
* Sepal measurements, especially sepal width, add little discriminative power on their own and could be deprioritized or combined with petal features in feature engineering.
* With these insights, a straightforward model (e.g., logistic regression or decision tree) using petal length and width can achieve high accuracy; further work can explore robustness, cross-validation, and feature scaling.

