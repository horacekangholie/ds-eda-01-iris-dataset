### 📚 EDA - Iris Dataset

#### 📄 About the dataset
The Iris dataset is a classic multivariate dataset introduced by Ronald Fisher, widely used in pattern recognition and machine learning. It contains 150 samples of iris flowers, divided equally among three species: setosa, versicolor, and virginica. Each sample includes four numerical features: `sepal length`, `sepal width`, `petal length`, and `petal width`, all measured in centimeters. These features are used to classify the species of the iris flower. The dataset is well-balanced, clean, and small, making it ideal for learning classification techniques and exploratory data analysis.

#### 📄 Histogram
Histograms are used in Exploratory Data Analysis (EDA) to visualize the distribution of a single numerical feature. They help you understand how values in a dataset are spread out, whether they are symmetrically distributed, skewed, or show signs of multimodal behavior (multiple peaks). Adding a KDE (Kernel Density Estimate) curve gives a smoothed version of the distribution, making patterns even clearer.

![Histogram](/assets/histogram.png)

<p class="has-line-data" data-line-start="0" data-line-end="1">📊 What the Histogram Tells About the Iris Dataset (from your plot)</p>
<pre><code>Sepal Length: The distribution is slightly right-skewed with most values clustered between 5 and 7 cm. It appears to follow a unimodal shape.

Sepal Width: This is closer to a normal distribution but slightly left-skewed, with a peak around 3.0 cm.

Petal Length: Shows a bimodal distribution, suggesting the presence of at least two distinct groups (likely related to different species). This is a strong signal that petal length is a useful feature for classification.

Petal Width: Also bimodal, with many samples having small values (near 0.2 cm) and another group between 1.0 to 2.0 cm. Like petal length, it likely differentiates between species.</code></pre>

