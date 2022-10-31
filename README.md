# PRML-HW

Computer programming homework for PRML course of Tsinghua Prof. Xuegong Zhang

---

### 10.1 HW2

完成了FDL的实现以及2D可视化，完成了Perceptron/Logistic Regression的使用（基于scikit-learn）

- Perceptron以及Logistic Regression的最优参数搜索
- Logistic Regression的ROC-AUC曲线绘制
- feature importance的计算（基于随机森林）

本地复现本作业的结果时请执行以下代码，检查环境是否齐全。并且注意，本作业是在Linux平台上进行的实现，内部有些路径（"/"）在windows上可能无法解析，请windows复现结果者自行解决。
    
```shell
python3 -m pip install -r requirements.txt
```

除了FDL之外，其余的两个实验均使用Jupyter Notebook，这里不再赘述`.ipynb`的执行方法，欲查看FDL实验的结果，请执行：
    
```shell
cd exe2/
python3 ./fdl.py
```

---

### 10.4 HW3

​		调用`sklearn`中实现的`KNN`算法，采取了三种不同的策略：

- KNN default setting (由`sklearn`库定义，例如其`n_neighbors = 5`)
- KNN，并且进行`GridSearchCV`
- KNN default setting，但输入使用PCA降维

​		代码中目前只保留了前两种方案，如果要测试PCA降维，请将代码的`25-27`行解注释：

```python
# pca = PCA(n_components = 16)
# train_set = pca.fit_transform(train_set, train_label)
# test_set = pca.fit_transform(test_set, test_label)
```

​		如果要运行代码，此处有三种可能的运行选择 （首先得`cd exe3/`）：

```shell
python3 ./knn_gridsearch.py 0
python3 ./knn_gridsearch.py 1
python3 ./knn_gridsearch.py
```

​		第一种运行方式：代表使用默认参数运行

​		第二种运行方式：代表使用Grid Search结果参数运行

​		第三种运行方式：代表进行Grid Search

​		每一种方式结束后，都会执行`k=3-49`的`n_neighbors`对训练、测试集精度影响分析的绘图。

---

### 10.8 HW4

​		运行MLP代码，需要使用Pytorch（基于pytorch的实现），如果您环境装的好（Pytorch的wheel CUDA版本与本地CUDA版本一致），那么应该可以使用Nvidia APEX混合精度加速库。则无需修改，直接使用如下方法运行：

```shell
cd exe4/
mkdir check_points/
python3 ./custom_mlp.py -s
```

​		如果您无法使用APEX加速库，则去掉`-s` flag（本实现不支持native amp，则不会进行任何混合精度加速）。

---

### 10.15 HW5 (simple SVM)

​		SVM对偶问题求解，依赖库：`pytorch`（CPU版本即可），`matplotlib`。运行方法如下：

```shell
cd exe5/
python3 ./svm_auto.py
```

---

### 10.21 HW5 - 2 (SVM package)

​		SVM（调用`sklearn`库进行了三种不同kernel函数的实验）。本代码并不提供简单的`kernel`函数切换，如果需要切换`kernel`，请自行修改代码中的（42-46）行:

```python
parameters = {'C': 0.5,
        'kernel' : 'poly',
        'degree': 3,
        'gamma': 'auto'
}
```

​		如果需要测试grid search，将`if __name__ == "__main__"`块中的`svm_test()`替换为`grid_search_params()`。使用如下方式运行：

```shell
cd exe5/
python3 ./svm_sklearn.py
```

---

### 10.31 HW6 (Feature Selection & Extraction & LASSO)

​		在`exe6/`文件夹中包含了两个文件：

```
knn_gridsearch.py --------- 包含了PCA-based feature extraction\PCA-KNN\KNN Grid Search\LASSO\LASSO Grid Search
pca.py -------------------- 包含了自己实现的PCA算法
```

​		运行：

```shell
cd exe6/
python3 ./knn_gridsearch.py				  // 什么都不加：lasso grid search
python3 ./knn_gridsearch.py	0或1			  // 进行knn测试
python3 ./knn_gridsearch.py	2及以上		// 进行lasso测试
```

