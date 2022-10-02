# PRML-HW

Computer programming homework for PRML course of Tsinghua Prof. Xuegong Zhang

---

### 10.1 HW3

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
cd exe3/
python3 ./fdl.py
```
