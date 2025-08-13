OpenClash Smart 智能策略组模型训练
```shell
cd ~/path/to/your/project
python3 -m venv .venv #(创建环境)
source .venv/bin/activate #(激活环境)
pip install pandas numpy scikit-learn lightgbm optuna #(安装依赖)
python train.py #(开始训练)

# or
# 修改训练目录WORKDIR
sh run_training.sh
```
