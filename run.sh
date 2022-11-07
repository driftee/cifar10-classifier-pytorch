# bashline
python3 run.py --model 0

# 减小卷积核大小，增加模型深度
python3 run.py --model 1

# 卷积核数量 - 减少
python3 run.py --model 2

# 卷积核数量 - 增加
python3 run.py --model 3

# 使用平均池化
python3 run.py --model 4

# 使用baseline并调整dropout率
python3 run.py --model 0 --dropout 0
python3 run.py --model 0 --dropout 0.1
python3 run.py --model 0 --dropout 0.15