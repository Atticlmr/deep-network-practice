# Ubuntu基础命令
## 查看显卡状态
```bash
# 每秒查看一次
watch -n 1 nvidia-smi
```
## 添加可执行权限

```bash
chmod +x ./<xxx.sh>
```

# 代码结构说明
```
项目根目录
│
├── dataset/             # 数据集目录
│   ├── archive_test    
│   ├── archive_train      
│   └── cifar-10-python        
│
├── config/             # 配置文件目录
│   └── common.h         
│
├── logs/                 # 日志文件和模型文件
│   └── program          
│
├── network/             # 神经网络结构文件
│   ├── README.md        # 项目说明
│   └── design.md        # 设计文档
│
├── utils/               # 小工具的封装
│   ├── test1.c          # 测试用例1
│   └── test2.c          # 测试用例2
│
├── .gitignore           # Git忽略文件
├── main.py              # 启动入口
└── README.md            # 项目根目录说明