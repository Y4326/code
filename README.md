##  代码结构

```text
STRA/
├── data_provider/           # 数据集读取与 DataLoader 构建
├── exp/                     # 实验入口（训练、验证、测试）
├── layers/
│   └── Retrieval.py         # 检索模块核心实现
├── models/
│   └── STRAF.py              # 主模型定义
├── run.py                   # 命令行训练入口
└── result_long_term_forecast.txt
```

**项目是：

```bash
python run.py
```**
