# data-pipeline-clean 代码逐行全解（零基础版）

> 目标：看完能说出每个 .py 文件里每一行代码在干什么。
> 前提：知道什么是变量、列表、字典、for 循环、if 判断。

---

# 第1部分：converter.py — CSV 转 JSONL

## 1.1 文件头 — import 语句

```python
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
```

### import 是什么意思？

Python 有"标准库"（装 Python 自带的）和"第三方库"（别人写好了你 pip install 安装的）。
import 库名 的意思是：把这段别人写好的代码拿过来用。不 import 就用不了。

### 逐条解释

**import json**
- json 是标准库，处理 JSON 格式的数据
- json.dumps(字典) → 把 Python 字典变成字符串 '{"name": "小明"}'
- json.loads(字符串) → 把 JSON 字符串变回 Python 字典
- 为什么需要：AI 训练数据常用 JSONL（每行一个 JSON 对象），所以需要这个库

**import uuid**
- uuid 是标准库，生成全局唯一的 ID
- uuid.uuid4() → 生成一个随机的 UUID 对象
- .hex → 把 UUID 对象转成纯十六进制字符串，比如 "a1b2c3d4e5f6g7h8"
- 为什么需要：每条训练数据都需要一个唯一标识符，不能重复

**from datetime import datetime**
- from 库 import 工具 的意思是：从某个库里只拿某个工具
- datetime.now() → 获取当前时间，比如 2026-06-02 12:43:18.090295
- .isoformat() → 转成标准格式字符串 "2026-06-02T12:43:18.090295"
- 为什么需要：记录"这份数据是什么时候转换的"

**from pathlib import Path**
- Path 是处理文件路径的类
- Path("a/b/c") → 创建一个路径对象
- .parent → 上一层目录
- .name → 文件名
- .resolve() → 转成绝对路径
- .exists() → 判断路径是否存在
- 为什么需要：比直接写字符串路径好用。Windows 用 \、Linux 用 /，Path 会自动处理

```python
# 不用 Path 的写法（麻烦）：
import os
path = os.path.join("data", "output", "file.jsonl")

# 用 Path 的写法（简单）：
path = Path("data") / "output" / "file.jsonl"
```

**from typing import Optional, List, Dict, Any**
- 这是**类型提示**，只给程序员看的，不影响代码运行
- Optional[str] → 这个变量可以是字符串，也可以是 None（空）
- List[str] → 这个变量是一个列表，每个元素是字符串
- Dict[str, Any] → 这个变量是一个字典，键是字符串，值可以是任何类型

```python
# 举个例子：
def say_hello(name: str) -> str:
    return "你好" + name

# name: str  → 意思是：name 参数应该是字符串
# -> str     → 意思是：返回值是字符串
# 你传 name=123（数字）也能运行，但 IDE 会警告你类型不对
```

**import pandas as pd**
- pandas 是第三方库（需 pip install pandas），最核心的 Python 数据分析库
- as pd 是别名，后续用 pd 代替 pandas
- 核心概念：**DataFrame**（数据框），可以想象成一张 Excel 表格

**import numpy as np**
- numpy 是第三方库，Python 数值计算的基础库
- np.random.choice(列表, 数量) → 从列表里随机选取
- np.random.seed(42) → 设置随机种子，保证每次生成的随机数一样

---

## 1.2 类定义

```python
class CSVToJSONLConverter:
    """CSV → JSONL 转换器，支持清洗、筛选和元数据注入"""
```

### 什么是类？

类（class）是一个模板。可以理解成**菜谱**。

```
        菜谱（类）
    ┌────────────────────┐
    │  材料（属性）       │
    │  步骤（方法）       │
    └────────────────────┘
             ↓
        面包（实例）
```

```python
c = CSVToJSONLConverter("input.csv")  # 创建实例（按菜谱做一份）
c.load()                               # 调用方法（开始第1步）
```

### """ 文档字符串

写在类或函数开头，告诉别人这是干嘛的，可以跨多行。

---

## 1.3 __init__ 方法（构造函数）

```python
def __init__(self, input_path: str):
    self.input_path = Path(input_path)
    self.df: Optional[pd.DataFrame] = None
    self.metadata: Dict[str, Any] = {
        "converted_at": datetime.now().isoformat(),
        "source_file": self.input_path.name,
    }
```

### def 是什么？

def 定义函数（function）。函数是一段可重复调用的代码块。

```python
def 函数名(参数):
    做事情
    return 结果
```

### __init__ 是什么？

类的特殊方法，**创建实例时自动调用**。

```python
c = CSVToJSONLConverter("input.csv")
# 写这一行的时候，Python 自动调用 __init__，参数 input_path = "input.csv"
```

### self 是什么？

self 代表"当前实例本身"。每个类的方法的第一个参数都是 self。

```python
class 学生:
    def __init__(self, 姓名):
        self.姓名 = 姓名
        self.成绩 = []

小明 = 学生("小明")
小红 = 学生("小红")
# 小明.成绩 和 小红.成绩 是两个不同的列表
```

### 逐行解释

```python
self.input_path = Path(input_path)
# Path("examples/sample_orders.csv") → 把字符串转成 Path 对象
# self.input_path 存起来，其他方法也能用
```

```python
self.df: Optional[pd.DataFrame] = None
# = None → 初始值是 None（空）
# 还没读文件，还没有数据，等 load() 调用了才会赋值
```

```python
self.metadata: Dict[str, Any] = {
    "converted_at": datetime.now().isoformat(),  # 当前时间
    "source_file": self.input_path.name,          # 文件名
}
# 记录：这份数据什么时候处理、来自哪个文件
```

---

## 1.4 load() 方法

```python
def load(self, **kwargs) -> "CSVToJSONLConverter":
    self.df = pd.read_csv(self.input_path, **kwargs)
    self.metadata["row_count"] = len(self.df)
    self.metadata["column_count"] = len(self.df.columns)
    self.metadata["columns"] = list(self.df.columns)
    print(f"  [✓] 加载 {len(self.df)} 行, {len(self.df.columns)} 列")
    return self
```

### **kwargs 是什么？

把"剩下的所有关键字参数"打包成一个字典。

```python
def test(**kwargs):
    print(kwargs)

test(name="小明", age=18)
# 输出: {"name": "小明", "age": 18}
```

本项目调用 load() 时没有传额外参数（**kwargs 是空的），但这样写让代码更灵活。

### -> "CSVToJSONLConverter" 是什么？

**返回值类型提示**。告诉别人：这个函数返回一个 CSVToJSONLConverter 类型的对象。
用引号是因为类还没完全定义完。

### 代码拆解

```python
self.df = pd.read_csv(self.input_path, **kwargs)
# pd.read_csv() → Pandas 读 CSV，返回一张表格（DataFrame）

self.metadata["row_count"] = len(self.df)
# len(DataFrame) → 行数

self.metadata["column_count"] = len(self.df.columns)
# df.columns → 列名列表，len(...) → 列的数量
```

```python
print(f"  [✓] 加载 {len(self.df)} 行, {len(self.df.columns)} 列")
# f"" → f-string，在引号前加 f，里面就能用 {} 嵌入变量
# 输出效果：  [✓] 加载 100 行, 5 列
```

```python
return self
# 返回实例本身
# 为了链式调用：c.load().fill_missing(...).export(...)
# 每一步返回 self，就能一步步点下去
```

---

## 1.5 fill_missing() 方法

```python
def fill_missing(self, strategy: dict) -> "CSVToJSONLConverter":
    before = self.df.isna().sum().sum()
    for col, method in strategy.items():
        if col not in self.df.columns:
            continue
        if method == "mean":
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif method == "median":
            self.df[col] = self.df[col].fillna(self.df[col].median())
        elif method == "mode":
            mode_val = self.df[col].mode()
            self.df[col] = self.df[col].fillna(mode_val[0] if not mode_val.empty else 0)
        elif method == "drop":
            self.df = self.df.dropna(subset=[col])
        else:
            self.df[col] = self.df[col].fillna(method)
    after = self.df.isna().sum().sum()
    print(f"  [✓] 缺失值填充: {before} → {after}")
    return self
```

### strategy 参数

```python
# strategy 是一个字典，比如：
{"customer_rating": "mean", "quantity": 1}
# customer_rating 列 → 用平均值填充空值
# quantity 列 → 用 1 填充空值
```

### .isna().sum().sum() 链式拆解

```python
# 数据：    age    score
# 0  20     90
# 1  NaN    80
# 2  30     NaN

self.df.isna()
# 判断每个格子是否为空
#      age    score
# 0  False   False
# 1   True   False
# 2  False    True

self.df.isna().sum()
# 对列求和，True=1, False=0
# age    1
# score  1

self.df.isna().sum().sum()
# 总数加起来 = 2
```

### for col, method in strategy.items()

```python
strategy = {"customer_rating": "mean", "quantity": 1}

strategy.items()
# → [("customer_rating", "mean"), ("quantity", 1)]

for col, method in strategy.items():
    # 第1轮：col="customer_rating", method="mean"
    # 第2轮：col="quantity", method=1
```

### continue

```python
if col not in self.df.columns:
    continue
# 如果这一列不存在，跳过当前循环，看下一列
```

### method 判断

```python
if method == "mean":
    # 平均值填充
    self.df[col] = self.df[col].fillna(self.df[col].mean())
    # .fillna(值) → 空值替换成指定值
    # .mean() → 平均值

elif method == "median":
    # 中位数填充（不受极端值影响）
    self.df[col] = self.df[col].fillna(self.df[col].median())

elif method == "mode":
    # 众数填充（出现次数最多的值）
    mode_val = self.df[col].mode()
    self.df[col] = self.df[col].fillna(mode_val[0] if not mode_val.empty else 0)
    # X if 条件 else Y → 三目运算符
    # 等价于：
    # if not mode_val.empty: fill_value = mode_val[0]
    # else: fill_value = 0

elif method == "drop":
    # 删除有空值的行
    self.df = self.df.dropna(subset=[col])

else:
    # 以上都不是，直接用 method 值填充
    self.df[col] = self.df[col].fillna(method)
```

---

## 1.6 select_columns() 和 add_record_id()

```python
def select_columns(self, columns: List[str]) -> "CSVToJSONLConverter":
    valid = [c for c in columns if c in self.df.columns]
    self.df = self.df[valid]
    print(f"  [✓] 保留列: {valid}")
    return self
```

### 列表推导式

```python
# 普通写法：
valid = []
for c in columns:
    if c in self.df.columns:
        valid.append(c)

# 列表推导式（一行）：
valid = [c for c in columns if c in self.df.columns]
#         ↑ 要生成的  ↑ for循环     ↑ 过滤条件
```

```python
def add_record_id(self, prefix: str = "REC"):
    ids = [f"{prefix}-{uuid.uuid4().hex[:8].upper()}" for _ in range(len(self.df))]
    self.df.insert(0, "record_id", ids)
    print(f"  [✓] 添加 {len(ids)} 个唯一ID，示例: {ids[0]}")
    return self
```

### for _ in range(...)

```python
# _ 表示"我不关心这个变量的值"
# range(100) → 0 到 99 共 100 个数

for _ in range(len(self.df)):
    # 循环行数次

# uuid.uuid4().hex → "a1b2c3d4e5f6g7h8i9j0..."
# [:8] → 取前8个 → "a1b2c3d4"
# .upper() → 转大写 → "A1B2C3D4"
# f"{prefix}-{...}" → "REC-A1B2C3D4"
```

self.df.insert(0, "record_id", ids) → 在第 0 列（最前面）插入一列叫 "record_id"。

---

## 1.7 export() 方法

```python
def export(self, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(exist_ok=True, parents=True)
    # 创建目录，不存在就创建

    with open(output, "w", encoding="utf-8") as f:
        # "w" = write 模式（覆盖写入）
        # as f → 把打开的文件叫做 f
        # with → 用完后自动关闭，不用手动 f.close()

        f.write(json.dumps({"#metadata": self.metadata}, ensure_ascii=False) + "\n")
        # 第1行：元数据

        for _, row in self.df.iterrows():
            # iterrows() → 一行一行遍历表格
            # _ → 行号（不关心）
            # row → 这一行数据（类似字典）

            record = {
                "record_id": row.get("record_id", uuid.uuid4().hex),
                # .get(键, 默认值) → 取这一行的值，没有就用默认值
                "data": {
                    k: (None if pd.isna(v) else v)
                    for k, v in row.items()
                    if k != "record_id"
                },
                # 字典推导式：遍历每一列
                # 跳过 record_id 列（已经单独放了）
                # 如果值是空的，写成 None（JSON 的 null）
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 每行写入一个 JSON，末尾加换行符

    size_kb = output.stat().st_size / 1024
    # .stat() → 文件信息，.st_size → 字节数，/1024 → KB
    print(f"  [✓] 导出: {output} ({size_kb:.1f} KB, {len(self.df)} 行)")
    return output
```

---

## 1.8 quick_convert() 和主入口

```python
def quick_convert(input_csv: str,
                  output_jsonl: str = "output/data.jsonl",
                  fill_strategy: Optional[dict] = None,
                  keep_cols: Optional[List[str]] = None,
                  add_ids: bool = True):
    # 参数默认值：不传 output_jsonl 就默认 "output/data.jsonl"

    c = CSVToJSONLConverter(input_csv)
    c.load()
    if fill_strategy:          # 如果不是 None
        c.fill_missing(fill_strategy)
    if keep_cols:
        c.select_columns(keep_cols)
    if add_ids:                # 如果 True
        c.add_record_id()
    return c.export(output_jsonl)
```

### if __name__ == "__main__":

```python
if __name__ == "__main__":
```

Python 中每个文件有个内置变量 __name__：
- 直接运行这个文件（`python converter.py`）→ __name__ = "__main__"
- 被其他文件 import → __name__ = "pipeline.converter"（模块名）

翻译：**"只有当我被直接运行时才执行下面的代码。"**

### 生成示例数据

```python
np.random.seed(42)  # 随机种子，保证每次生成的随机数一样

# :04d → 数字补零到4位，i=1 → "ORD-0001"
"order_id": [f"ORD-{i:04d}" for i in range(1, 101)],

# 从 4 种商品中随机选 100 次
"product": np.random.choice(["Laptop", "Mouse", "Keyboard", "Monitor"], 100),

# 1-9 的随机整数，100 个
"quantity": np.random.randint(1, 10, 100),

# 10-2000 的随机小数，保留2位
"price": np.random.uniform(10, 2000, 100).round(2),

# 从 [1,2,3,4,5,None] 按概率选，5% 的概率选到 None（模拟缺失值）
"customer_rating": np.random.choice(
    [1, 2, 3, 4, 5, None], 100,
    p=[0.02, 0.03, 0.1, 0.3, 0.5, 0.05]
),

sample.to_csv("examples/sample_orders.csv", index=False)
# index=False → 不把行号写进 CSV
```

---

# 第2部分：recorder.py — 交互记录器和校验器

## 2.1 导入部分

```python
import json              # JSON 处理
import subprocess         # 在 Python 里执行系统命令
import uuid               # 生成唯一 ID
import platform           # 获取操作系统信息
import sys                # 获取 Python 版本
import hashlib            # SHA256 哈希计算
import difflib            # 文本差异比较
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
```

### import subprocess — 最重要的导入

让 Python 能执行系统命令（如 git diff）。

```python
result = subprocess.run(
    "git rev-parse HEAD",       # 要执行的命令
    shell=True,                 # 通过系统的 shell 执行
    capture_output=True,        # 抓取输出，不打印到屏幕
    text=True,                  # 输出直接是字符串
)
print(result.stdout)            # 命令输出
print(result.returncode)        # 返回码（0=成功）
```

### import hashlib

```python
text = "你好世界"
text_bytes = text.encode()  # 字符串 → 字节
hash_obj = hashlib.sha256(text_bytes)  # 算 SHA256
hash_str = hash_obj.hexdigest()  # → "a1b2c3d4..."（64位） 
```

### import difflib

```python
a = ["第一行", "第二行"]
b = ["第一行", "第二行（改过）", "第三行"]

diff = difflib.unified_diff(a, b, lineterm='')
for line in diff:
    print(line)
# 输出类似 git diff 的格式
```

---

## 2.2 __init__

```python
def __init__(self, project_dir: str, task_id: Optional[str] = None):
    self.project_dir = Path(project_dir).resolve()
    # 相对路径 → 绝对路径

    if not self.project_dir.exists():
        raise FileNotFoundError(f"项目目录不存在: {project_dir}")
    # 如果不存在，主动报错

    self.task_id = task_id or f"TASK-{uuid.uuid4().hex[:8].upper()}"
    # 如果传了 task_id 就用传的，没传就自动生成

    self.record_id = f"REC-{uuid.uuid4().hex[:12].upper()}"
    self.steps: List[Dict[str, Any]] = []  # 空列表，后面逐步添加
    self._initial_commit: Optional[str] = None
    self._initial_file_content: str = ""
    self._start_time = datetime.now()  # 记录开始时间
```

---

## 2.3 _run() 方法 — 执行系统命令

```python
def _run(self, cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=self.project_dir,
            capture_output=True, timeout=30
        )

        for enc in ["utf-8", "gbk", "gb18030"]:
            try:
                text = result.stdout.decode(enc)
                # 尝试用这种编码解码
                return text.replace('\r\n', '\n').replace('\r', '').strip()
                # 清理换行符，去掉首尾空格
            except UnicodeDecodeError:
                continue  # 解码失败，试下一种编码

        # 都失败了，用 replace 模式兜底
        return result.stdout.decode("utf-8", errors="replace").strip()

    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {e}"
```

### try/except — 异常处理

```python
try:
    # 尝试执行这段代码
    可能报错的代码
except 某种错误:
    # 报错了执行这段
    处理错误的代码
```

### result.stdout.decode(enc) 

subprocess 返回的 stdout 是**字节**（bytes），不是字符串。

```python
b'hello'  # 字节类型
'hello'   # 字符串类型
b_hello.decode('utf-8')  # 字节 → 字符串
```

### .replace() 清理换行

```python
"hello\r\nworld".replace('\r\n', '\n')  # → "hello\nworld"
.replace('\r', '')                       # 再清一遍零散的 \r
.strip()                                 # 去掉首尾空白
```

---

## 2.4 _capture_git_state()

```python
def _capture_git_state(self) -> Optional[Dict[str, Any]]:
    if self._run("git rev-parse --is-inside-work-tree") != "true":
        return None  # 不在 git 仓库里，返回 None

    return {
        "commit_hash": self._run("git rev-parse HEAD"),
        "branch": self._run("git rev-parse --abbrev-ref HEAD"),
        "commit_message": self._run("git log -1 --format=%s"),
        "modified": [f for f in self._run("git diff --name-only").split("\n") if f],
        "untracked": [f for f in self._run("git ls-files --others --exclude-standard").split("\n") if f],
    }
```

### .split("\n") — 按换行拆成列表

```python
"a\nb\nc".split("\n")  # → ["a", "b", "c"]

# [f for f in list if f] → 过滤掉空字符串
[f for f in ["a", "", "b"] if f]  # → ["a", "b"]
```

---

## 2.5 snapshot_initial_state()

```python
def snapshot_initial_state(self) -> "InteractionRecorder":
    self._initial_commit = self._run("git rev-parse HEAD")
    # 拿到当前 Git commit 哈希

    self._initial_file_content = self._read_file("tasks/csv_reader.py")
    # 读取 csv_reader.py 的内容，存起来

    initial_hash = hashlib.sha256(self._initial_file_content.encode()).hexdigest()

    print(f"  [recorder] 初始状态: commit={self._initial_commit[:12]}, "
          f"hash={initial_hash[:12]}")
    return self
```

### 字符串切片 [:12]

```python
hash_str = "af349512cab7c1d2e3f4a5b6c7d8e9f0"
hash_str[:12]  # → "af349512cab7"（取前12个字符）
```

---

## 2.6 record() 和 record_run()

```python
def record(self, action: str, detail: Dict[str, Any]):
    self.steps.append({
        "step": len(self.steps) + 1,                  # 第几步
        "action": action,                               # 动作类型
        "timestamp": datetime.now().isoformat(),        # 时间戳
        **detail,  # 展开 detail 字典
    })
    return self
```

### **detail — 字典展开

```python
detail = {"file": "main.py", "change": "加了空行跳过"}
{
    "step": 1,
    "action": "edit_file",
    **detail,
}
# 等价于：
{
    "step": 1,
    "action": "edit_file",
    "file": "main.py",
    "change": "加了空行跳过"
}
```

```python
def record_run(self, cmd: str):
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=self.project_dir,
            capture_output=True, text=True, timeout=60
        )
        detail = {
            "cmd": cmd,
            "exit_code": result.returncode,  # 0=成功，非0=失败
            "stdout": (result.stdout or "").strip()[-500:],
            "stderr": (result.stderr or "").strip()[-500:],
        }
    except Exception as e:
        detail = {"cmd": cmd, "exit_code": -1, "stdout": "", "stderr": str(e)}

    self.record("run", detail)  # 也记录到交互步骤中
    return detail
```

---

## 2.7 finalize() — 组装最终输出

```python
def finalize(self, task_type: str = "feature_dev"):
    final_state = self._capture_git_state()
    unified_diff = self._generate_unified_diff()

    final_content = self._read_file("tasks/csv_reader.py")
    final_hash = hashlib.sha256(final_content.encode()).hexdigest()
    initial_hash = hashlib.sha256(self._initial_file_content.encode()).hexdigest()

    record = {
        "record_id": self.record_id,
        "task_id": self.task_id,
        "task_type": task_type,
        "env_info": self._capture_env_info(),
        "git_context": {
            "initial_commit": self._initial_commit,
            "unified_diff": unified_diff,
            "diff_length": len(unified_diff),
        },
        "file_snapshots": {
            "initial_content": self._initial_file_content,
            "final_content": final_content,
            "initial_hash": initial_hash,
            "final_hash": final_hash,
        },
        "interaction_trajectory": {
            "total_steps": len(self.steps),
            "duration_seconds": round(
                (datetime.now() - self._start_time).total_seconds(), 2
            ),
            "steps": self.steps,
        },
    }
    return record
```

round(数字, 2) → 保留2位小数。
(datetime.now() - start_time).total_seconds() → 算耗时。

---

## 2.8 ConsistencyValidator 类

### @staticmethod

```python
@staticmethod
def validate_from_record(record, project_dir=""):
    ...
```

**静态方法**：不需要创建实例就能调用。

```python
# 普通方法：
c = ConsistencyValidator()
c.validate_from_record(record)

# 静态方法：
ConsistencyValidator.validate_from_record(record)  # 直接调
```

### record.get("key", {})

```python
record.get("file_snapshots", {})
# 取键 "file_snapshots" 的值
# 如果键不存在，返回默认值 {}（空字典），不报错
# 跟 record["key"] 的区别：[] 取不存在的键会报错
```

### any() 函数

```python
any([False, False, True, False])  # → True（只要有一个True就返回True）
any([False, False, False])        # → False
```

### set() 集合

```python
s = set()
s.add("hello")
s.add("world")
s.add("hello")  # 重复添加，但集合里只有一个 "hello"
print(s)  # {"hello", "world"}
# 集合自动去重、不关心顺序
```

---

# 第3部分：run_agent_demo.py

## 3.1 路径设置

```python
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
```

__file__ → 当前文件的路径。
.resolve() → 转成绝对路径。
.parent → 上一层目录。
.parent.parent → 项目根目录。

sys.path → Python 的模块搜索路径列表。
insert(0, ...) → 把项目根目录插入到最前面，这样 import 就能找到。

---

## 3.2 其他常用语法

### glob("*") — 遍历目录

```python
for f in tasks_dir.glob("*"):
    if f.is_file():
        f.unlink()
# glob("*") → 匹配所有文件和文件夹
# is_file() → 判断是不是文件
# unlink() → 删除文件
```

### write_text() — 写文件

```python
csv_file.write_text(task.initial_code, encoding="utf-8", newline="\n")
# Path 对象的快捷写入方法
# newline="\n" → 强制用 Linux 换行符，避免 Windows \r\n 污染
```

### json.loads() — 解析 JSON

```python
json.loads('{"name": "小明"}')  # → {"name": "小明"}（Python 字典）
json.dumps({"name": "小明"}, ensure_ascii=False)  # → '{"name": "小明"}'
```

---

# 第4部分：tasks.py — 任务模板

```python
from dataclasses import dataclass

@dataclass
class AgentTask:
    task_id: str
    task_type: str
    title: str
    description: str
    initial_code: str
    expected_code: str
    verification_cmd: str
```

@dataclass → 自动生成 __init__ 方法。

```python
# 等价于手写：
class AgentTask:
    def __init__(self, task_id, task_type, title, ...):
        self.task_id = task_id
        self.task_type = task_type
        ...
```

---

# 第5部分：你实际用到的 Git 命令

| 命令 | 作用 | 输出示例 |
|------|------|---------|
| git rev-parse HEAD | 当前 commit 的哈希 | a8498cc4e... |
| git rev-parse --abbrev-ref HEAD | 当前分支名 | main |
| git log -1 --format=%s | 最近一次 commit 标题 | tmp: init task |
| git diff --name-only | 修改过的文件名 | tasks/csv_reader.py |
| git add tasks/ | 把变更存入暂存区 | 无输出 |
| git commit -m "msg" | 创建新 commit | (commit 信息) |
| git diff --no-color -- tasks/ | 只看 tasks/ 目录的 diff | (diff 内容) |
