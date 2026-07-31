# PFE Phase2 网页文案块

这些文案用于外部展示页。每段都尽量短，方便直接贴进页面组件。

## Hero

标题：

```text
PFE / 本地模型工作台
```

一句话：

```text
local 命中目标记忆，base 未命中；训练、评估、推广和 API 接入都有真实证据。
```

状态豆：

```text
已完成
```

主按钮：

```text
查看证据
```

次按钮：

```text
素材包
```

## 结果对比

标题：

```text
同题对比
```

说明：

```text
同一个记忆问题，base 没有答出目标代号，local adapter 命中。
```

对比行：

```text
base  /  记忆代号是“Memory”。  /  未命中
local /  DEMO-PHASE2-042        /  命中
```

## 闭环过程

标题：

```text
训练到接入
```

四个节点：

```text
生成版本：真实 PEFT 训练完成，产出 adapter 20260618-001。
评估闸门：recommendation=deploy，promotion gate 放行。
设为当前：adapter 已推广并被 Studio 加载。
API 接入：聊天 API 和反馈 API 都有可复制 handoff。
```

## 证据清单

标题：

```text
真实证据
```

说明：

```text
截图、命令输出和 JSON response 都来自同一次干净 Demo 彩排。
```

证据条：

```text
截图 / studio-workbench.png / 产品形态
截图 / base-local-proof.png / 效果对比
日志 / demo-phase2-smoke.txt / 已通过
JSON / demo-eval-summary.json / deploy
JSON / demo-promote-summary.json / loaded
JSON / base-vs-local-summary.json / 命中对比
```
