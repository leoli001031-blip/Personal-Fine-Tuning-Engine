# PFE 数据分层策略

更新时间：2026-04-16

## 目标

PFE 不把所有“用户说过的话”都直接训练进模型。

更合理的做法是把数据拆成 5 条不同的处理路径：

1. `memory`：稳定事实，供长期记忆与 prompt 注入
2. `profile`：长期偏好，供画像、路由、策略与风格注入
3. `prompt_context`：一次性上下文，只在当前会话生效
4. `signal -> sample -> training`：通过行为反馈筛选出的训练数据
5. `discard`：高风险 PII、密钥、一次性敏感信息，不进入训练

这个策略的目的不是减少个性化，而是避免把“应该存在记忆层的数据”错误地固化进模型权重。

## 一句话原则

- 用户是谁：优先进 `memory`
- 用户长期喜欢什么：优先进 `profile`
- 用户希望你怎么回答：先进 `profile`，再由反馈决定是否进入训练
- 当前这次任务的临时背景：只进 `prompt_context`
- 明确的接受/拒绝/编辑行为：优先作为 `signal`
- 高风险隐私和密钥：直接 `discard`

## 数据分流总表

| 数据类型 | 例子 | 主路径 | 是否直接训练 | 说明 |
|------|------|------|------|------|
| 身份事实 | “我叫小王” | `memory` | 否 | 这是用户事实，不是模型能力 |
| 角色事实 | “我是程序员” | `memory` | 否 | 更适合画像和上下文注入 |
| 长期兴趣/口味 | “我喜欢直接一点的回答” | `profile` | 否，先观察 | 先记偏好，再看是否长期稳定 |
| 回答风格偏好 | “希望你以后更温和、更鼓励式” | `profile` + `signal` | 条件式 | 先 prompt 生效，再通过反馈变成训练目标 |
| 一次性任务背景 | “我今天要给老板写邮件” | `prompt_context` | 否 | 这是会话上下文，不该进权重 |
| 用户接受回答 | 用户继续对话、不修改 | `signal -> SFT` | 是 | 高置信正样本 |
| 用户大幅编辑后接受 | 改写后继续使用 | `signal -> DPO/SFT` | 是 | 优先 DPO，缺链路时降级 SFT |
| 用户拒绝/重生成 | 删除、要求重答 | `signal -> DPO rejected` | 不能单独训练 | 需要和 accepted/edited 样本配对 |
| 高风险敏感信息 | 身份证、银行卡、密码、API Key | `discard` | 否 | 不能进入训练或长期记忆 |

## 显式用户信息的默认规则

### 1. 身份事实

例子：

- “我叫小王”
- “我的名字是 Alice”
- “我是老师”
- “我是一名后端工程师”

规则：

- 默认写入 `memory`
- 允许用于 prompt 注入
- 不直接进入训练样本

原因：

- 这类信息属于“用户数据库”，不是模型的通用风格能力
- 它可能会变，也可能只对当前用户有效
- 直接训练进模型会造成权重污染和隐私风险

### 2. 长期偏好

例子：

- “我喜欢简洁一点”
- “我不喜欢太空泛的回答”
- “我更偏好结构化输出”

规则：

- 默认写入 `profile`
- 允许用于 prompt 注入
- 不直接训练
- 如果后续被多次反馈强化，可升级成训练候选目标

### 3. 回答风格/人格偏好

例子：

- “希望你以后温和一点”
- “请你直接一点，不要绕”
- “我希望你像教练一样回应我”

规则：

- 先进入 `profile`
- 同时作为 `signal` 观察目标
- 当用户后续通过 accept / edit / reject 长期强化这个偏好时，升级为训练候选

原因：

- 这类数据很像“训练目标”，但不该因为用户说了一次就立即写进权重
- 更稳妥的策略是“显式声明 + 行为强化”后再训练

### 4. 一次性上下文

例子：

- “我今天要面试”
- “我现在在写周报”
- “这次是给老板发邮件”

规则：

- 只进入 `prompt_context`
- 默认不持久化，不训练

原因：

- 这类信息只对当前会话有意义
- 如果训练进权重，会造成噪声和误泛化

## 反馈信号如何进入训练

### SFT 候选

以下信号可以直接进入 SFT 候选：

- `accept`
- `copy`
- `edit` 但无法可靠配成 DPO 时的 edited text

要求：

- 置信度达到阈值
- 至少保留 `context + model_output`
- 不含高风险 PII

### DPO 候选

以下信号优先进入 DPO：

- 用户大幅编辑并接受
- 用户拒绝后重生成，再接受新版

要求：

- 必须有完整事件链：`session_id + request_id + source_event_ids`
- 必须能明确分出 `chosen` 和 `rejected`
- 如果链路不完整，不允许猜配对

### 不能直接训练的负信号

以下信号不能单独训练：

- `reject`
- `regenerate`

它们的作用是：

- 作为负样本证据保留
- 等待和 accepted/edited 的正样本配对
- 配对成功后进入 DPO

## 默认禁止训练的数据

以下内容默认不得进入训练：

- 身份证号
- 银行卡号
- 密码
- 私钥
- API Key
- 令牌
- 电话
- 邮箱
- 精确住址

这些内容也不建议长期写入普通 memory，除非用户明确授权并且有更细粒度的安全存储策略。

## 推荐落地方式

### 层 1：Memory / Profile

使用：

- `pfe_core.user_memory`
- `pfe_core.profile_extractor`

职责：

- 存名字、职业、兴趣、长期偏好
- 生成可注入 prompt 的用户画像

### 层 2：Signal

使用：

- `pfe_core.collector.chat_collector`
- `/pfe/feedback`
- `/pfe/signal`

职责：

- 接受/拒绝/编辑/重生成
- 保留事件链和 provenance

### 层 3：Training

使用：

- `pfe_core.curator.datasets`
- `pfe_core.trainer.dpo_dataset`
- `pfe_core.curator.distillation`

职责：

- 按规则把 signal 转成 SFT / DPO 样本
- 做置信度过滤、PII 过滤、事件链检查
- 当 signal metadata 中带有显式 `response_preference` 且被后续 accept / edit 行为强化时，
  给样本打上 `explicit_response_preference_reinforced=true` 和 `training_gate_reason`
- auto-train trigger 会把这类 `preference_reinforced` 样本单独计数，并按
  `preference_reinforced_sample_weight` 做加权，优先让“被显式声明且被行为验证过的偏好”
  更早进入训练调度

## 当前仓库中的对应实现

- `pfe_core.user_memory`：已经能提取名字、职业、偏好等显式事实
- `pfe_core.profile_extractor`：已经能从隐式信号提取风格、领域和互动模式
- `pfe_core.collector.chat_collector`：已经能提取 accept / edit / reject / regenerate
- `pfe_core.trainer.dpo_dataset`：已经能把 accepted / rejected / edited 信号转成 DPO 对
- `pfe_core.data_policy`：新增的统一策略模块，用于决定一条数据应去哪一层
- `pfe_core.curator.distillation`：已开始把“显式风格偏好 + 行为强化”标记为 `preference_reinforced` 训练候选

## 版本 1 的保守策略

为了避免过早把噪声训练进模型，建议默认采用以下策略：

1. 身份事实只进 memory
2. 风格偏好先只进 profile
3. 只有当风格偏好被显式声明且被后续反馈反复强化时，才升级为训练目标
4. reject / regenerate 只做负证据，不单独训练
5. DPO 必须要求完整事件链
6. 高风险 PII 永远不进训练

## 下一步建议

1. 在 `ChatCollector` 后增加显式 datum 提取步骤，把用户消息拆成 `identity_fact / role_fact / response_preference / ephemeral_context`
2. 在 curator 中接入 `data_policy.route_user_datum()`，统一 memory/profile/training 的分流
3. 在 sample metadata 中新增 `data_lane` / `training_gate_reason`
4. 对“回答风格偏好”增加“显式声明 + 行为强化”双重门槛
