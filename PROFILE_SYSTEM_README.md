# PFE Phase 2 用户画像系统

## 概述

用户画像系统从历史信号中构建用户画像，用于指导数据整理、训练策略和评估。

## 实现文件

### 核心模块

1. **`pfe-core/pfe_core/user_profile.py`**
   - `UserProfile` - 用户画像数据类，包含风格偏好、领域偏好、交互模式
   - `PreferenceScore` - 偏好分数，支持增量更新
   - `UserProfileStore` - 画像存储管理

2. **`pfe-core/pfe_core/profile_extractor.py`**
   - `ProfileExtractor` - 从 ImplicitSignal 提取画像
   - 支持领域分类、风格检测、交互模式识别
   - 增量更新画像（不重新计算全部）

### 集成模块

3. **`pfe-core/pfe_core/curator/distillation.py`**
   - 新增画像优先级数据整理
   - `_compute_profile_match_score()` - 计算样本与画像匹配度
   - `_apply_profile_prioritization()` - 提升匹配样本的分数

4. **`pfe-core/pfe_core/trainer/service.py`**
   - 新增 `_get_profile_adjusted_params()` - 根据画像调整训练参数
   - 支持根据用户领域偏好排序训练样本
   - 根据交互模式调整 replay ratio

5. **`pfe-core/pfe_core/evaluator/judge.py`**
   - 新增个性化匹配度评估
   - `_compute_profile_match_score()` - 计算输出与画像匹配度
   - 支持 profile_match 评分维度

### CLI 命令

6. **`pfe-cli/pfe_cli/main.py`**
   - `pfe profile show [--user-id]` - 查看画像
   - `pfe profile analyze` - 手动触发画像分析
   - `pfe profile export/import` - 画像导出导入
   - `pfe profile list` - 列出所有画像
   - `pfe profile delete` - 删除画像

## 使用方法

### 查看用户画像

```bash
pfe profile show --user-id default
pfe profile show --user-id default --json
```

### 手动触发画像分析

```bash
pfe profile analyze --user-id default
pfe profile analyze --user-id default --full  # 完整分析，非增量
```

### 导出/导入画像

```bash
pfe profile export --user-id default --output profile.json
pfe profile import --user-id new_user --input profile.json
```

### 列出所有画像

```bash
pfe profile list
```

### 删除画像

```bash
pfe profile delete --user-id default --yes
```

## 画像模型

```python
@dataclass
class UserProfile:
    user_id: str
    style_preferences: dict[str, PreferenceScore]  # 正式-随意，简洁-详细等
    domain_preferences: dict[str, PreferenceScore]  # 编程、写作、学习等
    interaction_patterns: dict[str, PreferenceScore]  # 喜欢例子、直接答案等
    profile_summary: str  # 画像摘要
    dominant_style: str  # 主导风格
    dominant_domains: list[str]  # 主导领域
```

## 领域分类

- `programming` - 编程
- `writing` - 写作
- `learning` - 学习
- `analysis` - 分析
- `creative` - 创意
- `business` - 商业

## 风格检测

- `formal` / `casual` - 正式/随意
- `concise` / `detailed` - 简洁/详细
- `technical` / `non_technical` - 技术/非技术

## 交互模式

- `likes_examples` - 喜欢例子
- `prefers_direct` - 偏好直接答案
- `wants_reasoning` - 关注推理过程
- `prefers_code` - 偏好代码
- `wants_alternatives` - 喜欢多种方案
- `seeks_validation` - 寻求确认

## 测试

```bash
# 运行画像系统测试
PYTHONPATH=/Users/zcc/Desktop/pfe/pfe-core:$PYTHONPATH python3 -c "
from pfe_core.user_profile import UserProfile, get_user_profile_store
from pfe_core.profile_extractor import ProfileExtractor

# 创建画像
store = get_user_profile_store()
profile = store.get_profile('test')
profile.update_domain_preference('programming', 0.9)
profile.update_style_preference('technical', 0.8)
profile.compute_dominant_traits()
print(profile.format_for_prompt())
"
```
