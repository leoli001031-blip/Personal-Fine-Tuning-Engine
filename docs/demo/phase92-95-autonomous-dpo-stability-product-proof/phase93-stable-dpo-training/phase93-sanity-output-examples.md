# Phase93 Sanity Output Examples

All examples are real local model outputs from simulated usage sessions. No actual user feedback is present.

## Exact Three-Line: Phase89

```text
当前唯一不确定信息是：复核查阅人姓名和联系方式是否已经填写。

结论：本次回复未通过固定模板校验，目前状态为未验证。
根据：未通过校检验收结果，无法确认任务完成情况。
下一步：需手动补充完整相关信息后进行再次提交。
```

The output is not an exact three-line block because it adds text before the block and uses `根据` instead of `依据`.

## Exact Three-Line: 12-Step DPO

```text
当前唯一信息为：复核日未填。

结论：本次处理未通过校检，当前状态待核实。
依据：生成结果不符合预定格式要求。
下一步：重试生成，并进行人工审核以确保符合标准。
```

The output still adds text before the three-line block, so the frozen raw-output scorer correctly records a miss.

## Provenance: Phase89

```text
模拟使用（simulated_usage）是指根据预设规则自动生成的内容，而实际用户反馈（actual_user_feedback）则是由真实用户提供的意见或评价。

两者之间的关系是：

- 如果生成的内容符合预期且无误，则可以认为其为真实的用户反馈；
- 反之，如果生成的内容不符合预期或存在错误，则应视为模拟使用的结果，并非真正的用户反馈。

总结来说，模拟使用的结果需要进一步核实以确定是否具有真实性。
```

## Provenance: 12-Step DPO

```text
模拟使用（simulated_usage）是指根据预设规则自动生成的数据，用于测试系统功能或评估算法性能等目的。

实际用户反馈（actual_user_feedback）则是指由真实用户直接给出的意见、评价或建议，这些意见对产品改进具有重要价值。

两者的关系是：

- 模拟使用通常不会包含真实的用户反馈，因为其目的是为了检验系统的稳定性和准确性，而不是收集用户的直接体验。
- 实际用户反馈则需要结合模拟使用的结果进行综合分析，以确保产品的整体表现符合预期目标。

综上所述，模拟使用和实际用户反馈之间存在明显的区别，前者侧重于技术验证，后者关注用户体验及产品优化方向。
```

The candidate is more semantically cautious, but it does not state the frozen explicit negative boundary required by the provenance scorer. It therefore does not receive a manufactured improvement.
