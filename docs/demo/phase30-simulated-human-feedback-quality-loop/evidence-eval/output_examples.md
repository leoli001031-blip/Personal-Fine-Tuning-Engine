# Phase30 Output Examples

## Base

- Scores: `{"citation_exact_match_rate": 0.5, "concise_output_rate": 0.35, "four_section_exact_rate": 0.0, "hard_negative_contrast_score": 0.0, "manual_confirmation_rate": 0.0, "missing_info_first_rate": 0.7, "no_external_law_rate": 0.75, "no_legal_conclusion_rate": 0.9, "preference_adherence_rate": 0.0}`

```text
### 摘要
本合同规定了付款义务的条款，包括付款期限和付款方式。客户需在发票日后三十日内付款，逾期服务可能暂停。

### 风险提示
在付款期限内，客户需要确保及时支付款项，否则可能会面临服务暂停的风险。

### 引用依据
- [phase30-holdout-source-001:phase30-holdout-chunk-001](https://www.example.com/document/phase30-holdout-source-001)

### 人工确认
无。
```

```text
**服务可用性目标和服务补偿可能写在订单或附件中。**  
**缺少订单正文、附件范围和变更记录。**
```

```text
### 四段式总结

#### 摘要
本合同终止条款旨在明确在任何一方发生重大违约时，另一方有权终止协议。该条款适用于重大违约情况，包括但不限于违反合同义务、未能履行合同义务或未按约定提供服务等。

#### 风险提示
在签订此类合同时，双方应仔细阅读并理解所有条款，特别是关于终止条款的详细规定。如果对条款存在误解或疑虑，应及时与法律顾问沟通，以确保合同的准确性和有效性。

#### 引用依据
- [phase30-holdout-source-003:phase30-holdout-chunk-003](https://www.example.com/phase30-holdout-source-003
```

## Adapter

- Scores: `{"citation_exact_match_rate": 0.35, "concise_output_rate": 0.15, "four_section_exact_rate": 0.0, "hard_negative_contrast_score": 0.0, "manual_confirmation_rate": 0.0, "missing_info_first_rate": 0.7, "no_external_law_rate": 0.9, "no_legal_conclusion_rate": 0.9, "preference_adherence_rate": 0.0}`

```text
### 摘要

本合同规定了付款义务的条款，包括付款期限和付款方式。客户需在发票日后三十日内付款，逾期服务可能暂停。

### 风险提示

1. **付款期限**：客户需在发票日后三十日内付款，逾期服务可能暂停。
2. **付款方式**：客户需按照合同约定的方式支付款项，否则可能影响服务的正常进行。

### 引用依据

- [phase30-holdout-source-001:phase30-holdout-chunk-001](https://www.example.com/phase30-holdout-source-001)
```

```text
**服务可用性目标和服务补偿可能写在订单或附件中。**  
**但缺少订单正文、附件范围和变更记录。**
```

```text
### 四段式整理报告

#### 摘要
本报告旨在整理和分析两段资料中的冲突，并提供可能的解决方案。资料中关于终止条款的描述存在差异，需进行人工确认以确保准确理解。

#### 风险提示
1. **资料来源一致性**：确保所有资料来源一致，避免因信息不一致导致的误解。
2. **法律咨询**：在处理涉及法律问题时，应寻求专业律师的帮助，以确保所有条款的准确性和合法性。
3. **数据准确性**：核实所有数据来源的准确性，特别是对于涉及敏感信息的资料。

#### 引用依据
- [phase30-holdout-source-003:phase30-holdout-chunk-
```
