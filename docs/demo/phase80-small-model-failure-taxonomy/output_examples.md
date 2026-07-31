# Phase80 Output Examples

These are real local diagnostic outputs on fresh simulated_usage sessions. They compare learning rate, decoding stop control, and model capacity; they are not product-benefit evidence.

## phase80-evidence_truthfulness-01

### base_0_5b_minimal

证书指纹和到期时间还没有被核验，这可能是导致证书轮换失败的原因之一。建议你检查证书指纹和到期时间是否正确，并且确保它们没有被修改或丢失。如果问题仍然存在，请提供更多的上下文信息，以便进一步诊断。

### runtime_0_5b

下一步：检查证书指纹和到期时间是否正确。

### phase79_high_lr_adapter

可以。

### phase80_low_lr_adapter

可以的。

### phase79_high_lr_stop_control

无法确定。建议联系技术支持或管理员获取帮助。

### base_4b_minimal

证书轮换后部署脚本退出码为 0，但线上端点仍返回旧证书，说明证书未成功替换。下一步应**重载服务**并**复测端点**。

### runtime_4b

依据：证书轮换后未核验指纹和到期时间，无法确认证书更新状态。
下一步：重载服务后复测端点。

## phase80-provenance_labeling-01

### base_0_5b_minimal

来源结论：脚本模拟了证书轮换并得到“部署脚本退出为 0，但线上端点仍返回旧证书”。人工验收动作：检查证书是否已成功轮换。

### runtime_0_5b

结论：脚本模拟了证书轮换并得到“部署脚本退出为 0，但线上端点仍返回旧证书”。 实际上，脚本未能成功完成证书轮换，导致部署脚本退出并返回旧证书。

### phase79_high_lr_adapter

source: 脚本模拟了证书轮换并得到“部署脚本退出为 0，但线上端点仍返回旧证书”
action: 确认脚本是否正常运行
source: 人工验收动作：检查证书是否已更新
action: 检查证书是否已更新

### phase80_low_lr_adapter

source conclusion: 脚本模拟了证书轮换并得到“部署脚本退出为 0，但线上端点仍返回旧证书”。人工验收动作：检查证书是否已成功轮换。

### phase79_high_lr_stop_control

source: 脚本模拟了新证书的生成过程。
action: 等待证书生成完成后再进行验证。
source: 检查在线端点是否能正常获取到新的证书。
action: 在线端点发送新证书，并检查是否有错误信息或异常情况发生。

### base_4b_minimal

**来源结论：**
该情况应标记为 `simulated_usage`，因为它是通过脚本模拟的证书轮换过程，未涉及真实用户实际使用场景或反馈，不能作为 `actual_user_feedback`。

---

**人工验收动作：**
人工检查线上服务的证书配置是否已正确更新，并验证服务端在新证书生效后是否正常响应请求，确保证书轮换流程在实际环境中有效执行。

### runtime_4b

依据：用户指示仅能使用 simulated_usage 作为证据来源。
结论：该情况应标记为 simulated_usage。
下一步：人工验证模拟环境中的证书轮换逻辑与真实环境配置是否一致。

## phase80-concise_workstyle-01

### base_0_5b_minimal

部署脚本退出为 0，但线上端点仍返回旧证书。建议立即进行证书轮换。

### runtime_0_5b

部署脚本退出为 0，但线上端点仍返回旧证书。建议立即进行证书轮换。

### phase79_high_lr_adapter

结论：部署脚本退出为 0，但线上端点仍返回旧证书。
依据：部署脚本退出为 0，但线上端点仍返回旧证书。
下一步：请相关团队成员检查并确认证书是否已正确更新。

### phase80_low_lr_adapter

部署脚本退出为 0，但线上端点仍返回旧证书。

### phase79_high_lr_stop_control

**结论**: 布局脚本成功完成证书更新任务。
**依据**: 系统重启后网络连接正常且无新证书生成。
**下一步**: 检查是否存在其他可能导致该问题的因素并进行相应处理。

### base_4b_minimal

证书轮换失败，线上端点仍使用旧证书。
依据：部署脚本退出码为 0，但线上端点返回旧证书。
下一步：检查部署脚本是否实际替换证书文件。
下一步：验证服务是否重新加载或重启。
下一步：检查配置文件是否更新证书路径。
下一步：联调确认证书状态及服务响应。

### runtime_4b

证书轮换失败，线上端点仍使用旧证书。
依据：部署脚本退出码为 0，但线上端点返回旧证书。
下一步：检查部署脚本是否实际替换证书文件。
下一步：验证服务是否重新加载或重启。
下一步：检查配置文件是否更新证书路径。
下一步：联调确认证书状态及服务响应。

## phase80-ordinary_direct-01

### base_0_5b_minimal

The certificate hasn't been renewed yet.

### runtime_0_5b

The certificate hasn't been renewed yet.

### phase79_high_lr_adapter

The certificate hasn't been renewed yet.

### phase80_low_lr_adapter

The certificate hasn't been renewed yet.

### phase79_high_lr_stop_control

certificate status in English.

### base_4b_minimal

The certificate is not yet replaced.

### runtime_4b

Certificate status: not yet replaced.
