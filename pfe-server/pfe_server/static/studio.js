const state = {
  runtime: null,
  workspaces: null,
  models: null,
  handoff: null,
  handoffTest: null,
  adapters: null,
  readiness: null,
  evalStatus: null,
  trainingPreflight: null,
  trainingJobs: null,
  status: null,
  errors: [],
};

const $ = (id) => document.getElementById(id);

function text(id, value) {
  const node = $(id);
  if (node) node.textContent = value == null || value === "" ? "-" : String(value);
}

function pill(id, label, tone) {
  const node = $(id);
  if (!node) return;
  node.textContent = label;
  node.className = "status-pill" + (tone ? " " + tone : "");
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.remove("show"), 1900);
}

async function loadJson(path) {
  const response = await fetch(path, { headers: { "accept": "application/json" } });
  if (!response.ok) throw new Error(path + " " + response.status);
  return response.json();
}

async function sendJson(path, body) {
  const response = await fetch(path, {
    method: "PUT",
    headers: {
      "accept": "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload && payload.error ? payload.error : path + " " + response.status);
  return payload;
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "accept": "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload && payload.detail ? payload.detail : path + " " + response.status);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function apiContract() {
  const runtime = state.runtime || {};
  const api = runtime.api && typeof runtime.api === "object" ? runtime.api : {};
  return {
    url: api.chat_completions_url || runtime.api_url || "",
    method: api.method || "POST",
    model: api.model_parameter || "base",
    feedbackUrl: api.feedback_url || "",
    authHeader: api.auth_header || "",
    contentType: api.content_type || "application/json",
  };
}

function buildApiExample() {
  const contract = apiContract();
  if (!contract.url) return "检查中";
  const lines = [
    "curl " + contract.url + " \\",
    "  -H \"content-type: " + contract.contentType + "\" \\",
  ];
  if (contract.authHeader) {
    lines.push("  -H \"" + contract.authHeader + "\" \\");
  }
  lines.push(
    "  -d '{\"model\":\"" + contract.model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
  );
  return lines.join("\n");
}

function currentModelCandidate() {
  const models = state.models || {};
  const candidates = Array.isArray(models.candidates) ? models.candidates : [];
  return candidates.find((candidate) => candidate.id === models.selected) || candidates[0] || {};
}

function buildHandoffSnapshot() {
  const runtime = state.runtime || {};
  const models = state.models || {};
  const adapters = state.adapters || {};
  const current = adapters.current || {};
  const candidate = currentModelCandidate();
  const contract = apiContract();
  return {
    urls: {
      web: runtime.web_url || "",
      api: contract.url || runtime.api_url || "",
      feedback: contract.feedbackUrl || (runtime.base_url ? runtime.base_url + "/pfe/feedback" : ""),
      studio: runtime.studio_url || "",
      dashboard: runtime.dashboard_url || "",
    },
    model: {
      selected: models.selected || candidate.id || "",
      label: candidate.label || models.selected_label || models.selected || "",
      api_parameter: contract.model || "base",
    },
    version: {
      current: current.version || adapters.latest_version || "",
      latest: adapters.latest_version || "",
      count: adapters.count || 0,
      pending_count: Array.isArray(adapters.pending) ? adapters.pending.length : 0,
    },
    runtime: {
      access_scope: runtime.access_scope || "仅本机",
      auth_mode: runtime.auth_mode || "",
    },
    closed_loop: {
      required_response_fields: ["session_id", "request_id"],
      feedback: {
        url: contract.feedbackUrl || (runtime.base_url ? runtime.base_url + "/pfe/feedback" : ""),
        actions: ["accept", "reject", "edit", "regenerate", "delete"],
      },
    },
  };
}

function currentHandoff() {
  return state.handoff && state.handoff.kind === "pfe_studio_handoff"
    ? state.handoff
    : buildHandoffSnapshot();
}

function buildHandoffText() {
  const handoff = currentHandoff();
  if (handoff.copy_text) return handoff.copy_text;
  const urls = handoff.urls || {};
  const model = handoff.model || {};
  const version = handoff.version || {};
  const closedLoop = handoff.closed_loop || {};
  const feedback = closedLoop.feedback || {};
  return [
    "PFE closed-loop handoff",
    "Web: " + (urls.web || "-"),
    "Chat API: " + (urls.api || "-"),
    "Feedback API: " + (urls.feedback || feedback.url || "-"),
    "Model parameter: " + (model.api_parameter || "base"),
    "Selected model: " + (model.selected || "-"),
    "Keep per answer: " + ((closedLoop.required_response_fields || ["session_id", "request_id"]).join(", ")),
    "Report actions: " + (((feedback && feedback.actions) || ["accept", "reject", "edit", "regenerate", "delete"]).join(", ")),
    "Current version: " + (version.current || "-"),
  ].join("\n");
}

const issueCopy = {
  needs_local_path: "先选择模型文件夹",
  real_local_inference_disabled: "点“本地回复”后生效",
  runtime_dependencies_missing: "本机推理依赖未安装",
  missing: "找不到这个模型文件夹",
  model_path_not_found: "找不到这个模型文件夹",
  invalid_workspace_name: "工作区名称格式不对",
};

function humanIssue(value) {
  const key = String(value || "");
  if (!key) return "";
  return issueCopy[key] || key.replace(/_/g, " ");
}

function humanIssueList(values) {
  return (Array.isArray(values) ? values : [])
    .map(humanIssue)
    .filter(Boolean);
}

function readinessBlockers(readiness) {
  return readiness && readiness.summary && Array.isArray(readiness.summary.blockers)
    ? readiness.summary.blockers
    : [];
}

function summaryTextFor(readiness) {
  const summary = readiness.summary || {};
  const blockers = readinessBlockers(readiness);
  if (blockers.includes("needs_local_path") || blockers.includes("missing") || blockers.includes("model_path_not_found")) {
    return "先选择模型文件夹，然后复制 API 或网页地址。";
  }
  if (blockers.includes("real_local_inference_disabled")) {
    return "API 和网页地址已就绪；需要本地回复时再打开本地模型。";
  }
  if (blockers.includes("runtime_dependencies_missing")) {
    return "API 和网页地址可用；本地模型回复还需要安装推理依赖。";
  }
  if (summary.label === "可继续") {
    return "本机服务可用，结果证据和接入地址已整理好。";
  }
  return summary.text || "本机服务已就绪。";
}

function replyModeLabel(readiness) {
  const inference = readiness && readiness.inference ? readiness.inference : {};
  return inference.real_local_ready ? "本地模型" : "演示回复";
}

function classify() {
  if (state.errors.length) {
    return { label: "有问题", tone: "bad", summary: "服务接口暂时不可用。", actionReady: false };
  }
  const readiness = state.readiness || {};
  if (readiness.summary) {
    const label = readiness.summary.label || "需确认";
    const tone = label === "可继续" ? "ok" : label === "有问题" ? "bad" : "warn";
    return {
      label,
      tone,
      summary: summaryTextFor(readiness),
      actionReady: true,
    };
  }
  if (!state.adapters || !state.adapters.current) {
    return { label: "需确认", tone: "warn", summary: "本机服务已就绪，当前还没有已使用的模型版本。", actionReady: true };
  }
  const pending = Array.isArray(state.adapters.pending) ? state.adapters.pending.length : 0;
  if (pending > 0) {
    return { label: "需确认", tone: "warn", summary: "本机服务可用，有新版本等待确认。", actionReady: true };
  }
  return { label: "可继续", tone: "ok", summary: "本机服务可用，API 和网页地址已就绪。", actionReady: true };
}

function renderRuntime() {
  const runtime = state.runtime || {};
  const service = runtime.status === "passed" ? "运行中" : "检查中";
  const access = runtime.access_scope || "仅本机";
  text("workspaceLabel", runtime.workspace ? runtime.workspace + " / " + access : "本机工作区");
  text("webUrlValue", runtime.web_url || "-");
  text("apiUrlValue", runtime.api_url || "-");
  text("heroApiValue", runtime.api_url ? "已就绪" : "检查中");
  text("apiModelValue", apiContract().model);
  text("apiExampleValue", buildApiExample());
  text("serviceValue", service);
  text("serviceMeta", [runtime.provider, access, runtime.auth_mode].filter(Boolean).join(" / "));
  pill("runtimeStatus", runtime.status === "passed" ? "可继续" : "检查中", runtime.status === "passed" ? "ok" : "");
  $("openWebButton").disabled = !runtime.web_url;
  $("copyApiTopButton").disabled = !runtime.api_url;
}

function renderHandoff() {
  const handoff = currentHandoff();
  const urls = handoff.urls || {};
  const model = handoff.model || {};
  const version = handoff.version || {};
  const runtime = handoff.runtime || {};
  const access = runtime.access_scope || (state.runtime && state.runtime.access_scope) || "仅本机";
  const versionLabel = version.current || version.latest || "暂无版本";
  const summary = urls.api
    ? "聊天和反馈 API 已就绪。"
    : "正在检查网页、API 和模型版本。";
  text("handoffValue", summary);
  text("handoffWebValue", urls.web || "-");
  text("handoffApiValue", urls.api || "-");
  text("handoffFeedbackValue", urls.feedback || "-");
  text("handoffModelValue", [model.api_parameter || "base", model.label || model.selected].filter(Boolean).join(" / "));
  text("handoffVersionValue", [versionLabel, access].filter(Boolean).join(" / "));
  const testResult = state.handoffTest || null;
  const testChat = testResult && testResult.chat ? testResult.chat : {};
  const testFeedback = testResult && testResult.feedback ? testResult.feedback : {};
  const testText = testResult
    ? (testResult.ok
      ? "已测试：聊天和反馈闭环通过 / " + [testChat.request_id, testFeedback.signal_type].filter(Boolean).join(" / ")
      : "测试未通过：" + (testResult.summary || "请检查本机服务"))
    : "未测试";
  text("handoffTestValue", testText);
  $("testHandoffButton").disabled = !urls.api || !urls.feedback;
  $("copyHandoffButton").disabled = !urls.api && !urls.web;
}

function renderWorkspaces() {
  const workspaces = state.workspaces || {};
  const select = $("workspaceSelect");
  select.textContent = "";
  const items = Array.isArray(workspaces.items) && workspaces.items.length
    ? workspaces.items
    : [{ id: workspaces.current || (state.runtime && state.runtime.workspace) || "user_default", label: workspaces.current || "user_default", current: true, switchable: true }];
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label || item.id;
    option.selected = Boolean(item.current || item.id === workspaces.current);
    option.disabled = item.switchable === false;
    select.appendChild(option);
  }
  select.disabled = !state.workspaces;
  $("saveWorkspaceButton").disabled = !state.workspaces;
}

function renderRealLocalToggle() {
  const button = $("realLocalToggleButton");
  const inference = state.readiness && state.readiness.inference ? state.readiness.inference : null;
  const enabled = Boolean(inference && inference.real_local_enabled);
  button.textContent = enabled ? "暂停回复" : "本地回复";
  button.disabled = !inference;
}

function renderModels() {
  const models = state.models || {};
  const select = $("modelSelect");
  select.textContent = "";
  const candidates = Array.isArray(models.candidates) && models.candidates.length
    ? models.candidates
    : [{ id: models.selected || "local-default", label: models.selected_label || models.selected || "local-default", selected: true }];
  for (const candidate of candidates) {
    const option = document.createElement("option");
    option.value = candidate.id;
    option.textContent = candidate.label || candidate.id;
    option.selected = Boolean(candidate.selected || candidate.id === models.selected);
    select.appendChild(option);
  }
  select.disabled = candidates.length <= 1;
  const selected = candidates.find((candidate) => candidate.id === select.value) || candidates[0] || {};
  const pathInput = $("modelPathInput");
  if (pathInput && document.activeElement !== pathInput) {
    pathInput.value = selected.local_path || "";
  }
  const selectedLabel = selected.label || models.selected_label || models.selected || "-";
  text("modelValue", selectedLabel);
  text("heroModelValue", selectedLabel);
  const modelSource = state.readiness && state.readiness.model ? state.readiness.model.source : null;
  const configuration = state.readiness && state.readiness.configuration ? state.readiness.configuration : null;
  const sourceLabel = modelSource && modelSource.state && modelSource.state !== "ready"
    ? humanIssue(modelSource.state)
    : (modelSource && modelSource.label ? modelSource.label : (selected.source ? selected.source : "config"));
  const effectiveLabel = configuration && configuration.reload_required
    ? "需要重启"
    : (configuration && configuration.effective_scope === "next_chat_request" ? "下一次请求生效" : "");
  text("modelMeta", [sourceLabel, effectiveLabel].filter(Boolean).join(" / "));
  const modelReady = modelSource ? Boolean(modelSource.ok) : Boolean(models.selected);
  const needsReload = Boolean(configuration && configuration.reload_required);
  text("modelPathState", modelReady && !needsReload ? "已保存" : humanIssue(modelSource && modelSource.state ? modelSource.state : "需确认"));
  pill("modelStatus", modelReady && !needsReload ? "可继续" : "需确认", modelReady && !needsReload ? "ok" : "warn");
}

async function saveModel(modelId) {
  if (!modelId || !state.models) return;
  if (modelId === state.models.selected) {
    toast("已是当前模型");
    return;
  }
  const select = $("modelSelect");
  select.disabled = true;
  pill("modelStatus", "检查中", "");
  try {
    const payload = await sendJson("/pfe/config/model", { base_model: modelId });
    state.models = payload.models || await loadJson("/pfe/models");
    state.readiness = await loadJson("/pfe/readiness");
    state.handoff = null;
    state.handoffTest = null;
    state.trainingPreflight = null;
    toast(payload.reload_required ? "已保存，需重启服务" : "已保存，下一次请求生效");
  } catch (error) {
    toast("保存失败");
  }
  render();
}

async function toggleRealLocal() {
  const inference = state.readiness && state.readiness.inference ? state.readiness.inference : null;
  const enabled = !(inference && inference.real_local_enabled);
  const button = $("realLocalToggleButton");
  button.disabled = true;
  try {
    const payload = await sendJson("/pfe/config/real-local", { enabled });
    state.readiness = payload.readiness || await loadJson("/pfe/readiness");
    state.handoff = null;
    state.handoffTest = null;
    state.trainingPreflight = null;
    toast(enabled ? "已切换为本地模型回复" : "已切回演示回复");
  } catch (error) {
    toast("设置失败");
  }
  render();
}

async function saveWorkspace(workspaceName) {
  const name = String(workspaceName || "").trim();
  if (!name) {
    toast("请输入工作区名称");
    return;
  }
  if (state.workspaces && name === state.workspaces.current) {
    toast("已是当前工作区");
    return;
  }
  $("workspaceSelect").disabled = true;
  $("saveWorkspaceButton").disabled = true;
  try {
    const payload = await postJson("/pfe/workspaces", { name });
    state.workspaces = payload.workspaces || await loadJson("/pfe/workspaces");
    state.runtime = payload.runtime || await loadJson("/pfe/runtime");
    state.adapters = payload.adapters || await loadJson("/pfe/adapters");
    state.readiness = payload.readiness || await loadJson("/pfe/readiness");
    state.models = await loadJson("/pfe/models");
    state.trainingJobs = await loadJson("/pfe/training/jobs").catch(() => null);
    state.evalStatus = await loadJson("/pfe/eval/status").catch(() => null);
    state.status = await loadJson("/pfe/status?detail=full").catch(() => null);
    state.handoff = null;
    state.handoffTest = null;
    state.trainingPreflight = null;
    if ($("workspaceInput")) $("workspaceInput").value = "";
    toast(payload.created ? "已创建并切换" : "已切换工作区");
  } catch (error) {
    toast("工作区设置失败");
  }
  render();
}

async function saveModelPath() {
  const input = $("modelPathInput");
  const modelId = input ? input.value.trim() : "";
  if (!modelId) {
    toast("请输入模型文件夹");
    return;
  }
  await saveModel(modelId);
}

async function testHandoff() {
  const button = $("testHandoffButton");
  button.disabled = true;
  text("handoffTestValue", "测试中");
  try {
    state.handoffTest = await postJson("/pfe/handoff/test", {
      message: "hello from PFE Studio",
      action: "accept",
    });
    toast(state.handoffTest && state.handoffTest.ok ? "接入测试通过" : "接入测试未通过");
  } catch (error) {
    state.handoffTest = { ok: false, summary: "请求失败" };
    toast("接入测试失败");
  }
  render();
}

function versionDate(item) {
  return item && (item.created_at || item.updated_at || item.timestamp) ? (item.created_at || item.updated_at || item.timestamp) : "";
}

function adapterShort(value) {
  const textValue = String(value || "").trim();
  if (!textValue) return "无";
  const parts = textValue.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || textValue;
}

function promotionGateText(item) {
  const gate = item && item.promotion_gate ? item.promotion_gate : {};
  const reason = gate.reason || "";
  if (!reason || gate.allowed) return "";
  if (reason === "eval_required") return "上线闸门：等待评估通过";
  if (reason === "failed_eval") return "上线闸门：评估未通过";
  if (reason === "archived") return "上线闸门：历史版本可回退";
  return "上线闸门：" + reason.replace(/_/g, " ");
}

function renderAdapters() {
  const adapters = state.adapters || {};
  const current = adapters.current;
  text("versionValue", current && current.version ? current.version : "暂无版本");
  text("heroVersionValue", current && current.version ? current.version : "暂无版本");
  const pendingEval = adapters.pending_eval_adapter || null;
  text("adapterBaseModelValue", adapterShort(adapters.base_model || (current && current.base_model) || (pendingEval && pendingEval.base_model)));
  text("adapterLatestValue", current && current.version ? current.version : "无");
  text("adapterPendingValue", pendingEval && pendingEval.version ? pendingEval.version : "无");
  text("adapterLoadedValue", adapters.adapter_loaded ? "是" : "否");
  const list = $("versionList");
  list.textContent = "";
  const versions = Array.isArray(adapters.versions) ? adapters.versions : [];
  if (!versions.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "还没有模型版本";
    list.appendChild(empty);
    pill("adapterStatus", "需确认", "warn");
    return;
  }
  for (const item of versions.slice(0, 4)) {
    const row = document.createElement("article");
    row.className = "version-item";
    const top = document.createElement("div");
    top.className = "version-top";
    const name = document.createElement("div");
    name.className = "version-name";
    name.textContent = item.version || "-";
    const badge = document.createElement("span");
    const stateLabel = item.user_state || item.state || "待验证";
    badge.className = "status-pill " + (stateLabel === "使用中" ? "ok" : stateLabel === "有问题" ? "bad" : "warn");
    badge.textContent = stateLabel;
    top.append(name, badge);

    const meta = document.createElement("div");
    meta.className = "version-meta";
    const parts = [
      item.artifact_role || (item.artifact_format ? "本地版本" : null),
      item.artifact_format || null,
      item.requires_export_step ? "需导出" : null,
      versionDate(item),
    ].filter(Boolean);
    meta.textContent = parts.join(" / ") || "版本记录";
    row.append(top, meta);
    const evidence = document.createElement("div");
    evidence.className = "version-evidence";
    const evidenceLines = [
      item.training_summary && item.training_summary.summary_line,
      item.eval_summary && item.eval_summary.summary_line,
      item.decision && item.decision.summary_line,
      promotionGateText(item),
    ].filter(Boolean);
    for (const line of evidenceLines) {
      const detail = document.createElement("div");
      detail.className = "version-evidence-line";
      detail.textContent = line;
      evidence.appendChild(detail);
    }
    if (evidence.childNodes.length) {
      row.appendChild(evidence);
    }
    const actions = document.createElement("div");
    actions.className = "version-actions";
    const apis = item.action_api || {};
    if (item.can_promote && apis.promote) {
      actions.append(actionButton("设为当前", apis.promote, item.version));
    }
    if (item.can_rollback && apis.rollback) {
      actions.append(actionButton("回退", apis.rollback, item.version));
    }
    if (item.can_archive && apis.archive) {
      actions.append(actionButton("归档", apis.archive, item.version));
    }
    if ((item.can_eval || item.eval_running) && apis.eval) {
      actions.append(actionButton(item.eval_running ? "评估中" : "评估", apis.eval, item.version, "eval", Boolean(item.eval_running)));
    }
    if (actions.childNodes.length) {
      row.appendChild(actions);
    }
    list.appendChild(row);
  }
  pill("adapterStatus", current ? "可继续" : "需确认", current ? "ok" : "warn");
}

function actionButton(label, api, version, actionType, disabled) {
  const button = document.createElement("button");
  button.className = "button";
  button.type = "button";
  button.textContent = label;
  button.dataset.adapterAction = api;
  button.dataset.adapterVersion = version || "";
  button.dataset.adapterLabel = label;
  button.dataset.adapterActionType = actionType || "adapter";
  button.disabled = Boolean(disabled);
  return button;
}

function trainingJobLabel(status) {
  const value = String(status || "");
  if (value === "queued") return "等待中";
  if (value === "running") return "生成中";
  if (value === "completed") return "已完成";
  if (value === "failed") return "有问题";
  if (value === "cancelled") return "已取消";
  return value || "未开始";
}

function currentTrainingJob() {
  const jobs = state.trainingJobs || {};
  return jobs.active || jobs.latest || null;
}

function confirmAction(title, detail) {
  return window.confirm(detail ? title + "\n\n" + detail : title);
}

function adapterActionConfirmation(label, version) {
  if (label === "设为当前") {
    return {
      title: "设为当前版本？",
      detail: "版本 " + version + " 会成为 API 回复使用的模型版本。",
    };
  }
  if (label === "回退") {
    return {
      title: "回退到这个版本？",
      detail: "当前版本会切回 " + version + "。",
    };
  }
  if (label === "归档") {
    return {
      title: "归档这个版本？",
      detail: "归档后仍可在历史中查看，需要时可以回退。",
    };
  }
  if (label === "评估") {
    return {
      title: "评估这个版本？",
      detail: "系统会后台评估 " + version + "，完成后刷新版本建议。",
    };
  }
  return {
    title: label + "？",
    detail: version ? "版本 " + version : "",
  };
}

function isRetryableTrainingJob(job) {
  return Boolean(job && job.retry_url && (job.status === "failed" || job.status === "cancelled"));
}

function renderTrainingJob() {
  const job = currentTrainingJob();
  if (!job) {
    text("trainingJobMeta", "最近任务：-");
    return null;
  }
  const id = String(job.job_id || "").slice(0, 8);
  const version = job.adapter_version ? " / 版本 " + job.adapter_version : "";
  const event = job.latest_event && job.latest_event.message ? " / " + job.latest_event.message : "";
  const cancel = job.cancellation_requested ? " / 已请求停止" : "";
  const error = job.error ? " / " + job.error : "";
  text("trainingJobMeta", "最近任务：" + id + " / " + trainingJobLabel(job.status) + version + cancel + event + error);
  return job;
}

async function runAdapterAction(button) {
  const api = button.dataset.adapterAction;
  const version = button.dataset.adapterVersion;
  const label = button.dataset.adapterLabel || "操作";
  const actionType = button.dataset.adapterActionType || "adapter";
  if (!api || !version) return;
  const confirmation = adapterActionConfirmation(label, version);
  if (!confirmAction(confirmation.title, confirmation.detail)) return;
  button.disabled = true;
  try {
    if (actionType === "eval") {
      const payload = await postJson(api, { confirm: true, version });
      state.evalStatus = payload;
      state.adapters = payload.adapters || await loadJson("/pfe/adapters");
      state.handoff = null;
      state.handoffTest = null;
      toast("已开始评估");
      refreshEvalStatus();
    } else {
      const payload = await postJson(api, { confirm: true });
      state.adapters = payload.adapters || await loadJson("/pfe/adapters");
      state.readiness = await loadJson("/pfe/readiness");
      state.handoff = null;
      state.handoffTest = null;
      toast("已完成");
    }
  } catch (error) {
    const payload = error && error.payload ? error.payload : {};
    if (payload.adapters) state.adapters = payload.adapters;
    if (payload.code === "promotion_eval_required") {
      toast("先评估通过");
    } else {
      toast("操作失败");
    }
  }
  render();
}

async function refreshEvalStatus() {
  window.clearTimeout(refreshEvalStatus.timer);
  try {
    state.evalStatus = await loadJson("/pfe/eval/status");
    state.adapters = state.evalStatus.adapters || await loadJson("/pfe/adapters");
    state.handoff = null;
    state.handoffTest = null;
    render();
    if (state.evalStatus && state.evalStatus.state === "running") {
      refreshEvalStatus.timer = window.setTimeout(refreshEvalStatus, 1200);
    }
  } catch (error) {
    state.evalStatus = state.evalStatus || null;
  }
}

function renderTrainingPreflight() {
  const preflight = state.trainingPreflight;
  const startButton = $("startTrainingButton");
  const cancelButton = $("cancelTrainingButton");
  const retryButton = $("retryTrainingButton");
  cancelButton.disabled = true;
  retryButton.disabled = true;
  const job = renderTrainingJob();
  if (job && (job.status === "queued" || job.status === "running")) {
    text("trainingValue", trainingJobLabel(job.status));
    text("trainingMeta", job.cancellation_requested ? "已记录停止请求；运行中的训练会自行结束。" : "任务正在处理，完成后会出现在版本列表。");
    pill("trainingStatus", "等待中", "warn");
    startButton.disabled = true;
    cancelButton.disabled = Boolean(job.cancellation_requested);
    return;
  }
  if (isRetryableTrainingJob(job)) {
    retryButton.disabled = false;
  }
  if (!preflight) {
    text("trainingValue", job ? trainingJobLabel(job.status) : "未检查");
    text("trainingMeta", job && isRetryableTrainingJob(job) ? "可重新生成这个版本。" : "选择模型后可检查。");
    pill("trainingStatus", job && isRetryableTrainingJob(job) ? "可重试" : "待检查", job && isRetryableTrainingJob(job) ? "warn" : "");
    startButton.disabled = true;
    return;
  }
  const blocked = Array.isArray(preflight.blocked_by) ? preflight.blocked_by : [];
  const warnings = Array.isArray(preflight.warnings) ? preflight.warnings : [];
  const ready = Boolean(preflight.ready);
  text("trainingValue", ready ? "可生成版本" : "需处理条件");
  const details = humanIssueList(blocked.length ? blocked : warnings);
  text("trainingMeta", details.length ? details.join(" / ") : "确认后创建训练任务。");
  pill("trainingStatus", ready ? "可继续" : "需确认", ready ? "ok" : "warn");
  startButton.disabled = !ready;
}

async function refreshTrainingJobs() {
  window.clearTimeout(refreshTrainingJobs.timer);
  try {
    state.trainingJobs = await loadJson("/pfe/training/jobs");
    state.adapters = await loadJson("/pfe/adapters");
    state.handoff = null;
    state.handoffTest = null;
    render();
    const job = state.trainingJobs && state.trainingJobs.active;
    if (job && (job.status === "queued" || job.status === "running") && !job.cancellation_requested) {
      refreshTrainingJobs.timer = window.setTimeout(refreshTrainingJobs, 1200);
    }
  } catch (error) {
    state.trainingJobs = state.trainingJobs || null;
  }
}

async function requestTrainingPreflight() {
  $("trainingPreflightButton").disabled = true;
  pill("trainingStatus", "检查中", "");
  try {
    const payload = await postJson("/pfe/training/jobs", { method: "sft" });
    state.trainingPreflight = payload.preflight || payload;
    toast("训练条件已检查");
  } catch (error) {
    const payload = error && error.payload ? error.payload : {};
    if (error && error.status === 409 && payload.code === "confirmation_required" && payload.preflight) {
      state.trainingPreflight = payload.preflight;
      toast("训练条件已检查");
    } else {
      toast("检查失败");
    }
  }
  $("trainingPreflightButton").disabled = false;
  render();
}

async function cancelTraining() {
  const job = currentTrainingJob();
  if (!job || !job.cancel_url || (job.status !== "queued" && job.status !== "running")) {
    toast("没有可停止的任务");
    return;
  }
  if (!confirmAction("停止生成？", "会记录停止请求；正在运行的底层训练会自行结束。")) return;
  $("cancelTrainingButton").disabled = true;
  try {
    const payload = await postJson(job.cancel_url, { confirm: true });
    state.trainingJobs = payload.jobs || await loadJson("/pfe/training/jobs");
    state.handoff = null;
    state.handoffTest = null;
    toast(payload.message || "已请求停止");
    refreshTrainingJobs();
  } catch (error) {
    toast("停止失败");
  }
  render();
}

async function retryTraining() {
  const job = currentTrainingJob();
  if (!isRetryableTrainingJob(job)) {
    toast("没有可重新生成的任务");
    return;
  }
  if (!confirmAction("重新生成？", "会复用原训练配置，并重新走条件检查。")) return;
  $("retryTrainingButton").disabled = true;
  try {
    const payload = await postJson(job.retry_url, { confirm: true });
    state.trainingPreflight = payload.preflight || state.trainingPreflight;
    state.trainingJobs = payload.jobs || await loadJson("/pfe/training/jobs");
    state.adapters = await loadJson("/pfe/adapters");
    state.handoff = null;
    state.handoffTest = null;
    toast("已重新开始生成");
    refreshTrainingJobs();
  } catch (error) {
    const payload = error && error.payload ? error.payload : {};
    if (payload.preflight) state.trainingPreflight = payload.preflight;
    toast("重新生成失败");
  }
  render();
}

async function startTraining() {
  const preflight = state.trainingPreflight;
  if (!preflight || !preflight.ready) {
    await requestTrainingPreflight();
    if (!state.trainingPreflight || !state.trainingPreflight.ready) return;
  }
  if (!confirmAction("生成新版本？", "会创建一个后台任务，完成后出现在版本列表。")) return;
  $("startTrainingButton").disabled = true;
  try {
    const payload = await postJson("/pfe/training/jobs", { method: "sft", confirm: true });
    state.trainingPreflight = payload.preflight || state.trainingPreflight;
    state.trainingJobs = payload.jobs || await loadJson("/pfe/training/jobs");
    state.adapters = await loadJson("/pfe/adapters");
    state.handoff = null;
    state.handoffTest = null;
    toast("已开始生成版本");
    refreshTrainingJobs();
  } catch (error) {
    const payload = error && error.payload ? error.payload : {};
    if (payload.preflight) state.trainingPreflight = payload.preflight;
    toast("启动失败");
  }
  render();
}

function renderStatus() {
  const readiness = state.readiness || {};
  if (readiness.inference) {
    const configuration = readiness.configuration || {};
    const configMeta = configuration.effective_scope === "next_chat_request"
      ? "模型配置下一次请求生效"
      : "";
    text("requestValue", replyModeLabel(readiness));
    text("heroReplyValue", replyModeLabel(readiness));
    const blockers = humanIssueList(readinessBlockers(readiness));
    text("requestMeta", blockers.length
      ? blockers.join(" / ")
      : (configMeta || (readiness.summary && readiness.summary.text ? readiness.summary.text : "-")));
    return;
  }
  const status = state.status || {};
  const lifecycle = status.metadata && status.metadata.lifecycle ? status.metadata.lifecycle : {};
  const serve = lifecycle.serve || {};
  const last = serve.last_check || {};
  text("requestValue", last.path || "/pfe/status");
  text("heroReplyValue", last.path ? "已检查" : "检查中");
  text("requestMeta", last.status_code ? String(last.status_code) : serve.state || "-");
}

function renderSummary() {
  const result = classify();
  pill("overallStatus", result.label, result.tone);
  pill("workOrderStatus", result.label, result.tone);
  text("summaryText", result.summary);
  renderRealLocalToggle();
}

function render() {
  renderRuntime();
  renderWorkspaces();
  renderModels();
  renderAdapters();
  renderHandoff();
  renderTrainingPreflight();
  renderStatus();
  renderSummary();
}

async function refresh() {
  state.errors = [];
  pill("overallStatus", "检查中", "");
  pill("runtimeStatus", "检查中", "");
  pill("modelStatus", "检查中", "");
  pill("adapterStatus", "检查中", "");
  try {
    const [runtime, workspaces, models, adapters, handoff, readiness, trainingJobs, evalStatus, status] = await Promise.all([
      loadJson("/pfe/runtime"),
      loadJson("/pfe/workspaces"),
      loadJson("/pfe/models"),
      loadJson("/pfe/adapters"),
      loadJson("/pfe/handoff"),
      loadJson("/pfe/readiness"),
      loadJson("/pfe/training/jobs").catch(() => null),
      loadJson("/pfe/eval/status").catch(() => null),
      loadJson("/pfe/status?detail=full").catch(() => null),
    ]);
    state.runtime = runtime;
    state.workspaces = workspaces;
    state.models = models;
    state.adapters = adapters;
    state.handoff = handoff;
    state.readiness = readiness;
    state.trainingJobs = trainingJobs;
    state.evalStatus = evalStatus;
    state.status = status;
    state.handoffTest = null;
  } catch (error) {
    state.errors.push(error);
    state.runtime = state.runtime || {};
    state.workspaces = state.workspaces || {};
    state.models = state.models || {};
    state.handoff = state.handoff || null;
    state.handoffTest = state.handoffTest || null;
    state.adapters = state.adapters || {};
  }
  render();
  const active = state.trainingJobs && state.trainingJobs.active;
  if (active && (active.status === "queued" || active.status === "running") && !active.cancellation_requested) {
    window.clearTimeout(refreshTrainingJobs.timer);
    refreshTrainingJobs.timer = window.setTimeout(refreshTrainingJobs, 1200);
  }
  if (state.evalStatus && state.evalStatus.state === "running") {
    window.clearTimeout(refreshEvalStatus.timer);
    refreshEvalStatus.timer = window.setTimeout(refreshEvalStatus, 1200);
  }
}

async function copyText(value) {
  if (!value || value === "-") return;
  let copied = false;
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      await navigator.clipboard.writeText(value);
      copied = true;
    }
  } catch (error) {
    copied = false;
  }
  if (!copied) {
    const input = document.createElement("textarea");
    input.value = value;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.top = "-1000px";
    input.style.left = "-1000px";
    document.body.appendChild(input);
    input.select();
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }
    input.remove();
  }
  toast(copied ? "已复制" : "复制失败");
}

$("refreshButton").addEventListener("click", refresh);
$("openWebButton").addEventListener("click", () => {
  const url = state.runtime && state.runtime.web_url;
  if (url) window.open(url, "_blank", "noopener,noreferrer");
});
$("copyApiTopButton").addEventListener("click", () => {
  copyText(state.runtime && state.runtime.api_url);
});
$("copyHandoffButton").addEventListener("click", () => {
  copyText(buildHandoffText());
});
$("testHandoffButton").addEventListener("click", testHandoff);
$("modelSelect").addEventListener("change", (event) => saveModel(event.target.value));
$("workspaceSelect").addEventListener("change", (event) => saveWorkspace(event.target.value));
$("saveWorkspaceButton").addEventListener("click", () => saveWorkspace($("workspaceInput").value));
$("saveModelPathButton").addEventListener("click", saveModelPath);
$("realLocalToggleButton").addEventListener("click", toggleRealLocal);
$("trainingPreflightButton").addEventListener("click", requestTrainingPreflight);
$("startTrainingButton").addEventListener("click", startTraining);
$("cancelTrainingButton").addEventListener("click", cancelTraining);
$("retryTrainingButton").addEventListener("click", retryTraining);
$("workspaceInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") saveWorkspace(event.target.value);
});
$("modelPathInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") saveModelPath();
});
document.addEventListener("click", (event) => {
  const action = event.target.closest("[data-adapter-action]");
  if (action) {
    runAdapterAction(action);
    return;
  }
  const button = event.target.closest("[data-copy-target]");
  if (!button) return;
  const target = $(button.getAttribute("data-copy-target"));
  copyText(target && target.textContent);
});

refresh();
