# 干净首发仓库流程

更新时间：2026-04-21

## 适用场景

这个流程适用于当前仓库这种情况：

- 还没有正式首发 commit
- 本地 `.git` 元数据已经变重或不可信
- 目录里存在大量本地产物，但源码本体其实很轻
- 需要为 GitHub 准备一个干净、可公开发布的首发仓库

当前仓库正符合这个条件，所以推荐使用“备份旧 `.git`，重新初始化新仓库”的方式，而不是继续修补旧元数据。

## 这套流程做什么

推荐脚本：[`tools/init_clean_git_repo.sh`](../tools/init_clean_git_repo.sh)

默认行为是 **dry-run**，只打印计划，不会修改任何 Git 元数据。

执行模式会：

1. 把当前 `.git` 目录整体备份为 `.git-backup-YYYYMMDD-HHMMSS/`
2. 初始化一个新的 `main` 分支仓库
3. 只加入首发所需的源码、测试、文档和小型工具脚本
4. 依赖 `.gitignore` 自动排除模型权重、训练产物、虚拟环境、缓存和 `tools/llama.cpp/`

## 推荐包含的首发内容

- `README.md`
- `.gitignore`
- `pyproject.toml`
- `Makefile`
- `ENGINE_DEV_DOC.md`
- `PROFILE_SYSTEM_README.md`
- `docs/`
- `examples/`
- `pfe-cli/`
- `pfe-core/`
- `pfe-server/`
- `tests/`
- `tools/`

默认不包含以下偏内部文档：

- `AGENT.md`
- `CLAUDE.md`
- `CHAT_COLLECTOR_INTEGRATION_SUMMARY.md`

如果你确认这些内容也适合公开，可以在执行时加 `--with-internal-docs`。

## 实际用法

先看计划：

```bash
tools/init_clean_git_repo.sh
```

确认无误后执行：

```bash
tools/init_clean_git_repo.sh --execute
```

如果你想连内部协作文档一起加入首发：

```bash
tools/init_clean_git_repo.sh --execute --with-internal-docs
```

执行完成后，建议立刻检查：

```bash
git status --short
```

然后再进行首次提交：

```bash
git commit -m "Initial open-source release"
git remote add origin <your-github-url>
git push -u origin main
```

## 为什么不直接沿用旧 `.git`

原因很简单：

- 当前仓库没有正式 commit，但本地 `.git` 已经非常重
- 旧元数据里可能混有大对象包、缓存历史或无效锁文件
- 这些内容和公开源码仓库无关，却会显著增加首发出错概率

“备份旧 `.git` + 新建干净仓库”更适合首发，也更容易在 GitHub 上得到一个轻量、可维护的历史起点。

## 备份目录说明

脚本不会直接删除旧元数据，而是把它挪到：

```text
.git-backup-YYYYMMDD-HHMMSS/
```

这个目录已被 `.gitignore` 忽略，不会进入新仓库。

如果确认新仓库状态完全没问题，后续你可以再手动删除旧备份。
