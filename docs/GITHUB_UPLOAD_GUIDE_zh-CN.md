# GitHub 上传指南

> 本项目包不会自动连接或操作任何 GitHub 账号。以下命令仅供你在自己的 `xinyushu766-art` 账号下手动执行。

建议仓库名称：`AutoGraphFM`

建议仓库描述：

```text
Auditable graph-temporal forecasting system for automotive demand with leakage-safe backtesting, graph falsification, drift monitoring, and release governance.
```

建议 Topics：

```text
machine-learning
time-series
forecasting
graph-neural-networks
pytorch
mlops
automotive
data-engineering
```

## 方法一：网页上传

1. 在 GitHub 点击 **New repository**。
2. Repository name 填 `AutoGraphFM`。
3. 选择 Public。
4. 不要勾选自动生成 README、LICENSE 或 `.gitignore`，这些文件已经包含在项目中。
5. 创建仓库后，选择 **uploading an existing file**，上传解压后的全部文件。

网页上传不适合大量文件时，使用下面的 Git 命令。

## 方法二：Git 命令

在解压后的项目目录打开终端：

```bash
git init
git branch -M main
git add .
git commit -m "Initial public release of AutoGraphFM v0.18.0"
git remote add origin https://github.com/xinyushu766-art/AutoGraphFM.git
git push -u origin main
```

## 上传后设置

1. 打开仓库的 **About** 设置，填写上面的描述和 Topics。
2. 在 **Actions** 页面确认 CI workflow 正常运行。
3. 在 **Releases** 创建 `v0.18.0`，标题可写 `Portfolio-ready public release`，正文可直接使用 `docs/GITHUB_RELEASE_NOTES_v0.18.0.md`。
4. 将 `README.md` 中的 CI badge 检查为绿色。
5. 不要上传 `.env`、`outputs/`、数据库文件、模型 checkpoint 或外部原始数据。
