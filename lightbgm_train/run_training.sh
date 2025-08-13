#!/bin/sh

# ==============================================================================
# ## OpenClash 智能权重模型自动化训练脚本 (集成重启功能)
#
# **功能**:
#   1. 自动从 OpenClash 目录拷贝最新的训练数据。
#   2. 使用 Docker (CPU模式) 启动一个包含完整环境的训练任务。
#   3. 将训练好的新模型自动部署回 OpenClash 目录。
#   4. 【新增】自动重启 OpenClash 服务，使新模型立即生效。
#   5. 记录完整的训练和部署日志。
# ==============================================================================

# ---
# **配置部分**
# ---
WORKDIR="/mnt/sdb1/workdir/lightbgm_train" # 脚本和 train.py 所在的工作目录
OPENCLASH_DATA_DIR="/etc/openclash"        # OpenClash 的数据和配置目录
LOG_FILE="${WORKDIR}/training_log.txt"     # 训练日志文件的存放位置
DOCKER_IMAGE="zivee/clash-lighbgm-trainer:latest" # 您使用的 Docker 镜像

# ---
# **脚本主体**
# ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "====== 开始新一轮的模型训练任务 (CPU 模式) ======"

# **步骤 1: 准备训练数据**
log "步骤 1: 正在拷贝最新的训练数据..."
SOURCE_DATA_FILE="${OPENCLASH_DATA_DIR}/smart_weight_data.csv"
DEST_DATA_FILE="${WORKDIR}/smart_weight_data.csv"
if [ ! -f "$SOURCE_DATA_FILE" ]; then
    log "[错误] 源数据文件不存在: ${SOURCE_DATA_FILE}"
    exit 1
fi
cp -f "$SOURCE_DATA_FILE" "$DEST_DATA_FILE" || { log "[错误] 数据拷贝失败！"; exit 1; }
log "数据拷贝成功。"

# **步骤 2: 启动 Docker 容器进行训练**
log "步骤 2: 启动 Docker 容器开始训练 (CPU)..."
docker run --rm \
  -v "${WORKDIR}":/app \
  -w /app \
  "${DOCKER_IMAGE}" \
  python train.py >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    log "[错误] Docker 训练过程失败！详情请查看日志: ${LOG_FILE}"
    exit 1
fi
log "Docker 训练过程成功完成。"

# **步骤 3: 部署新模型**
log "步骤 3: 正在部署新训练的模型..."
TRAINED_MODEL_PATH="${WORKDIR}/models/Model.bin"
DEST_MODEL_PATH="${OPENCLASH_DATA_DIR}/Model.bin"
if [ ! -f "$TRAINED_MODEL_PATH" ]; then
    log "[错误] 未找到训练好的模型文件: ${TRAINED_MODEL_PATH}"
    exit 1
fi
cp -f "$TRAINED_MODEL_PATH" "$DEST_MODEL_PATH" || { log "[错误] 模型部署失败！"; exit 1; }
log "新模型已成功部署到: ${DEST_MODEL_PATH}"

# **【新增】步骤 4: 重启 OpenClash 服务**
log "步骤 4: 正在重启 OpenClash 服务以加载新模型..."
# 检查 OpenClash 服务脚本是否存在
if [ ! -f "/etc/init.d/openclash" ]; then
    log "[警告] 未找到 OpenClash 服务脚本 /etc/init.d/openclash，跳过重启。"
    log "您可能需要手动重启 OpenClash 来应用新模型。"
else
    /etc/init.d/openclash restart >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        log "OpenClash 服务重启成功。"
    else
        log "[错误] OpenClash 服务重启失败！请检查 OpenClash 运行状态。"
    fi
fi

log "====== 所有步骤已成功完成！======"
echo "" >> "$LOG_FILE"
exit 0