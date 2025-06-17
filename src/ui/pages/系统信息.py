import streamlit as st
from src.ui.api_client import (
    api_request,
    get_system_status_summary,
    get_worker_summary,
    get_gpu_status
)
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("📊 系统信息")
st.markdown("查看系统健康状况和工作节点状态")

# === OVERALL SYSTEM STATUS ===
status_summary = get_system_status_summary()

if status_summary["color"] == "success":
    st.success(status_summary["message"])
elif status_summary["color"] == "warning":
    st.warning(status_summary["message"])
else:
    st.error(status_summary["message"])

st.markdown("---")

# === WORKER STATUS ===
st.subheader("👷 工作节点状态")

worker_summary = get_worker_summary()

# Overall worker metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("总工作节点", worker_summary["total_workers"])
with col2:
    st.metric("健康节点", worker_summary["healthy_workers"])

# Individual worker types
st.write("**各类型工作节点:**")

worker_type_info = {
    "gpu-inference": {"name": "🧠 AI推理", "desc": "处理智能问答和文档重排序"},
    "gpu-embedding": {"name": "🔢 向量嵌入", "desc": "文档向量化和相似度搜索"},
    "gpu-whisper": {"name": "🎵 语音转录", "desc": "视频音频转文字处理"},
    "cpu": {"name": "💻 基础处理", "desc": "PDF解析和文本预处理"}
}

worker_counts = worker_summary["worker_counts"]

cols = st.columns(2)
for i, (worker_type, type_info) in enumerate(worker_type_info.items()):
    col = cols[i % 2]

    with col:
        count = worker_counts.get(worker_type, 0)
        status_icon = "🟢" if count > 0 else "🔴"
        status_text = f"运行中 ({count})" if count > 0 else "不可用"

        st.markdown(f"**{type_info['name']} {status_icon}**")
        st.caption(f"状态: {status_text}")
        st.caption(type_info['desc'])
        st.markdown("")

st.markdown("---")

# === GPU RESOURCE STATUS ===
st.subheader("🖥️ GPU资源状态")

gpu_status = get_gpu_status()

if gpu_status["available"]:
    # Overall GPU metrics
    total_memory = gpu_status["total_memory_gb"]
    used_memory = gpu_status["used_memory_gb"]
    usage_percent = gpu_status["usage_percent"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("总显存", f"{total_memory:.1f} GB")
    with col2:
        st.metric("已使用", f"{used_memory:.1f} GB ({usage_percent:.1f}%)")

    # Usage status indicator
    if usage_percent < 60:
        st.success(f"GPU负载正常 ({usage_percent:.1f}%)")
    elif usage_percent < 85:
        st.warning(f"GPU负载较高 ({usage_percent:.1f}%)")
    else:
        st.error(f"GPU负载很高 ({usage_percent:.1f}%)")

    st.progress(usage_percent / 100.0)

    # Individual GPU details
    gpu_details = gpu_status["gpu_details"]
    if len(gpu_details) > 1:
        with st.expander("各GPU详情"):
            for gpu in gpu_details:
                usage = gpu["usage_percent"]
                status_emoji = "🟢" if usage < 60 else "🟡" if usage < 85 else "🔴"
                st.write(f"{status_emoji} **{gpu['name']}**: {usage:.1f}% 使用率 "
                        f"({gpu['used_gb']:.1f}GB / {gpu['total_gb']:.1f}GB)")
else:
    st.info("🖥️ 当前使用CPU处理模式")

st.markdown("---")

# === QUEUE STATUS ===
st.subheader("📋 任务队列状态")

try:
    queue_data = api_request("/job-chains", method="GET", silent=True)
    if queue_data:
        queue_status = queue_data.get("queue_status", {})

        if queue_status:
            queue_names = {
                "inference_tasks": "🧠 AI推理队列",
                "embedding_tasks": "🔢 嵌入队列",
                "transcription_tasks": "🎵 转录队列",
                "cpu_tasks": "💻 CPU队列"
            }

            cols = st.columns(2)
            for i, (queue_key, friendly_name) in enumerate(queue_names.items()):
                col = cols[i % 2]

                if queue_key in queue_status:
                    queue_info = queue_status[queue_key]
                    if isinstance(queue_info, dict):
                        status = queue_info.get("status", "unknown")
                        waiting = queue_info.get("waiting_tasks", 0)

                        with col:
                            if status == "busy":
                                st.markdown(f"**{friendly_name}** 🔄")
                                st.caption(f"处理中，等待: {waiting}")
                            else:
                                st.markdown(f"**{friendly_name}** ✅")
                                st.caption(f"空闲，等待: {waiting}")
                else:
                    with col:
                        st.markdown(f"**{friendly_name}** ❓")
                        st.caption("状态未知")
        else:
            st.info("队列信息不可用")
    else:
        st.info("无法获取队列状态")
except:
    st.info("队列状态获取失败")

st.markdown("---")

# === SYSTEM ACTIONS ===
st.subheader("🚀 快速操作")

action_cols = st.columns(3)

with action_cols[0]:
    if st.button("🧠 智能查询", use_container_width=True, type="primary"):
        st.switch_page("pages/智能查询.py")

with action_cols[1]:
    if st.button("📤 上传资料", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with action_cols[2]:
    if st.button("📋 查看任务", use_container_width=True):
        st.switch_page("pages/后台任务.py")

# === ADMIN SECTION ===
with st.expander("🔧 管理员工具"):
    st.warning("⚠️ 仅供系统管理员使用")

    admin_cols = st.columns(3)

    with admin_cols[0]:
        if st.button("刷新状态"):
            st.rerun()

    with admin_cols[1]:
        if st.button("清理GPU缓存"):
            try:
                result = api_request("/system/clear-gpu-cache", method="POST",
                                   data={"gpu_id": "gpu_0"})
                if result:
                    st.success("GPU缓存已清理")
                else:
                    st.error("清理失败")
            except:
                st.error("清理操作失败")

    with admin_cols[2]:
        if st.button("重启Workers"):
            try:
                result = api_request("/system/restart-workers", method="POST")
                if result:
                    st.success("重启信号已发送")
                else:
                    st.error("重启失败")
            except:
                st.error("重启操作失败")

    # Show detailed technical data
    if st.checkbox("显示详细技术信息"):
        st.subheader("详细系统数据")

        try:
            detailed_health = api_request("/system/health/detailed", silent=True)
            if detailed_health:
                st.json(detailed_health)
            else:
                st.error("无法获取详细信息")
        except:
            st.error("获取详细信息失败")

st.markdown("---")
st.caption("系统信息 - 监控整体系统健康状况")
st.caption("如需查看具体任务进度，请访问\"后台任务\"页面")