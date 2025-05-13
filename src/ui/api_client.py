"""
统一的 API 客户端，用于 Streamlit UI 组件。

此模块提供了一个一致的接口，用于发送 API 请求，
具有强大的错误处理、重试和超时管理功能。
"""

import streamlit as st
import httpx
import time
from typing import Dict, List, Any, Optional, Union, Callable


def api_request(
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
        retries: int = 1,
        silent: bool = False,
        handle_error: Optional[Callable] = None,
) -> Optional[Dict]:
    """
    发送 API 请求，具有增强的错误处理和重试逻辑。

    参数:
        endpoint: API 端点路径
        method: HTTP 方法 (GET, POST, DELETE)
        data: 请求体数据
        files: 要上传的文件
        params: 查询参数
        timeout: 请求超时时间（秒）
        retries: 失败时重试次数
        silent: 是否抑制错误消息
        handle_error: 自定义错误处理函数

    返回:
        响应数据，失败时返回 None
    """
    if "api_url" not in st.session_state or "api_key" not in st.session_state:
        if not silent:
            st.error("API 配置未初始化")
        return None

    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"

    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                if method == "GET":
                    response = client.get(url, headers=headers, params=params)
                elif method == "POST":
                    if files:
                        response = client.post(url, headers=headers, data=data, files=files)
                    else:
                        response = client.post(url, headers=headers, json=data)
                elif method == "DELETE":
                    response = client.delete(url, headers=headers)
                else:
                    if not silent:
                        st.error(f"不支持的请求方法: {method}")
                    return None

                # 处理 HTTP 错误 (4xx, 5xx)
                if response.status_code >= 400:
                    error_msg = f"API 错误 ({response.status_code}): {response.text}"

                    # 检查是否应该重试
                    should_retry = attempt < retries and (
                        # 重试服务器错误 (5xx)
                            response.status_code >= 500 or
                            # 重试速率限制错误
                            response.status_code == 429
                    )

                    if should_retry:
                        # 指数退避
                        wait_time = (2 ** attempt) * 0.5
                        time.sleep(wait_time)
                        continue

                    # 处理错误
                    if handle_error:
                        return handle_error(error_msg)
                    elif not silent:
                        st.error(error_msg)
                    return None

                # 成功
                return response.json()

        except httpx.TimeoutException:
            error_msg = f"请求超时 ({timeout} 秒)"

            # 检查是否应该重试
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # 处理错误
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

        except httpx.ConnectError:
            error_msg = "无法连接到 API 服务器"

            # 检查是否应该重试
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # 处理错误
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

        except Exception as e:
            error_msg = f"连接错误: {str(e)}"

            # 检查是否应该为某些错误重试
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # 处理错误
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

    # 如果执行到这里，所有重试都失败了
    return None


def check_worker_availability(worker_type: str) -> bool:
    """
    检查特定类型的 worker 是否可用。

    参数:
        worker_type: 要检查的 worker 类型 (gpu-inference, gpu-embedding 等)

    返回:
        如果 worker 可用则为 True，否则为 False
    """
    # 尝试获取详细的健康信息
    health_info = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True,
        timeout=2.0
    )

    if not health_info:
        return False

    # 检查 workers
    workers = health_info.get("workers", {})
    for worker_id, info in workers.items():
        if worker_type in worker_id and info.get("status") == "healthy":
            return True

    return False