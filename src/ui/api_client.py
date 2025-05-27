"""
Simplified API client - src/ui/api_client.py
"""

import streamlit as st
import httpx
import time
from typing import Dict, Optional, Any

def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: float = 10.0,
    silent: bool = False,
) -> Optional[Dict]:
    """Simple API request function"""

    if "api_url" not in st.session_state or "api_key" not in st.session_state:
        if not silent:
            st.error("系统配置错误")
        return None

    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"

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

            if response.status_code >= 400:
                if not silent:
                    if response.status_code == 503:
                        st.error("系统暂时不可用，请稍后再试")
                    else:
                        st.error("请求失败，请重试")
                return None

            return response.json()

    except httpx.TimeoutException:
        if not silent:
            st.error("请求超时，请重试")
        return None
    except httpx.ConnectError:
        if not silent:
            st.error("无法连接到服务器")
        return None
    except Exception as e:
        if not silent:
            st.error("网络错误，请检查连接")
        return None

    return None