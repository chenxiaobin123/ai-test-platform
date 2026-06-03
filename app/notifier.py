import json
import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def send_dingtalk_webhook(webhook_url: str, title: str, text: str) -> bool:
    """发送钉钉机器人消息（Markdown格式）"""
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": title, "text": f"## {title}\n\n{text}"},
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("✅ 钉钉通知发送成功")
            return True
        logger.error(f"❌ 钉钉通知失败: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"❌ 钉钉通知异常: {e}")
        return False


def send_feishu_webhook(webhook_url: str, title: str, text: str) -> bool:
    """发送飞书机器人消息（富文本格式）"""
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}},
            "elements": [{"tag": "markdown", "content": text}],
        },
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("✅ 飞书通知发送成功")
            return True
        logger.error(f"❌ 飞书通知失败: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"❌ 飞书通知异常: {e}")
        return False


def send_wecom_webhook(webhook_url: str, title: str, text: str) -> bool:
    """发送企业微信机器人消息（Markdown格式）"""
    payload = {
        "msgtype": "markdown",
        "markdown": {"content": f"## {title}\n\n{text}"},
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("✅ 企业微信通知发送成功")
            return True
        logger.error(f"❌ 企业微信通知失败: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"❌ 企业微信通知异常: {e}")
        return False


def send_test_result_notification(
    case_name: str,
    status: str,
    result_msg: str,
    report_path: Optional[str] = None,
):
    """根据环境变量自动选择平台发送测试结果通知"""
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji = "✅" if status == "success" else "❌"
    title = f"{emoji} AI自动化测试 - {case_name}"

    # 消息截断
    short_msg = result_msg[:800] + ("..." if len(result_msg) > 800 else "")
    text = (
        f"**用例名称**：{case_name}\n"
        f"**执行结果**：{status}\n"
        f"**完成时间**：{now_str}\n\n"
        f"**详细结果**：\n{short_msg}"
    )
    if report_path:
        text += f"\n\n📊 [查看报告]({report_path})"

    webhooks = {
        "dingtalk": os.getenv("DINGTALK_WEBHOOK"),
        "feishu": os.getenv("FEISHU_WEBHOOK"),
        "wecom": os.getenv("WECOM_WEBHOOK"),
    }

    sent_any = False
    if webhooks["dingtalk"]:
        sent_any |= send_dingtalk_webhook(webhooks["dingtalk"], title, text)
    if webhooks["feishu"]:
        sent_any |= send_feishu_webhook(webhooks["feishu"], title, text)
    if webhooks["wecom"]:
        sent_any |= send_wecom_webhook(webhooks["wecom"], title, text)

    if not sent_any:
        logger.info("⚠️ 未配置任何通知 Webhook，跳过通知")