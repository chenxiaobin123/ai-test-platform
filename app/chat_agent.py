import json
import logging
import os
import re
import threading
import uuid
from queue import Queue
from typing import Dict, List, Optional, Generator

from openai import OpenAI
from dotenv import load_dotenv

from . import crud
from .test_runner import run_web_test

load_dotenv()
logger = logging.getLogger(__name__)

chat_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=120.0,
    max_retries=1
)

AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

# 活跃的测试运行队列，key为run_id，value为Queue用于SSE推送
_active_runs: Dict[str, Queue] = {}

CHAT_SYSTEM_PROMPT = """你是一个AI自动化测试助手，帮助用户管理和执行自动化测试用例。

你的能力：
1. 列出所有可用的测试用例
2. 根据名称或ID运行指定的测试用例
3. 查看测试任务的执行状态和结果
4. 回答关于测试平台的问题

规则：
- 当用户说"运行"、"执行"、"跑"某个用例时，你需要调用 run_test_case 函数
- 当用户问"有哪些用例"、"列出用例"时，你需要调用 list_test_cases 函数
- 当用户问"查看结果"、"状态"时，你需要调用 check_task_status 函数
- 请用中文回复，简洁专业
- 如果用户没有指定具体用例，先列出用例让用户选择
- 运行用例后会返回一个run_id，用户可以用它查看实时进度
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_test_cases",
            "description": "列出所有可用的自动化测试用例",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_test_case",
            "description": "运行指定的测试用例。用户可以通过用例名称或ID来指定。如果存在多个匹配的用例，会列出让用户选择。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用例名称关键词或用例ID，如'登录'、'TC-001'、'1'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_all_test_cases",
            "description": "运行所有可用的自动化测试用例",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_run_status",
            "description": "查看某个测试运行的实时状态和结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "运行ID，从run_test_case的返回结果中获取"
                    }
                },
                "required": ["run_id"]
            }
        }
    }
]


def _get_db():
    from .database import SessionLocal
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def _find_test_cases(query: str, db) -> list:
    """根据关键词查找测试用例"""
    try:
        cases = crud.get_test_cases(db, skip=0, limit=200)
    except Exception:
        cases = crud.get_test_cases(db, skip=0, limit=200)

    query_lower = query.strip().lower()

    if query_lower in ("全部", "所有", "all"):
        return cases

    if query_lower.isdigit():
        case_id = int(query_lower)
        for c in cases:
            if c.id == case_id:
                return [c]

    matched = []
    for c in cases:
        name_lower = (c.name or "").lower()
        desc_lower = (c.description or "").lower()
        if query_lower in name_lower or query_lower in desc_lower:
            matched.append(c)

    return matched


def _run_test_in_thread(case_id: int, case_name: str, url: str,
                        steps, run_id: str, status_queue: Queue):
    """在后台线程中运行测试，通过queue推送状态"""
    from .database import SessionLocal
    db = SessionLocal()
    try:
        status_queue.put({"type": "status", "run_id": run_id,
                          "message": f"🚀 开始运行测试: {case_name}",
                          "case_id": case_id})

        status_queue.put({"type": "status", "run_id": run_id,
                          "message": f"🌐 打开页面: {url}"})

        final_status, result_msg = run_web_test(
            case_id=case_id,
            url=url,
            steps=steps,
            is_login_case=False,
            db=db
        )

        status_queue.put({"type": "status", "run_id": run_id,
                          "message": f"\n{result_msg}"})

        if final_status == "success":
            status_queue.put({"type": "complete", "run_id": run_id,
                              "status": "success",
                              "message": f"✅ 测试通过: {case_name}"})
        else:
            status_queue.put({"type": "complete", "run_id": run_id,
                              "status": "failed",
                              "message": f"❌ 测试失败: {case_name}"})
    except Exception as e:
        logger.error(f"测试执行异常: {e}")
        status_queue.put({"type": "complete", "run_id": run_id,
                          "status": "error",
                          "message": f"❌ 执行异常: {str(e)}"})
    finally:
        try:
            db.close()
        except Exception:
            pass


def execute_tool_call(tool_name: str, arguments: dict):
    """执行工具调用，返回结果"""
    db = _get_db()
    try:
        if tool_name == "list_test_cases":
            cases = crud.get_test_cases(db, skip=0, limit=200)
            if not cases:
                return {"result": "📭 当前没有自动化测试用例。请先在「自动化用例」标签页创建用例。"}
            lines = ["📋 当前可用的测试用例："]
            for c in cases:
                lines.append(f"  - [ID:{c.id}] {c.name}")
            return {"result": "\n".join(lines)}

        elif tool_name == "run_test_case":
            query = arguments.get("query", "")
            cases = _find_test_cases(query, db)

            if not cases:
                return {"result": f"❌ 未找到匹配「{query}」的测试用例。请检查用例名称或先用 /list 查看所有用例。"}

            if len(cases) > 5:
                lines = [f"🔍 找到 {len(cases)} 个匹配「{query}」的用例，太多结果，请更精确地指定："]
                for c in cases[:10]:
                    lines.append(f"  - [ID:{c.id}] {c.name}")
                return {"result": "\n".join(lines)}

            if len(cases) > 1:
                lines = [f"🔍 找到 {len(cases)} 个匹配「{query}」的用例，请指定要运行哪一个："]
                for c in cases:
                    lines.append(f"  - [ID:{c.id}] {c.name}")
                return {"result": "\n".join(lines)}

            target_case = cases[0]
            run_id = str(uuid.uuid4())[:8]
            status_queue = Queue()
            _active_runs[run_id] = status_queue

            try:
                raw_steps = target_case.steps
                if isinstance(raw_steps, str):
                    steps_data = json.loads(raw_steps)
                else:
                    steps_data = raw_steps
            except Exception:
                steps_data = [{"action": "click", "selector": "body", "value": "",
                               "description": target_case.name}]

            thread = threading.Thread(
                target=_run_test_in_thread,
                args=(target_case.id, target_case.name, target_case.url or "about:blank",
                      steps_data, run_id, status_queue),
                daemon=True
            )
            thread.start()

            return {
                "result": f"🚀 已启动测试: **{target_case.name}**\n\n运行ID: `{run_id}`\n\n"
                          f"测试正在后台执行中...你可以：\n"
                          f"- 发送「查看 {run_id}」查看实时进度\n"
                          f"- 等待执行完成后查看结果",
                "run_id": run_id
            }

        elif tool_name == "run_all_test_cases":
            cases = crud.get_test_cases(db, skip=0, limit=200)
            if not cases:
                return {"result": "📭 当前没有自动化测试用例。"}

            run_ids = []
            for c in cases:
                run_id = str(uuid.uuid4())[:8]
                status_queue = Queue()
                _active_runs[run_id] = status_queue

                try:
                    raw_steps = c.steps
                    if isinstance(raw_steps, str):
                        steps_data = json.loads(raw_steps)
                    else:
                        steps_data = raw_steps
                except Exception:
                    steps_data = [{"action": "click", "selector": "body", "value": "",
                                   "description": c.name}]

                thread = threading.Thread(
                    target=_run_test_in_thread,
                    args=(c.id, c.name, c.url or "about:blank",
                          steps_data, run_id, status_queue),
                    daemon=True
                )
                thread.start()
                run_ids.append(run_id)

            cases_list = "\n".join([f"  - [ID:{c.id}] {c.name} ({rid})"
                                    for c, rid in zip(cases, run_ids)])
            return {
                "result": f"🚀 已启动全部 {len(cases)} 个测试用例：\n{cases_list}\n\n"
                          f"测试正在后台执行中...你可以发送「查看 [运行ID]」查看实时进度。"
            }

        elif tool_name == "check_run_status":
            run_id = arguments.get("run_id", "")
            if run_id not in _active_runs:
                return {"result": f"❌ 未找到运行ID `{run_id}`。运行可能已完成或ID不正确。"}

            q = _active_runs[run_id]
            messages = []
            found_complete = False

            try:
                while True:
                    msg = q.get(timeout=0.3)
                    if msg.get("type") == "complete":
                        found_complete = True
                        messages.append(msg.get("message", ""))
                        break
                    messages.append(msg.get("message", ""))
            except Exception:
                pass

            if found_complete:
                del _active_runs[run_id]
                return {"result": "\n".join(messages)}
            else:
                if messages:
                    return {"result": "🔄 测试运行中...\n" + "\n".join(messages[-8:])}
                else:
                    return {"result": f"🔄 运行 `{run_id}` 仍在执行中，还没有输出。请稍后再查看。"}

        return {"result": f"未知工具: {tool_name}"}
    finally:
        try:
            db.close()
        except Exception:
            pass


def chat_with_agent(user_message: str, conversation_history: List[Dict] = None):
    """与AI Agent对话，返回AI回复和可能触发的操作"""
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-20:])

    messages.append({"role": "user", "content": user_message})

    try:
        response = chat_client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2000,
            temperature=0.3,
            timeout=120
        )

        ai_message = response.choices[0].message

        result = {
            "reply": ai_message.content or "",
            "tool_calls": [],
            "run_id": None
        }

        if ai_message.tool_calls:
            for tc in ai_message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except Exception:
                    arguments = {}

                logger.info(f"🔧 AI调用工具: {tool_name}({arguments})")
                tool_result = execute_tool_call(tool_name, arguments)

                result["tool_calls"].append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": tool_result.get("result", "")
                })

                if tool_result.get("run_id"):
                    result["run_id"] = tool_result["run_id"]

            if result["tool_calls"]:
                tool_results_text = "\n\n".join([
                    tc["result"] for tc in result["tool_calls"]
                ])

                assistant_tool_msgs = []
                tool_result_msgs = []
                for tc, atc in zip(result["tool_calls"], ai_message.tool_calls):
                    assistant_tool_msgs.append({
                        "role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": atc.id, "type": "function",
                            "function": {"name": atc.function.name,
                                         "arguments": atc.function.arguments}
                        }]
                    })
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": atc.id,
                        "content": tc["result"]
                    })

                follow_up_messages = messages + assistant_tool_msgs + tool_result_msgs

                follow_up = chat_client.chat.completions.create(
                    model=AI_MODEL,
                    messages=follow_up_messages,
                    max_tokens=2000,
                    temperature=0.3,
                    timeout=120
                )

                ai_reply = follow_up.choices[0].message.content
                if ai_reply:
                    result["reply"] = ai_reply
                else:
                    result["reply"] = tool_results_text

        return result

    except Exception as e:
        logger.error(f"AI对话异常: {e}")
        return {
            "reply": f"❌ AI服务暂时不可用: {str(e)}",
            "tool_calls": [],
            "run_id": None
        }


def get_run_status_stream(run_id: str) -> Generator[str, None, None]:
    """SSE流式获取测试运行状态"""
    if run_id not in _active_runs:
        yield f"data: {json.dumps({'type': 'error', 'message': '运行ID不存在或已完成', 'run_id': run_id}, ensure_ascii=False)}\n\n"
        return

    q = _active_runs[run_id]

    while True:
        try:
            msg = q.get(timeout=30)
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

            if msg.get("type") == "complete":
                del _active_runs[run_id]
                break
        except Exception:
            yield f"data: {json.dumps({'type': 'ping', 'run_id': run_id, 'message': '⏳ 等待中...'}, ensure_ascii=False)}\n\n"