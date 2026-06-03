from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from . import models, schemas, crud, test_runner, ai_engine
from .database import engine, get_db
from pydantic import BaseModel
import asyncio
import json
import logging
import uuid
import re
import os
from typing import List, Tuple, Optional
from fastapi.concurrency import run_in_threadpool
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

recording_sessions = {}

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI自动化测试平台", version="1.0.0")

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.warning(f"已自动创建 {STATIC_DIR} 目录，请将 index.html 等前端资源放入其中")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

AUTH_STATE_FILE = "auth_state.json"


@app.get("/")
def read_root():
    if os.path.exists(os.path.join(STATIC_DIR, "index.html")):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    return FileResponse("index.html")


# ------------------- 请求模型 -------------------
class AILocatorRequest(BaseModel):
    url: str
    element_description: str


class AIGenerateStepsRequest(BaseModel):
    url: str
    requirement: str


class AIGenerateCasesRequest(BaseModel):
    url: str
    requirement: str


class BatchRunRequest(BaseModel):
    case_ids: List[int]


class BatchQueryRequest(BaseModel):
    task_ids: List[int]


class AIDocumentGenerateRequest(BaseModel):
    document_text: str = ""
    doc_type: str = "prd"
    url: str = ""
    source_url: str = ""
    login_url: str = ""
    login_username: str = ""
    login_password: str = ""
    login_username_selector: str = ""
    login_password_selector: str = ""
    login_submit_selector: str = ""


class AIExploratoryGenerateRequest(BaseModel):
    url: str
    login_url: str = ""
    username: str = ""
    password: str = ""
    username_selector: str = ""
    password_selector: str = ""
    submit_selector: str = ""


# ------------------- 测试用例接口 -------------------
@app.post("/api/test-cases/", response_model=schemas.TestCaseResponse)
def create_test_case(case: schemas.TestCaseCreate, db: Session = Depends(get_db)):
    return crud.create_test_case(db=db, case=case)


@app.get("/api/test-cases/", response_model=list[schemas.TestCaseResponse])
def read_test_cases(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_test_cases(db, skip=skip, limit=limit)


@app.get("/api/test-cases/{case_id}", response_model=schemas.TestCaseResponse)
def read_test_case(case_id: int, db: Session = Depends(get_db)):
    db_case = crud.get_test_case(db, case_id=case_id)
    if db_case is None:
        raise HTTPException(status_code=404, detail="测试用例不存在")
    return db_case


# ------------------- 测试执行接口 -------------------
@app.post("/api/test-cases/{case_id}/run", response_model=schemas.TestTaskResponse)
def run_test_case(case_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_case = crud.get_test_case(db, case_id=case_id)
    if not db_case:
        raise HTTPException(status_code=404, detail="测试用例不存在")
    db_task = crud.create_test_task(db, case_id=case_id)
    background_tasks.add_task(execute_test_task, db_task.id, db_case, db)
    return db_task


@app.get("/api/test-tasks/{task_id}", response_model=schemas.TestTaskResponse)
def read_test_task(task_id: int, db: Session = Depends(get_db)):
    db_task = crud.get_test_task(db, task_id=task_id)
    if not db_task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return db_task


# ------------------- 批量执行 -------------------
@app.post("/api/test-cases/batch-run")
def batch_run_test_cases(req: BatchRunRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    task_ids = []
    for cid in req.case_ids:
        case = crud.get_test_case(db, cid)
        if case:
            task = crud.create_test_task(db, case_id=cid)
            background_tasks.add_task(execute_test_task, task.id, case, db)
            task_ids.append(task.id)
    return {"task_ids": task_ids, "count": len(task_ids)}


@app.post("/api/test-tasks/batch", response_model=List[schemas.TestTaskResponse])
def batch_read_test_tasks(req: BatchQueryRequest, db: Session = Depends(get_db)):
    return crud.get_test_tasks_by_ids(db, req.task_ids)


# ------------------- 后台执行函数 -------------------
def execute_test_task(task_id: int, test_case: models.TestCase, db: Session):
    crud.update_test_task(db, task_id, status="running")
    try:
        raw_steps = test_case.steps
        if isinstance(raw_steps, str):
            steps_data = json.loads(raw_steps)
        elif isinstance(raw_steps, list):
            steps_data = raw_steps
        else:
            raise ValueError(f"steps 字段格式错误：{type(raw_steps)}")

        if test_case.test_type == "web":
            is_login = (
                    "登录" in test_case.name
                    or "login" in test_case.name.lower()
                    or "登录" in test_case.description
                    or any(
                step.get("action") == "fill" and
                ("password" in step.get("selector", "").lower() or "pwd" in step.get("selector", "").lower())
                for step in steps_data
            )
            )
            status, result = test_runner.run_web_test(
                case_id=test_case.id,
                url=test_case.url,
                steps=steps_data,
                is_login_case=is_login,
                db=db
            )
        else:
            status = "failed"
            result = "暂不支持的测试类型"
        crud.update_test_task(db, task_id, status=status, result=result)
    except Exception as e:
        crud.update_test_task(db, task_id, status="failed", result=f"执行异常: {str(e)}")


# ------------------- AI能力接口 -------------------
@app.post("/api/ai/generate-locator")
async def api_ai_generate_locator(request: AILocatorRequest):
    try:
        locator = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_locator, request.url, request.element_description),
            timeout=120.0
        )
        return {"success": True, "locator": locator}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI生成定位器超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")


@app.post("/api/ai/generate-steps")
async def api_ai_generate_steps(request: AIGenerateStepsRequest):
    try:
        steps = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_test_steps, request.url, request.requirement),
            timeout=120.0
        )
        return {"success": True, "steps": steps}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI生成步骤超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")


@app.post("/api/ai/generate-cases")
async def api_ai_generate_cases(request: AIGenerateCasesRequest):
    try:
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_comprehensive_test_cases, request.url, request.requirement),
            timeout=180.0
        )
        return {"success": True, "cases": cases, "count": len(cases)}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI生成多场景用例超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")


@app.post("/api/ai/generate-and-save-cases")
async def api_ai_generate_and_save_cases(request: AIGenerateCasesRequest, db: Session = Depends(get_db)):
    try:
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_comprehensive_test_cases, request.url, request.requirement),
            timeout=180.0
        )
        saved_cases = _save_generated_cases(cases, db)
        return {
            "success": True,
            "saved_count": len(saved_cases),
            "cases": saved_cases,
            "message": f"成功生成并保存{len(saved_cases)}个测试用例"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI生成多场景用例超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成并保存失败：{str(e)}")


def _save_generated_cases(cases: list, db: Session) -> list:
    """统一保存AI生成的用例"""
    saved_cases = []
    for case in cases:
        case_id = case.get("id", "")
        case_name = case.get("name", "")
        display_name = f"{case_id} {case_name}" if case_id else case_name

        preconditions = case.get("preconditions", "")
        expected_result = case.get("expected_result", "")
        scenario_type = case.get("scenario_type", "")

        description_parts = []
        if scenario_type:
            description_parts.append(f"【场景类型】{scenario_type}")
        if preconditions:
            description_parts.append(f"【前置条件】{preconditions}")
        if expected_result:
            description_parts.append(f"【预期结果】{expected_result}")
        description_parts.append(case.get("description", case_name))

        case_create = schemas.TestCaseCreate(
            name=display_name,
            description="\n".join(description_parts),
            url=case["url"],
            test_type=case["test_type"],
            steps=json.dumps(case["steps"], ensure_ascii=False)
        )
        db_case = crud.create_test_case(db=db, case=case_create)
        saved_cases.append(db_case)
    return saved_cases


@app.post("/api/ai/generate-from-document")
async def api_ai_generate_from_document(request: AIDocumentGenerateRequest):
    try:
        document_text = request.document_text
        if request.source_url and not document_text:
            document_text = await asyncio.wait_for(
                run_in_threadpool(ai_engine.fetch_url_content, request.source_url,
                                  request.login_url, request.login_username,
                                  request.login_password, request.login_username_selector,
                                  request.login_password_selector, request.login_submit_selector),
                timeout=180.0
            )
        if not document_text:
            raise HTTPException(status_code=400, detail="请提供文档内容或网页URL")
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_cases_from_document,
                              document_text, request.doc_type, request.url),
            timeout=180.0
        )
        return {"success": True, "cases": cases, "count": len(cases)}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI文档解析生成超时")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"文档解析生成失败：{str(e)}")


@app.post("/api/ai/generate-from-document-and-save")
async def api_ai_generate_from_document_and_save(request: AIDocumentGenerateRequest, db: Session = Depends(get_db)):
    try:
        document_text = request.document_text
        if request.source_url and not document_text:
            document_text = await asyncio.wait_for(
                run_in_threadpool(ai_engine.fetch_url_content, request.source_url,
                                  request.login_url, request.login_username,
                                  request.login_password, request.login_username_selector,
                                  request.login_password_selector, request.login_submit_selector),
                timeout=180.0
            )
        if not document_text:
            raise HTTPException(status_code=400, detail="请提供文档内容或网页URL")
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_cases_from_document,
                              document_text, request.doc_type, request.url),
            timeout=180.0
        )
        saved_cases = _save_generated_cases(cases, db)
        return {
            "success": True,
            "saved_count": len(saved_cases),
            "cases": saved_cases,
            "message": f"成功从文档生成并保存{len(saved_cases)}个测试用例"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI文档解析生成超时")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"文档解析生成并保存失败：{str(e)}")


@app.post("/api/ai/generate-from-document-upload")
async def api_ai_generate_from_document_upload(
    file: UploadFile = File(...),
    doc_type: str = Form(default="prd"),
    url: str = Form(default=""),
    save: bool = Form(default=False),
    db: Session = Depends(get_db)
):
    try:
        file_bytes = await file.read()
        document_text = await asyncio.wait_for(
            run_in_threadpool(ai_engine.parse_document_file, file.filename, file_bytes),
            timeout=30.0
        )
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_cases_from_document,
                              document_text, doc_type, url),
            timeout=180.0
        )
        if save:
            saved_cases = _save_generated_cases(cases, db)
            return {
                "success": True,
                "saved_count": len(saved_cases),
                "cases": saved_cases,
                "parsed_text": document_text[:500],
                "message": f"成功上传解析并生成{len(saved_cases)}个测试用例"
            }
        return {
            "success": True,
            "cases": cases,
            "count": len(cases),
            "parsed_text": document_text[:500],
            "message": f"成功上传解析并生成{len(cases)}个用例"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI文档解析生成超时")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传文档生成失败：{str(e)}")


@app.post("/api/ai/generate-exploratory")
async def api_ai_generate_exploratory(request: AIExploratoryGenerateRequest):
    try:
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_exploratory_cases,
                              request.url, request.login_url, request.username,
                              request.password, request.username_selector,
                              request.password_selector, request.submit_selector),
            timeout=240.0
        )
        return {"success": True, "cases": cases, "count": len(cases)}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI探索性生成超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"探索性生成失败：{str(e)}")


@app.post("/api/ai/generate-exploratory-and-save")
async def api_ai_generate_exploratory_and_save(request: AIExploratoryGenerateRequest, db: Session = Depends(get_db)):
    try:
        cases = await asyncio.wait_for(
            run_in_threadpool(ai_engine.ai_generate_exploratory_cases,
                              request.url, request.login_url, request.username,
                              request.password, request.username_selector,
                              request.password_selector, request.submit_selector),
            timeout=240.0
        )
        saved_cases = _save_generated_cases(cases, db)
        return {
            "success": True,
            "saved_count": len(saved_cases),
            "cases": saved_cases,
            "message": f"成功探索性生成并保存{len(saved_cases)}个测试用例"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI探索性生成超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"探索性生成并保存失败：{str(e)}")

# ===================== 录制 WebSocket 接口 =====================
@app.websocket("/ws/record")
async def record_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"🆕 新录制会话: {session_id}")
    recording_sessions[session_id] = {
        "steps": [],
        "browser": None,
        "page": None
    }

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            logger.info(f"📥 收到指令: {action} | 会话: {session_id} | 数据: {data}")

            if action == "start":
                url = data.get("url")
                logger.info(f"🚀 启动录制，URL: {url}")
                pw = await async_playwright().start()

                browser = await pw.chromium.launch(
                    headless=False,
                    slow_mo=500,
                    args=['--start-maximized']
                )

                if os.path.exists(AUTH_STATE_FILE):
                    context = await browser.new_context(
                        storage_state=AUTH_STATE_FILE,
                        no_viewport=True
                    )
                    logger.info("🔐 已复用登录态")
                else:
                    context = await browser.new_context(no_viewport=True)

                page = await context.new_page()

                def on_console(msg):
                    text = msg.text
                    logger.info(f"🌐 浏览器日志: [{msg.type}] {text}")
                    match = re.search(r'message:\s*"?([\u4e00-\u9fa5，。！？、；：“”‘’（）《》【】\w\s]+?)"?(?:\s*[,}]|$)', text)
                    if match:
                        notif_text = match.group(1).strip()
                        if re.search(r'[\u4e00-\u9fa5]', notif_text):
                            steps = recording_sessions[session_id]["steps"]
                            if not steps or steps[-1].get("action") != "notification" or steps[-1].get(
                                    "value") != notif_text:
                                step = {
                                    "action": "notification",
                                    "selector": "",
                                    "value": notif_text,
                                    "description": f"检测到通知：{notif_text}"
                                }
                                steps.append(step)
                                logger.info(f"   🔔 从控制台日志捕获通知: {notif_text}")

                page.on("console", on_console)

                # 新增：捕获前端JS错误并记录
                def on_pageerror(error):
                    error_msg = f"前端JS错误: {error.message[:100]}"
                    logger.error(f"🔴 {error_msg}")
                    step = {
                        "action": "notification",
                        "selector": "",
                        "value": error_msg,
                        "description": error_msg
                    }
                    recording_sessions[session_id]["steps"].append(step)

                page.on("pageerror", on_pageerror)

                def record_click(selector: str, text: str = "", tag_name: str = ""):
                    if selector:
                        description = "点击"
                        if text:
                            clean_text = text.strip()
                            if clean_text:
                                if len(clean_text) > 20:
                                    clean_text = clean_text[:20] + "..."
                                description += f"【{clean_text}】"
                            else:
                                description += f"元素 {selector}"
                        else:
                            description += f"元素 {selector}"
                        step = {
                            "action": "click",
                            "selector": selector,
                            "value": "",
                            "description": description
                        }
                        recording_sessions[session_id]["steps"].append(step)
                        logger.info(f"   🖱️ 录制点击: {selector}")

                def record_fill(selector: str, value: str, text: str = "", tag_name: str = "", placeholder: str = ""):
                    if selector:
                        description = ""
                        if placeholder:
                            description = f"输入{placeholder}：{value}"
                        elif text:
                            description = f"在【{text}】输入：{value}"
                        else:
                            description = f"在 {selector} 输入：{value}"
                        step = {
                            "action": "fill",
                            "selector": selector,
                            "value": value,
                            "description": description
                        }
                        recording_sessions[session_id]["steps"].append(step)
                        logger.info(f"   ⌨️ 录制输入: {selector} -> {value}")

                def record_notification(text: str):
                    if text:
                        step = {
                            "action": "notification",
                            "selector": "",
                            "value": text,
                            "description": f"检测到通知：{text}"
                        }
                        recording_sessions[session_id]["steps"].append(step)
                        logger.info(f"   🔔 录制通知: {text}")

                def record_hover(selector: str, text: str = "", tag_name: str = ""):
                    if selector:
                        description = "鼠标悬浮到"
                        if text:
                            clean_text = text.strip()
                            if clean_text:
                                if len(clean_text) > 20:
                                    clean_text = clean_text[:20] + "..."
                                description += f"【{clean_text}】"
                            else:
                                description += f"元素 {selector}"
                        else:
                            description += f"元素 {selector}"
                        step = {
                            "action": "hover",
                            "selector": selector,
                            "value": "",
                            "description": description
                        }
                        recording_sessions[session_id]["steps"].append(step)
                        logger.info(f"   🖱️ 录制悬浮: {selector}")

                def record_contextmenu(selector: str, text: str = "", tag_name: str = ""):
                    step = {
                        "action": "contextmenu",
                        "selector": selector,
                        "value": "",
                        "description": f"右键点击【{text or selector}】"
                    }
                    recording_sessions[session_id]["steps"].append(step)
                    logger.info(f"   🖱️ 录制右键: {selector}")

                await page.expose_function("__recordClick", record_click)
                await page.expose_function("__recordFill", record_fill)
                await page.expose_function("__recordNotification", record_notification)
                await page.expose_function("__recordHover", record_hover)
                await page.expose_function("__recordContextMenu", record_contextmenu)
                logger.info("✅ 已暴露录制回调函数（含右键）")

                init_script = """
function getStableSelector(el) {
    if (!el || el === document.body || el === document.documentElement) return null;

    // 新增：优先处理Ant Design下拉选项（最稳定的选择器）
    if (el.closest && el.closest('.ant-select-item-option')) {
        var option = el.closest('.ant-select-item-option');
        if (option && option.getAttribute('title')) {
            return '[title="' + option.getAttribute('title').replace(/"/g, '\\\\"') + '"]';
        }
    }

    // 优先使用元素自身的 id（排除纯数字、长哈希等）
    if (el.id && !/^\\d+$/.test(el.id) && !/^[a-z0-9]{8,}$/.test(el.id) && el.id.length < 40) {
        return '#' + el.id;
    }

    // 使用 title 属性（下拉选项常用）
    if (el.title && el.title.length < 50) {
        return '[title="' + el.title.replace(/"/g, '\\\\"') + '"]';
    }

    // data 属性
    if (el.dataset) {
        if (el.dataset.testid) return '[data-testid="' + el.dataset.testid + '"]';
        if (el.dataset.cy) return '[data-cy="' + el.dataset.cy + '"]';
        if (el.dataset.id) return '[data-id="' + el.dataset.id + '"]';
    }

    // name 属性
    if (el.name) return '[name="' + el.name.replace(/"/g, '\\\\"') + '"]';

    // placeholder（仅输入框）
    if (el.placeholder && ["INPUT", "TEXTAREA"].includes(el.tagName)) {
        return 'input[placeholder="' + el.placeholder.replace(/"/g, '\\\\"') + '"]';
    }

    // aria-label
    if (el.getAttribute("aria-label")) {
        return '[aria-label="' + el.getAttribute('aria-label').replace(/"/g, '\\\\"') + '"]';
    }

    // 针对 Ant Design Select 组件，尝试获取内部 input 的 id
    if (el.closest && el.closest('.ant-select-selector')) {
        var input = el.querySelector('input');
        if (input && input.id) {
            return '#' + input.id;
        }
    }

    // 按钮、链接等使用文本内容
    if (["BUTTON", "A", "SPAN", "LI", "DIV", "LABEL"].includes(el.tagName) && el.textContent.trim()) {
        var text = el.textContent.trim().replace(/\\s+/g, ' ').slice(0, 40);
        if (text.length > 2 && !/^[\\d\\W]+$/.test(text) && text.length < 50) {
            if (/[\\u4e00-\\u9fff]/.test(text)) {
                return 'text=/' + text.split('').join('\\\\s*') + '/';
            }
            return 'text="' + text.replace(/"/g, '\\\\"') + '"';
        }
    }

    // 原有类名拼接逻辑
    var cls = el.className && el.className.trim();
    if (cls) {
        var classes = cls.split(/\\s+/).filter(function(c) {
            return !/^[a-z0-9]+(-[a-z0-9]+)*:/.test(c) &&
                   !/\\[.*\\]/.test(c) &&
                   !/^!/.test(c) &&
                   !/^(pt|pr|pb|pl|px|py|mt|mr|mb|ml|mx|my|w|h|min-w|min-h|max-w|max-h)-/.test(c) &&
                   !/^(text|font|leading|tracking|bg|border|rounded|shadow|opacity|z-index)-/.test(c) &&
                   !/^(flex|grid|block|inline|hidden|float|clear|static|relative|absolute|fixed|sticky)-/.test(c) &&
                   !/^(gap|space|justify|items|align|content|self|place)-/.test(c) &&
                   !/^(overflow|scroll|whitespace|break|truncate|cursor|pointer-events|select-none)-/.test(c) &&
                   !/^(ant|el|rc)-[a-z]+(-open|-active|-selected|-disabled|-hover|-focus|-checked|-expanded|-collapsed)$/.test(c) &&
                   !/^(ant|el|rc)-[a-z0-9]+-\\d+/.test(c) &&
                   c.length < 40 && c.length > 1;
        });
        if (classes.length > 0) {
            return el.tagName.toLowerCase() + '.' + classes.join('.');
        }
    }
        // 针对下拉选项，优先使用父元素 title 属性或内部文本
    if (el.closest('.ant-select-item-option')) {
        var option = el.closest('.ant-select-item-option');
        if (option && option.getAttribute('title')) {
            return '[title="' + option.getAttribute('title').replace(/"/g, '\\"') + '"]';
        }
        var content = option.querySelector('.ant-select-item-option-content');
        if (content && content.textContent.trim()) {
            var text = content.textContent.trim();
            if (/[\u4e00-\u9fff]/.test(text)) {
                return 'text=/' + text.split('').join('\\s*') + '/';
            }
            return 'text="' + text.replace(/"/g, '\\"') + '"';
        }
    }

    return el.tagName.toLowerCase();
}

document.addEventListener('click', async function(e) {
    var el = e.target;
    var depth = 0;
    var originalEl = el;
    while (el && depth < 5 && el !== document.body) {
        if ((el.classList && el.classList.contains('cursor-pointer')) ||
            (el.textContent && el.textContent.trim() && (el.tagName === 'BUTTON' || el.tagName === 'A' || el.tagName === 'DIV'))) {
            break;
        }
        el = el.parentElement;
        depth++;
    }
    if (!el || el === document.body) el = originalEl;
    var sel = getStableSelector(el);
    if (sel) {
        var text = el.textContent || '';
        await window.__recordClick(sel, text, el.tagName);
    }
}, true);

document.addEventListener('input', async function(e) {
    var el = e.target;
    if (["INPUT", "TEXTAREA"].includes(el.tagName) && el.type !== "checkbox" && el.type !== "radio") {
        var sel = getStableSelector(el);
        if (sel) {
            var text = el.textContent || '';
            var placeholder = el.placeholder || '';
            await window.__recordFill(sel, el.value, text, el.tagName, placeholder);
        }
    }
}, true);

document.addEventListener('change', async function(e) {
    var el = e.target;
    if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
        var sel = getStableSelector(el);
        var checked = el.checked;
        if (sel) {
            await window.__recordClick(sel, el.labels?.[0]?.innerText || (checked ? '选中' : '取消选中'), el.tagName);
        }
    }
    if (el.classList && (el.classList.contains('ant-select') || el.classList.contains('el-select'))) {
        var selectedText = el.querySelector('.ant-select-selection-item')?.innerText || el.value;
        var sel = getStableSelector(el);
        if (sel) {
            await window.__recordClick(sel, selectedText, 'SELECT');
        }
    }
}, true);

document.addEventListener('contextmenu', async function(e) {
    var el = e.target;
    var sel = getStableSelector(el);
    if (sel) {
        await window.__recordContextMenu(sel, el.textContent || '右键菜单', el.tagName);
    }
}, true);

var hoverTimer = null;
var hoverTarget = null;
document.addEventListener('mouseover', async function(e) {
    var el = e.target;
    var interactive = false;
    var cur = el;
    while (cur && cur !== document.body) {
        var tag = cur.tagName;
        if (tag === 'BUTTON' || tag === 'A') { interactive = true; el = cur; break; }
        if (tag === 'LI' && cur.closest('[class*="ant-menu"], [class*="el-menu"], [role="menu"]')) { interactive = true; el = cur; break; }
        if (cur.getAttribute('aria-haspopup') === 'true') { interactive = true; el = cur; break; }
        if (cur.classList && (cur.classList.contains('ant-dropdown-trigger') ||
                             cur.classList.contains('ant-menu-submenu-title') ||
                             cur.classList.contains('el-dropdown-link') ||
                             cur.classList.contains('dropdown-toggle'))) {
            interactive = true; el = cur; break;
        }
        if (cur.getAttribute('role') === 'button' || cur.getAttribute('role') === 'menuitem') { interactive = true; el = cur; break; }
        if (cur.hasAttribute('data-toggle')) { interactive = true; el = cur; break; }
        if (cur.className && /dropdown|menu|submenu/i.test(cur.className)) { interactive = true; el = cur; break; }
        cur = cur.parentElement;
    }
    if (!interactive) return;
    var text = el.textContent?.trim();
    if (!text) return;
    if (hoverTarget !== el) {
        if (hoverTimer) clearTimeout(hoverTimer);
        hoverTarget = el;
        hoverTimer = setTimeout(async function() {
            var sel = getStableSelector(el);
            if (sel) {
                console.log('🖱️ 录制悬浮:', sel, text);
                await window.__recordHover(sel, text, el.tagName);
            }
            hoverTimer = null;
        }, 200);
    }
}, true);
document.addEventListener('mouseout', function(e) {
    if (!hoverTimer || !hoverTarget) return;
    if (!hoverTarget.contains(e.relatedTarget) && e.target === hoverTarget) {
        clearTimeout(hoverTimer);
        hoverTimer = null;
        hoverTarget = null;
    }
}, true);

function hijackNotifiers() {
    if (window.antd && window.antd.notification) {
        const origNotif = window.antd.notification;
        ['open','info','success','error','warning'].forEach(function(method) {
            const orig = origNotif[method];
            if (typeof orig === 'function') {
                origNotif[method] = function(...args) {
                    const config = args[0] || {};
                    const text = config.description || config.message || '';
                    if (text) {
                        window.__recordNotification(String(text));
                        console.log('🔔 代理 notification:', text);
                    }
                    return orig.apply(this, args);
                };
            }
        });
    }
    if (window.antd && window.antd.message) {
        const origMsg = window.antd.message;
        ['open','info','success','error','warning','loading'].forEach(function(method) {
            const orig = origMsg[method];
            if (typeof orig === 'function') {
                origMsg[method] = function(...args) {
                    const content = typeof args[0] === 'string' ? args[0] : (args[0]?.content || '');
                    if (content) {
                        window.__recordNotification(String(content));
                        console.log('🔔 代理 message:', content);
                    }
                    return orig.apply(this, args);
                };
            }
        });
    }
}
var hijackInterval = setInterval(function() {
    if (window.antd) {
        hijackNotifiers();
        clearInterval(hijackInterval);
    }
}, 200);

const observer = new MutationObserver((mutations) => {
    mutations.forEach(function(mutation) {
        mutation.addedNodes.forEach(function(node) {
            if (node.nodeType === 1) {
                var noticeMsg = node.querySelector?.('.ant-notification-notice-message');
                var noticeDesc = node.querySelector?.('.ant-notification-notice-description');
                if (!noticeDesc) noticeDesc = node.querySelector?.('.ant-message-notice-content');
                if (!noticeMsg) noticeMsg = node.querySelector?.('.ant-alert-message');
                if (!noticeDesc) noticeDesc = node.querySelector?.('.ant-alert-description');

                var text = '';
                if (noticeDesc) text = noticeDesc.textContent.trim();
                else if (noticeMsg) text = noticeMsg.textContent.trim();

                if (text) {
                    window.__recordNotification(text);
                    console.log('🔔 observer 录制通知:', text);
                }
            }
        });
    });
});
observer.observe(document.body, { childList: true, subtree: true });

console.log('✅ 增强版录制监听已注入（支持Popover菜单、下拉、右键）');
"""
                await page.add_init_script(init_script)
                logger.info("✅ 增强监听脚本已注入")

                await page.goto(url)
                recording_sessions[session_id]["browser"] = browser
                recording_sessions[session_id]["page"] = page

                await websocket.send_json({"status": "started", "session_id": session_id})
                logger.info(f"✅ 录制已启动，会话ID: {session_id}")

            elif action == "stop":
                logger.info("🛑 停止录制请求")
                session = recording_sessions.get(session_id)

                if session and session["browser"]:
                    raw_steps = session["steps"]
                    enhanced_steps, has_notification_assert = await ai_enhance_recorded_steps(raw_steps)

                    page = session.get("page")
                    if page and not has_notification_assert and enhanced_steps:
                        try:
                            smart_assert = await asyncio.wait_for(
                                ai_engine.generate_smart_assertion_async(page, enhanced_steps),
                                timeout=60.0
                            )
                            if smart_assert.get("value"):
                                enhanced_steps.append(smart_assert)
                                logger.info("✅ 已追加 AI 智能断言")
                            else:
                                logger.warning("⚠️ AI 生成的断言值为空，跳过追加")
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ AI 智能断言生成超时，跳过")
                        except Exception as e:
                            logger.error(f"❌ AI 断言生成异常: {e}")

                    is_login_operation = any(
                        step.get("action") == "fill" and
                        ("password" in step.get("selector", "").lower() or "pwd" in step.get("selector", "").lower())
                        for step in enhanced_steps
                    ) and any(
                        step.get("action") == "click" and
                        ("login" in step.get("selector", "").lower() or "登录" in step.get("description", ""))
                        for step in enhanced_steps
                    )

                    if is_login_operation:
                        try:
                            context = page.context
                            await context.storage_state(path=AUTH_STATE_FILE)
                            logger.info("💾 检测到登录操作，已自动保存登录态")
                            for step in enhanced_steps:
                                if "description" in step and "登录" not in step["description"]:
                                    step["description"] = f"登录 - {step['description']}"
                        except Exception as e:
                            logger.error(f"❌ 录制登录态保存失败: {e}")

                    await session["browser"].close()
                    logger.info(f"📦 录制完成，最终步骤共 {len(enhanced_steps)} 条")
                    await websocket.send_json({"status": "stopped", "steps": enhanced_steps})

                    if session_id in recording_sessions:
                        del recording_sessions[session_id]
                        logger.info(f"🗑️ 会话 {session_id} 已销毁")
                else:
                    logger.warning("⚠️ 未找到录制会话或浏览器已关闭")

    except WebSocketDisconnect:
        logger.warning(f"⚠️ WebSocket 连接断开，会话: {session_id}")
        session = recording_sessions.get(session_id)
        if session and session["browser"]:
            await session["browser"].close()
        if session_id in recording_sessions:
            del recording_sessions[session_id]


# ------------------- 编辑/删除接口 -------------------
@app.put("/api/test-cases/{case_id}", response_model=schemas.TestCaseResponse)
def update_test_case(case_id: int, case: schemas.TestCaseCreate, db: Session = Depends(get_db)):
    db_case = crud.get_test_case(db, case_id=case_id)
    if not db_case:
        raise HTTPException(status_code=404, detail="用例不存在")
    return crud.update_test_case(db, case_id, case)


@app.delete("/api/test-cases/{case_id}")
def delete_test_case(case_id: int, db: Session = Depends(get_db)):
    if not crud.get_test_case(db, case_id):
        raise HTTPException(status_code=404, detail="用例不存在")
    crud.delete_test_case(db, case_id)
    return {"success": True, "message": "删除成功"}


# ===================== 录制步骤清洗函数 =====================
async def ai_enhance_recorded_steps(raw_steps: list[dict], url: str = "") -> Tuple[list[dict], bool]:
    logger.info("🤖 开始清洗与增强录制步骤")

    def simplify_selector(sel: str) -> str:
        if not sel:
            return sel
        # 新增：强制修复所有text选择器的引号问题（无论是否完整）
        sel = re.sub(r'text=["“]?([^"”]*)["”]?', r'text=\1', sel)
        if sel.startswith("#"):
            return sel
        parts = sel.split(".", 1)
        if len(parts) == 2 and len(parts[1]) > 30:
            tag = parts[0]
            classes = parts[1].split(".")
            stable = [c for c in classes if not re.match(r'^(css|ant|el|rc)-\w{3,}-?\d{1,}[a-zA-Z]*$', c)]
            if stable:
                return f"{tag}.{'.'.join(stable)}"
            ant_classes = [c for c in classes if c.startswith("ant-") and not re.match(r'ant-\w+-\d+', c)]
            if ant_classes:
                return f"{tag}.{'.'.join(ant_classes)}"
        return sel.replace("!.", ".").replace("!bg", "bg").replace("!border", "border").replace("$", r"\$")

    FILTER_PATTERNS = [
        r'\.ant-notification',
        r'\.ant-message',
        r'\.ant-alert',
    ]

    def should_filter(step):
        sel = step.get("selector", "")
        action = step.get("action", "")
        if action == "click" and ("menu" in sel or "dropdown" in sel or "item" in sel or "text=" in sel):
            return False
        if action == "contextmenu":
            return False
        if action == "hover" and ("menu" in sel or "dropdown" in sel):
            return False
        for pat in FILTER_PATTERNS:
            if re.search(pat, sel):
                return True
        return False

    def is_valid_notification_text(text: str) -> bool:
        if not text:
            return False
        if re.search(r'(Failed to execute|is not of type|TypeError|Uncaught |Cannot read propert|is not a function|MutationObserver)', text):
            return False
        if re.search(r'[\u4e00-\u9fff]', text):
            return True
        if len(text) > 5 and not re.match(r'^[0-9a-fA-F\-]+$', text):
            return True
        return False

    try:
        cleaned = []
        i = 0
        while i < len(raw_steps):
            step = raw_steps[i]
            action = step.get("action")
            selector = simplify_selector(step.get("selector", ""))
            # 清理 text 选择器中的多余引号（包括中文引号）
            selector = re.sub(r'text="([^"]*)"', r'text=\1', selector)
            selector = re.sub(r'text=“([^”]*)”', r'text=\1', selector)

            if action == "notification":
                i += 1
                continue

            if should_filter(step):
                i += 1
                continue

            # 过滤无用悬浮
            if action == "hover":
                useless_hover_patterns = [
                    r'div\.flex',
                    r'div\.ant-col',
                    r'div\.ant-row',
                    r'div\.ant-table-footer',
                    r'div\.ant-select-selector',
                    r'div\.ant-cascader-menu',
                    r'li\.ant-menu-item.*ant-menu-item-selected',
                    r'li\.ant-menu-item.*ant-menu-item-active',
                    r'div\.ant-select-item-option-content',
                    r'div\.ant-cascader-menu-item-content',
                ]
                is_useless = False
                for pat in useless_hover_patterns:
                    if re.search(pat, selector):
                        is_useless = True
                        break
                if is_useless:
                    i += 1
                    continue
                if cleaned and cleaned[-1].get("action") == "hover" and cleaned[-1].get("selector") == selector:
                    i += 1
                    continue
                if (i + 1 < len(raw_steps) and
                    raw_steps[i+1].get("action") == "click" and
                    simplify_selector(raw_steps[i+1].get("selector", "")) == selector):
                    i += 1
                    continue

            # 合并连续相同输入框的点击
            if action == "click" and i > 0 and raw_steps[i - 1].get("action") == "click" and selector == raw_steps[i - 1].get("selector"):
                i += 1
                continue

            if action == "fill":
                merged_value = step.get("value", "")
                j = i + 1
                while j < len(raw_steps) and raw_steps[j].get("action") == "fill" and simplify_selector(raw_steps[j].get("selector", "")) == selector:
                    merged_value = raw_steps[j].get("value", "")
                    j += 1
                cleaned.append({
                    "action": "fill",
                    "selector": selector,
                    "value": merged_value,
                    "description": f"在 {selector} 输入：{merged_value}"
                })
                i = j
                continue

            if action == "assert":
                cleaned.append({
                    "action": "assert_text",
                    "selector": selector if selector else "body",
                    "value": step.get("value", ""),
                    "description": step.get("description", "验证操作结果")
                })
                i += 1
                continue

            cleaned.append({
                "action": action,
                "selector": selector,
                "value": step.get("value", ""),
                "description": step.get("description", "")
            })
            i += 1
            # 1.5 后处理：如果 click 选择器是 div 或包含 ant-select-item-option-content，且描述中有【xxx】，则替换为 text=xxx
            for step in cleaned:
                if step.get("action") == "click":
                    sel = step.get("selector", "")
                    desc = step.get("description", "")
                    # 处理通用 div/span 以及 ant-select-item-option-content
                    if sel in ("div", "span") or "ant-select-item-option-content" in sel:
                        match = re.search(r'【(.*?)】', desc)
                        if match:
                            text = match.group(1)
                            if text and len(text) >= 2:
                                step["selector"] = f'text={text}'
                                logger.info(f"   🔧 根据描述修正选择器为文本: {step['selector']}")

        # 1. 删除下拉菜单中的多余悬浮
        i = 0
        while i < len(cleaned) - 2:
            if (cleaned[i].get("action") == "click" and
                "select" in cleaned[i].get("selector", "").lower() and
                cleaned[i+1].get("action") == "hover" and
                cleaned[i+2].get("action") == "click"):
                hover_sel = cleaned[i+1].get("selector", "")
                click_sel = cleaned[i+2].get("selector", "")
                if hover_sel != click_sel:
                    logger.info(f"   🗑️ 删除多余悬浮: {cleaned[i+1].get('description', '')}")
                    del cleaned[i+1]
                    continue
            i += 1

        # 2. 删除无用的容器点击（ant-form-item，全部删除）
        i = 0
        while i < len(cleaned) - 1:
            if (cleaned[i].get("action") == "click" and
                "ant-form-item" in cleaned[i].get("selector", "") and
                cleaned[i+1].get("action") == "fill"):
                logger.info(f"   🗑️ 删除多余容器点击: {cleaned[i].get('description', '')}")
                del cleaned[i]
                continue
            i += 1

        # 2.5 为动态表单项的 fill 自动插入等待步骤
        i = 0
        while i < len(cleaned) - 1:
            if (cleaned[i].get("action") == "click" and
                cleaned[i+1].get("action") == "fill" and
                re.search(r'_\d+_', cleaned[i+1].get("selector", ""))):
                wait_step = {
                    "action": "wait_for_selector",
                    "selector": cleaned[i+1]["selector"],
                    "value": "",
                    "description": f"等待元素出现: {cleaned[i+1]['selector']}"
                }
                cleaned.insert(i+1, wait_step)
                logger.info(f"   ⏳ 自动插入等待步骤: {wait_step['description']}")
                i += 1
            i += 1

        # 3. 为下拉选择操作自动添加等待步骤
        i = 0
        while i < len(cleaned) - 1:
            if (cleaned[i].get("action") == "click" and
                ("select" in cleaned[i].get("selector", "").lower() or
                 cleaned[i].get("selector", "").startswith("#form_item_") and "_typeList" in cleaned[i].get("selector", ""))):
                # 点击下拉后，等待下拉菜单出现
                wait_step = {
                    "action": "wait_for_selector",
                    "selector": ".ant-select-dropdown",
                    "value": "",
                    "description": "等待下拉菜单展开"
                }
                cleaned.insert(i+1, wait_step)
                logger.info(f"   ⏳ 自动为下拉选择添加等待步骤")
                i += 2
                continue
            i += 1

        # 4. 优化菜单容器点击
        for idx, step in enumerate(cleaned):
            if step.get("action") == "click" and ("ul.ant-menu" in step.get("selector", "") or "ant-menu-root" in step.get("selector", "")):
                for j in range(idx - 1, -1, -1):
                    prev_step = cleaned[j]
                    if prev_step.get("action") != "hover":
                        continue
                    hover_sel = prev_step.get("selector", "")
                    hover_desc = prev_step.get("description", "")
                    is_menu_hover = (
                        "span.menu-title" in hover_sel
                        or "title=" in hover_sel
                        or "ant-menu" in hover_sel
                        or ("悬浮" in hover_desc and "【" in hover_desc)
                    )
                    if is_menu_hover:
                        match = re.search(r'【(.*?)】', hover_desc)
                        if "title=" in hover_sel:
                            if match and "【" not in step.get("description", ""):
                                step["description"] = f"点击【{match.group(1)}】"
                            logger.info(f"   🔧 子菜单点击，保留 UL 选择器供播放端智能处理")
                        elif match:
                            menu_text = match.group(1)
                            step["selector"] = f'text="{menu_text}"'
                            step["description"] = f"点击【{menu_text}】"
                            logger.info(f"   🔧 优化菜单点击: {step['selector']}")
                        break

        # 4.5 优化按钮点击：若前面有 hover 且描述含【xxx】，则替换选择器
        for idx in range(1, len(cleaned)):
            step = cleaned[idx]
            if step.get("action") == "click" and "button" in step.get("selector", ""):
                prev_step = cleaned[idx - 1]
                if prev_step.get("action") == "hover":
                    hover_desc = prev_step.get("description", "")
                    match = re.search(r'【(.*?)】', hover_desc)
                    if match:
                        btn_text = match.group(1)
                        if len(btn_text) >= 2:
                            step["selector"] = f'text={btn_text} >> nth=0'
                            step["description"] = f"点击【{btn_text}】"
                            logger.info(f"   🔧 优化按钮点击: {step['selector']}")

        # 5. 从原始步骤中提取有效通知，取最后一条作为断言
        notification_texts = [s["value"] for s in raw_steps if
                              s.get("action") == "notification" and is_valid_notification_text(s["value"])]
        has_notification_assert = False
        if notification_texts:
            last_notification = notification_texts[-1]
            clean_text = last_notification.strip()
            cleaned.append({
                "action": "assert_text",
                "selector": f"text={clean_text}",
                "value": clean_text,
                "description": f"验证通知提示「{clean_text}」"
            })
            has_notification_assert = True
            logger.info(f"📌 从录制通知生成断言: {clean_text}")
        else:
            logger.info("⏭️ 未发现有效的通知文本，跳过通知断言")

        logger.info(f"✅ 清洗完成：原始 {len(raw_steps)} 步 → 优化后 {len(cleaned)} 步")
        return cleaned, has_notification_assert

    except Exception as e:
        logger.error(f"❌ 步骤清洗失败: {e}")
        return raw_steps, False