import json
import os
import uuid
import logging
from playwright.sync_api import sync_playwright
import allure
from allure_commons.types import AttachmentType
from . import crud
from .ai_engine import execute_with_fallback, _split_selectors
from .notifier import send_test_result_notification

logger = logging.getLogger(__name__)

AUTH_STATE_FILE = "auth_state.json"
os.makedirs("allure-report", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


def run_web_test(
        case_id: int,
        url: str,
        steps,  # 兼容 str 或 list
        is_login_case: bool = False,
        db=None
):
    """
    纯通用 Web 自动化测试执行引擎，无任何业务硬编码。
    特性：
    - AI 自愈定位
    - 用例级失败自动重试（最多 1 次）
    - 自动保存已被 AI 更新的定位器
    - 执行完成后发送飞书/钉钉/企业微信通知
    - 自动保存/复用登录态
    """
    task_id = str(uuid.uuid4())

    # 兼容字符串和列表两种 steps 格式
    if isinstance(steps, str):
        steps_list = json.loads(steps)
    elif isinstance(steps, list):
        steps_list = steps
    else:
        raise ValueError(f"steps 字段类型错误: {type(steps)}")

    max_test_retry = 0
    final_status = "success"
    final_log = []
    last_exception = None

    for retry_attempt in range(max_test_retry + 1):
        if retry_attempt > 0:
            logger.info(f"⚠️ 第 {retry_attempt} 次执行失败，开始自动重试...")

        updated_steps = [step.copy() for step in steps_list]
        step_updated = False
        execution_log = []

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False, slow_mo=500)

                # ===================== 修复：统一登录态加载逻辑 =====================
                # 非登录用例优先加载已保存的登录态
                if not is_login_case and os.path.exists(AUTH_STATE_FILE):
                    try:
                        context = browser.new_context(storage_state=AUTH_STATE_FILE)
                        execution_log.append("🔐 检测到已保存的登录态，自动复用登录会话！")
                        logger.info("✅ 已从 auth_state.json 加载登录态")
                    except Exception as e:
                        execution_log.append(f"⚠️ 登录态加载失败，将使用全新会话: {e}")
                        logger.warning(f"登录态加载失败: {e}")
                        context = browser.new_context()
                else:
                    context = browser.new_context()
                    if is_login_case:
                        execution_log.append("🆕 执行登录用例，完成后将自动保存登录态")
                        logger.info("🆕 开始执行登录用例，将保存登录态")
                # ==================================================================

                page = context.new_page()
                page.set_viewport_size({"width": 1920, "height": 1080})
                page.set_default_timeout(20000)

                with allure.step(f"打开测试页面: {url}"):
                    page.goto(url, timeout=60000)
                    page.wait_for_load_state("networkidle", timeout=60000)
                    page.wait_for_timeout(2000)

                    screenshot_path = f"screenshots/{task_id}_open.png"
                    page.screenshot(path=screenshot_path)
                    allure.attach.file(screenshot_path, name="打开页面",
                                       attachment_type=AttachmentType.PNG)
                    execution_log.append("✅ 成功打开测试页面")

                for idx, step in enumerate(updated_steps):
                    step_num = idx + 1
                    action = step.get("action")
                    original_selector = step.get("selector")
                    value = step.get("value")
                    expected = step.get("expected")
                    desc = step.get("description", f"步骤{step_num}")

                    execution_log.append(f"\n步骤{step_num}: {desc}")
                    execution_log.append(f"原定位器组合: {original_selector}")

                    selector_str = original_selector or ""
                    is_dropdown_step = (
                        "ant-select-dropdown" in selector_str
                        or "ant-select-item" in selector_str
                        or "option" in selector_str.lower()
                        or (action in ("hover",) and "下拉" in desc)
                    ) or (
                        "title=" in selector_str and (
                            "ant-select" in selector_str.lower()
                            or "dropdown" in selector_str.lower()
                            or "option" in selector_str.lower()
                        )
                    )

                    if not is_dropdown_step:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    else:
                        page.wait_for_timeout(300)
                    page.wait_for_timeout(500)

                    with allure.step(f"步骤{step_num}: {desc}"):
                        try:
                            final_selector = execute_with_fallback(
                                page=page,
                                selector_str=original_selector,
                                action=action,
                                value=value if action in ("fill", "assert_text", "select") else None,
                                desc=desc
                            )
                            execution_log.append(f"   ✅ 操作成功: {desc}")

                            # 点击操作后自动等待页面稳定
                            if action == "click":
                                page.wait_for_load_state("networkidle", timeout=5000)

                            # 保存 AI 更新后的定位器
                            original_selectors = [s.strip() for s in _split_selectors(original_selector) if s.strip()]
                            if original_selectors and final_selector != original_selectors[0]:
                                new_selectors = [final_selector] + [s for s in original_selectors if
                                                                    s != final_selector]
                                new_selector_str = "|".join(new_selectors)
                                execution_log.append(f"🔄 主选择器已失效，自动更新定位器：")
                                execution_log.append(f"   旧: {original_selector}")
                                execution_log.append(f"   新: {new_selector_str}")
                                updated_steps[idx]["selector"] = new_selector_str
                                step_updated = True

                            # 成功截图
                            screenshot_path = f"screenshots/{task_id}_step_{step_num}.png"
                            page.screenshot(path=screenshot_path)
                            allure.attach.file(screenshot_path, name=f"步骤{step_num}截图",
                                               attachment_type=AttachmentType.PNG)

                        except Exception as e:
                            # 步骤失败截图
                            error_screenshot = f"screenshots/{task_id}_error_step_{step_num}.png"
                            page.screenshot(path=error_screenshot)
                            allure.attach.file(error_screenshot, name=f"步骤{step_num}失败截图",
                                               attachment_type=AttachmentType.PNG)
                            execution_log.append(f"   ❌ 步骤执行失败: {str(e)}")
                            raise

                # ===================== 修复：登录用例执行成功后强制保存登录态 =====================
                if is_login_case and final_status == "success":
                    try:
                        context.storage_state(path=AUTH_STATE_FILE)
                        execution_log.append("\n💾 登录态已成功保存！后续所有用例将自动复用此登录状态")
                        logger.info(f"💾 登录态已保存至 {AUTH_STATE_FILE}")
                    except Exception as e:
                        execution_log.append(f"\n⚠️ 登录态保存失败: {e}")
                        logger.error(f"登录态保存失败: {e}")
                # ==================================================================

                browser.close()
                execution_log.append("\n🎉 所有测试步骤执行完成！")

                if step_updated and db:
                    _save_updated_case(db, case_id, url, updated_steps)
                    execution_log.append("\n✅ 已自动更新测试用例的定位器！下次执行将直接使用最优选择器")

                final_status = "success"
                final_log = execution_log
                break

        except Exception as e:
            last_exception = e
            if retry_attempt < max_test_retry:
                try:
                    browser.close()
                except Exception:
                    pass
                continue

            final_status = "failed"
            final_log = execution_log
            if step_updated and db:
                try:
                    _save_updated_case(db, case_id, url, updated_steps)
                    final_log.append("\n⚠️ 测试执行失败，但已自动保存已更新的定位器。")
                except Exception as save_err:
                    final_log.append(f"\n⚠️ 保存定位器失败: {save_err}")

    # ───────── 生成 Allure 报告 ─────────
    report_path = f"allure-report/{task_id}"
    os.system(f"allure generate allure-results -o {report_path} --clean")
    if final_status == "success":
        result_msg = "\n".join(final_log) + f"\n\n✅ 测试通过！报告路径：{report_path}"
    else:
        result_msg = "\n".join(final_log) + f"\n\n❌ 测试失败：{last_exception}\n报告路径：{report_path}"

    # ───────── 发送通知 ─────────
    try:
        case_name = steps_list[0].get("description", f"用例{case_id}") if steps_list else f"用例{case_id}"
        send_test_result_notification(
            case_name=case_name,
            status=final_status,
            result_msg=result_msg,
            report_path=report_path,
        )
    except Exception as noti_err:
        logger.warning(f"⚠️ 发送通知失败: {noti_err}")

    return final_status, result_msg


def _save_updated_case(db, case_id, url, updated_steps):
    """将更新后的步骤写回数据库（不修改用例名称、URL等其他属性）"""
    # 只更新步骤，保持其他属性不变
    case_data = {
        "steps": json.dumps(updated_steps, ensure_ascii=False)
    }
    crud.update_test_case(db=db, case_id=case_id, case=case_data)