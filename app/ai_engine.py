import os
import base64
import json
import logging
import re
import io
from typing import List, Dict, Tuple
from functools import wraps
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI, APITimeoutError, APIError
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page, expect
from playwright.async_api import Page as AsyncPage

# ======================== 配置 ========================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 同步客户端（用于同步API）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=60.0,
    max_retries=2
)

# 异步客户端（用于WebSocket中的异步调用，解决事件循环阻塞问题）
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=60.0,
    max_retries=2
)

AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
MAX_RELOCATE_RETRY = int(os.getenv("MAX_RELOCATE_RETRY", "2"))
VIEWPORT = {"width": 1920, "height": 1080}
PAGE_LOAD_TIMEOUT = 60000
AI_REQUEST_TIMEOUT = 90

# ======================== 提示词模板 ========================
SYSTEM_PROMPT_COMMON = """
你是一名 Playwright 自动化测试专家，根据网页截图和 DOM 片段生成 CSS 或文本选择器。

【核心原则】
1. 所有属性必须从提供的 DOM 片段中真实提取，严禁臆造。
2. 每个选择器必须唯一匹配目标元素，避免匹配多个。
3. 选择器优先级（严格按顺序）：
   - data-testid、data-cy 等测试专用属性
   - 页面中真实存在且唯一的 id（包括框架生成的稳定id，如 #form_item_list_0_unitPrice）
   - aria-label、title、placeholder、name、type 等语义属性
   - 稳定的 class 组合（如 .ant-input-number-input:not([readonly])）
   - 正则文本匹配：中文必须用 text=/确\\s*认/
   - 结构选择器：如 tr:first-child td:nth-child(3) input
4. 绝对禁止使用任何动态生成的属性：
   - 随机class（如 .css-123abc、.ant-input-12345、.el-button-67890）
   - 自动生成的临时id（如 #input-123、#el-id-456）
   - style属性、onclick事件、href中的动态参数
5. 禁止使用 Playwright 不支持的伪类或语法：
   - :contains() → 用 :has-text() 代替
   - ::before、::after
   - role 选择器不支持 [form='xxx']、[aria-modal='true'] 等属性
6. 禁止使用 name=xxx 引擎格式，必须用 [name='xxx']。
7. 多元素冲突时通过 readonly、type、class、id 等属性区分。
8. 对于可能存在多个同名文本的情况，必须加上 >> nth=N 明确指定第几个
   示例：text=工作台 >> nth=0
9. 输出 3 个选择器，用 | 分隔，只返回选择器字符串，不要任何解释。
"""

STEP_GENERATION_PROMPT = SYSTEM_PROMPT_COMMON + """
你现在要生成完整的自动化测试步骤，包括最后一步“验证操作是否成功”。

【弹窗中的选择列表】
如果需要在弹窗中选择某一行数据，必须点击该行中的单选框（radio）或复选框。
- 正确选择器：`div[role='dialog'] tr:has-text('目标文本') .ant-radio-input`
- 绝对不要只使用 `:text('目标文本')`，因为可能匹配多个元素导致选择错误。

【验证步骤（assert_text）】
- 必须定位页面上实际显示的数据（例如表格单元格），而不是弹出提示消息。
- 示例：`selector: .ant-table-tbody td:has-text('东莞市雅普文具包装制品有限公司') | text=/东莞市雅普文具包装制品有限公司/`
- 绝对不要使用 .ant-message-success 或类似的临时通知元素。

返回纯JSON数组，每条步骤包含：
- action: fill / click / wait_for_selector / assert_text
- selector: 使用上述规则生成的唯一选择器
- value: (fill/assert_text时) 要输入或验证的文本
- description: 中文描述
不要额外解释。
"""

COMPREHENSIVE_TEST_CASE_PROMPT = """
你是一名资深测试专家，需要根据用户提供的业务需求，生成**全面、高覆盖率**的自动化测试用例集。

【覆盖要求 —— 必须严格遵守】
1. 每个功能点至少生成：1 条正向用例 + 至少 3 条异常/边界用例
2. 必须覆盖以下所有测试场景类型：
   - ✅ 正常流程（Happy Path）：最典型的成功操作流程
   - ✅ 异常操作：重复提交、并发冲突、非法操作、权限不足
   - ✅ 边界条件：最大值、最小值、空值、超长文本、临界值
   - ✅ 权限校验：无权限访问、越权操作、未登录访问
3. 优先覆盖：高频操作 > 核心链路 > 高危边界

【用例字段规范】
每个测试用例必须包含以下字段：
- id：用例编号，格式 TC-{模块缩写}-{3位序号}，如 TC-ORDER-001、TC-LOGIN-002
  模块缩写根据需求自动推断，如：订单→ORDER、登录→LOGIN、客户→CUST、商品→PROD
- name：用例名称，清晰描述测试场景（如"正向-创建销售订单"、"异常-重复提交同一订单"）
- preconditions：前置条件，列出执行该用例前必须满足的条件
- steps：符合Playwright格式的步骤数组，和之前的格式完全一致
- expected_result：预期结果，用中文描述操作后应该出现的现象
- scenario_type：场景类型，取值为：正向流程 / 异常操作 / 边界条件 / 权限校验

【步骤生成规则 —— 与之前完全相同】
- 使用稳定的选择器
- 中文文本用正则处理空格
- 最后一步必须是验证操作结果
- 不要使用临时通知作为验证目标

【输出要求】
- 返回纯JSON数组，每个元素是一个完整的测试用例
- 不要任何额外解释、不要markdown、不要代码块
- 用例数量：每个功能点至少4条（1正+3异常/边界），整体至少8条
"""


# ======================== 工具函数 ========================
def _escape_text_regex_meta(selector: str) -> str:
    """转义 text=/.../ 正则表达式内部的特殊字符（/ 和 |）"""
    result = []
    i = 0
    while i < len(selector):
        if selector[i:i + 6] == 'text=/':
            result.append('text=/')
            i += 6
            while i < len(selector):
                if selector[i] == '\\':
                    if i + 1 < len(selector):
                        result.append(selector[i:i + 2])
                        i += 2
                    else:
                        result.append(selector[i])
                        i += 1
                elif selector[i] == '/':
                    rest = selector[i + 1:i + 10]
                    is_closing = (
                        not rest
                        or rest[0] in (' ', '>', ')', '|', '\n', '\r')
                    )
                    if is_closing:
                        result.append('/')
                        i += 1
                        break
                    else:
                        result.append('\\/')
                        i += 1
                elif selector[i] == '|':
                    result.append('\\|')
                    i += 1
                else:
                    result.append(selector[i])
                    i += 1
        else:
            result.append(selector[i])
            i += 1
    return ''.join(result)


def clean_selector(selector: str) -> str:
    """清理选择器中的常见错误字符，并转义 CSS 特殊字符"""
    selector = selector.strip()
    selector = _escape_text_regex_meta(selector)
    # 彻底解决text选择器引号问题（支持中英文引号、无引号）
    selector = re.sub(r'text=["\u201c\u201d]([^"\u201c\u201d]*)["\u201c\u201d]?', r'text=\1', selector)
    # 修复 :has-text() 中的多余引号（AI 常犯的错误：:has-text(工厂")）
    # 关键：只修复内容不以引号开头的（保护已正确包裹的 :has-text("工厂")）
    selector = re.sub(r':has-text\(([^"\u201c\u201d]+)["\u201c\u201d]\s*\)', r':has-text("\1")', selector)
    selector = re.sub(r':has-text\(([^"\u201c\u201d]+?)\)', r':has-text("\1")', selector)
    # 修复常见的拼写错误
    selector = selector.replace("chhild", "child").replace("seaarch", "search")
    selector = selector.replace("oof-type", "of-type").replace(":first", ":first-of-type")
    # 转义类名中包含的小数点，例如 .leading-3.75 → .leading-3\.75
    selector = re.sub(r'(?<=[\w-])\.(\d+)', r'\\.\1', selector)
    # 转义 $ 符号，避免 Playwright 解析失败
    selector = selector.replace("$", r"\$")
    # 移除末尾多余的引号或分号
    selector = re.sub(r'[;"]$', '', selector)
    return selector


def _escape_css_text(text: str) -> str:
    if not text:
        return ""
    return text.replace('"', '\\"')


def _split_selectors(selector_str: str) -> list:
    """Split selectors by | but not inside text=/.../ regex patterns."""
    parts = []
    current = []
    in_regex = False
    i = 0
    while i < len(selector_str):
        if not in_regex and selector_str[i:i + 6] == 'text=/':
            in_regex = True
            current.append('text=/')
            i += 6
            continue
        if in_regex:
            if selector_str[i] == '\\':
                if i + 1 < len(selector_str):
                    current.append(selector_str[i:i + 2])
                else:
                    current.append(selector_str[i])
                i += 2
                continue
            if selector_str[i] == '/':
                in_regex = False
                current.append('/')
                i += 1
                continue
            current.append(selector_str[i])
            i += 1
            continue
        if selector_str[i] == '|':
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
            i += 1
            continue
        current.append(selector_str[i])
        i += 1

    part = ''.join(current).strip()
    if part:
        parts.append(part)

    return parts


def _fix_json_string(json_str: str) -> str:
    """尝试修复常见的AI返回JSON格式问题"""
    s = json_str
    s = re.sub(r',\s*([}\]])', r'\1', s)
    s = re.sub(r'[\u201c\u201d]', '"', s)
    s = re.sub(r'[\u2018\u2019]', "'", s)
    s = re.sub(r'\uff0c', ',', s)
    s = re.sub(r'\uff1a', ':', s)
    s = re.sub(r'"\s*\n\s*"', '",\n"', s)
    s = re.sub(r'}\s*\n\s*"', '},\n"', s)
    s = re.sub(r']\s*\n\s*"', '],\n"', s)
    s = re.sub(r'"\s*\n\s*\{', '",\n{', s)
    s = re.sub(r'"\s*\n\s*\[', '",\n[', s)
    return s


def extract_json_from_response(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        fixed = _fix_json_string(json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        try:
            last_brace = fixed.rfind('}')
            last_bracket = fixed.rfind(']')
            end_pos = max(last_brace, last_bracket)
            if end_pos > 0:
                truncated = fixed[:end_pos + 1]
                if truncated.startswith('[') and not truncated.endswith(']'):
                    truncated += ']'
                elif truncated.startswith('{') and not truncated.endswith('}'):
                    truncated += '}'
                result = json.loads(truncated)
                logger.info(f"   🔧 JSON修复成功（截断修复）")
                return result
        except json.JSONDecodeError:
            pass

        try:
            brace_count = 0
            bracket_count = 0
            for ch in fixed:
                if ch == '{': brace_count += 1
                elif ch == '}': brace_count -= 1
                elif ch == '[': bracket_count += 1
                elif ch == ']': bracket_count -= 1
            if fixed.startswith('[') and bracket_count > 0:
                fixed = fixed + ']' * bracket_count
            elif fixed.startswith('{') and brace_count > 0:
                fixed = fixed + '}' * brace_count
            result = json.loads(fixed)
            logger.info(f"   🔧 JSON修复成功（补全括号）")
            return result
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("无法解析AI返回的JSON", text, 0)


def retry_api_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except (APITimeoutError, APIError) as e:
                retry_count += 1
                wait_time = 2 ** retry_count
                logger.warning(f"API调用失败 ({retry_count}/{max_retries})，{wait_time}秒后重试: {e}")
                import time
                time.sleep(wait_time)
        raise Exception(f"AI API调用失败，已达最大重试次数 {max_retries}")

    return wrapper


def capture_page_context(url: str) -> Tuple[str, str, str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()
        try:
            page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception:
                    pass
            page.wait_for_timeout(2000)
            title = page.title()
            html = page.content()[:20000]
            screenshot = page.screenshot(full_page=True, type="png")
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
            return title, html, screenshot_b64
        finally:
            browser.close()


def capture_current_page_context(page: Page) -> Tuple[str, str]:
    """同步版本：用于 test_runner.py"""
    page.wait_for_timeout(500)
    html = page.content()[:20000]
    screenshot = page.screenshot(full_page=True, type="png")
    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
    return html, screenshot_b64


async def capture_current_page_context_async(page: AsyncPage) -> Tuple[str, str]:
    """异步版本：用于录制 WebSocket"""
    await page.wait_for_timeout(500)
    html = await page.content()
    html = html[:20000]
    screenshot = await page.screenshot(full_page=True, type="png")
    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
    return html, screenshot_b64


def validate_locators_on_page(url: str, selector_str: str) -> List[str]:
    valid = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        try:
            page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception:
                    pass
            page.wait_for_timeout(2000)
            for sel in _split_selectors(selector_str):
                sel = clean_selector(sel)
                if not sel:
                    continue
                try:
                    count = page.locator(sel).count()
                    if count == 1:
                        valid.append(sel)
                        logger.info(f"   ✅ 验证通过: {sel}")
                    elif count > 1:
                        logger.warning(f"   ⚠️ 匹配{count}个: {sel}")
                except Exception as e:
                    logger.warning(f"   ⚠️ 选择器无效: {sel} -> {e}")
        finally:
            browser.close()
    return valid


# ======================== 核心业务函数 ========================
@retry_api_call
def ai_generate_locator(url: str, element_description: str) -> str:
    logger.info(f"🔍 生成定位器: {element_description}")
    try:
        title, dom, screenshot_b64 = capture_page_context(url)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COMMON},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"页面标题：{title}\n定位元素：{element_description}\nDOM片段：\n{dom}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high"
                        }}
                    ]
                }
            ],
            max_tokens=300,
            temperature=0,
            timeout=AI_REQUEST_TIMEOUT
        )
        selector_str = response.choices[0].message.content.strip()
        logger.info(f"   AI返回原始选择器: {selector_str}")
        valid_selectors = validate_locators_on_page(url, selector_str)
        if valid_selectors:
            return valid_selectors[0]
        fallback = clean_selector(_split_selectors(selector_str)[0])
        logger.warning(f"   ⚠️ 所有选择器均无效，使用降级: {fallback}")
        return fallback
    except Exception as e:
        logger.error(f"❌ 生成定位器失败: {e}", exc_info=True)
        raise


@retry_api_call
def ai_relocate_from_current_page(page: Page, element_desc: str) -> str:
    logger.info("🤖 触发AI实时重定位...")
    try:
        dom, screenshot_b64 = capture_current_page_context(page)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COMMON},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"重新定位元素：{element_desc}\n当前DOM片段：\n{dom}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high"
                        }}
                    ]
                }
            ],
            max_tokens=200,
            temperature=0,
            timeout=AI_REQUEST_TIMEOUT
        )
        new_selector = response.choices[0].message.content.strip()
        new_selector = clean_selector(new_selector)
        valid_selectors = []
        for sel in _split_selectors(new_selector):
            sel = clean_selector(sel.strip())
            if not sel:
                continue
            try:
                count = page.locator(sel).count()
                if count == 1:
                    valid_selectors.append(sel)
                    logger.info(f"   ✅ 重定位验证通过: {sel}")
                elif count > 1:
                    logger.warning(f"   ⚠️ 重定位匹配{count}个: {sel}")
            except Exception as e:
                logger.warning(f"   ⚠️ 重定位无效: {sel} -> {e}")
        final_selector = clean_selector(valid_selectors[0]) if valid_selectors else clean_selector(_split_selectors(new_selector)[0].strip())
        if not final_selector:
            raise Exception("AI重定位返回空选择器")
        logger.info(f"✅ 重定位成功，最终选择器：{final_selector}")
        return final_selector
    except Exception as e:
        logger.error(f"❌ 重定位失败: {e}")
        raise  # ✅ 修复：不再返回空字符串，直接抛出异常


@retry_api_call
def ai_generate_test_steps(url: str, requirement: str) -> List[Dict]:
    logger.info(f"🤖 生成测试步骤: {requirement[:100]}...")
    try:
        title, dom, _ = capture_page_context(url)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": STEP_GENERATION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"页面标题：{title}\n"
                        f"页面URL：{url}\n"
                        f"操作需求：{requirement}\n"
                        f"请生成完整的自动化测试步骤，包括所有必要的中间操作和最终验证。\n"
                        f"页面DOM片段（供参考选择器）：\n{dom}"
                    )
                }
            ],
            max_tokens=2500,
            temperature=0.1,
            timeout=AI_REQUEST_TIMEOUT
        )
        ai_text = response.choices[0].message.content.strip()
        steps = extract_json_from_response(ai_text)
        if isinstance(steps, dict):
            steps = steps.get("steps", [])
        valid_steps = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            action = s.get("action")
            selector = s.get("selector")
            value = s.get("value")
            desc = s.get("description", "")
            if not action or not selector:
                continue
            if action not in ("fill", "click", "wait_for_selector", "assert_text"):
                continue
            if action in ("fill", "assert_text") and not value:
                if action == "assert_text":
                    logger.warning(f"   ⚠️ assert_text缺少期望值，已跳过: {desc}")
                    continue
            valid_steps.append({
                "action": action,
                "selector": selector,
                "value": value or "",
                "description": desc
            })
        logger.info(f"✅ 生成{len(valid_steps)}个有效步骤")
        return valid_steps
    except Exception as e:
        logger.error(f"❌ 生成步骤失败: {e}", exc_info=True)
        raise


@retry_api_call
def ai_generate_comprehensive_test_cases(url: str, requirement: str) -> List[Dict]:
    logger.info(f"🤖 生成多场景用例: {requirement[:100]}...")
    try:
        title, dom, _ = capture_page_context(url)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": COMPREHENSIVE_TEST_CASE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"页面标题：{title}\n"
                        f"页面URL：{url}\n"
                        f"业务需求：{requirement}\n"
                        f"页面DOM片段（供参考选择器）：\n{dom}"
                    )
                }
            ],
            max_tokens=3500,
            temperature=0.1,
            timeout=AI_REQUEST_TIMEOUT
        )
        ai_text = response.choices[0].message.content.strip()
        test_cases = extract_json_from_response(ai_text)
        if isinstance(test_cases, dict):
            test_cases = test_cases.get("test_cases", [])
        valid_cases = []
        for case in test_cases:
            if not case.get("name") or not case.get("steps"):
                continue
            valid_steps = []
            for step in case["steps"]:
                if not step.get("action") or not step.get("selector"):
                    continue
                if step.get("action") not in ("fill", "click", "wait_for_selector", "assert_text"):
                    continue
                if step.get("action") in ("fill", "assert_text") and not step.get("value"):
                    continue
                valid_steps.append(step)
            if valid_steps:
                case["steps"] = valid_steps
                case["url"] = url
                case["test_type"] = "web"
                valid_cases.append(case)
        logger.info(f"✅ 成功生成{len(valid_cases)}个有效测试用例")
        return valid_cases
    except Exception as e:
        logger.error(f"❌ 生成多场景测试用例失败: {e}", exc_info=True)
        raise


# ======================== 步骤执行辅助（含复选框修复） ========================
def _pick_unique_element(locator, action: str, value: str = None, desc: str = ""):
    """当选择器匹配多个元素时，智能选取一个最合适的"""
    count = locator.count()
    if count == 0:
        return None
    if count == 1:
        return locator.first

    desc_text = ""
    if desc:
        desc_match = re.search(r'【(.*?)】', desc)
        if desc_match:
            desc_text = desc_match.group(1).strip()
        if not desc_text and "点击" in desc:
            desc_text = desc.replace("点击", "").strip()

    if action in ("click", "hover") and desc_text:
        for i in range(count):
            el = locator.nth(i)
            try:
                actual = el.text_content() or ""
                if desc_text in actual:
                    logger.info(f"   🎯 从{count}个匹配中定位到描述'{desc_text}'对应的第{i + 1}个元素")
                    return el
            except Exception:
                continue

    if action == "assert_text" and value is not None:
        for i in range(count):
            el = locator.nth(i)
            try:
                actual = el.text_content() or ""
                if value in actual:
                    logger.info(f"   🎯 从{count}个匹配中找到包含'{value}'的第{i + 1}个元素")
                    return el
            except Exception:
                continue
        return None

    if action == "fill":
        for i in range(count):
            el = locator.nth(i)
            try:
                is_readonly = el.evaluate("el => el.readOnly || el.hasAttribute('readonly') || el.disabled")
                if not is_readonly:
                    logger.info(f"   🎯 从{count}个匹配中自动选择了第{i + 1}个可编辑元素")
                    return el
            except Exception:
                continue
    return locator.first


def execute_with_fallback(page: Page, selector_str: str, action: str,
                          value: str = None, desc: str = "", retry_count: int = 0) -> str:
    """
    尝试依次使用 selector_str 中用 | 分隔的多个选择器，
    若全部失效则先尝试本地快速修复，再调用 AI 重定位，最多重试 MAX_RELOCATE_RETRY 次。
    支持动作：click, fill, wait_for_selector, hover, assert_text, select
    """
    last_error = None
    selectors = [clean_selector(s) for s in _split_selectors(selector_str) if clean_selector(s)]

    for selector in selectors:
        try:
            logger.info(f"   🛠️ 尝试选择器: {selector}")
            base_locator = page.locator(selector)

            try:
                count = base_locator.count()
            except Exception as e:
                logger.warning(f"   ⚠️ 选择器内部错误（跳过）: {selector} → {e}")
                continue

            # 动态等待元素出现
            if count == 0:
                try:
                    page.wait_for_selector(selector, state="attached", timeout=10000)
                    base_locator = page.locator(selector)
                    count = base_locator.count()
                except:
                    raise PlaywrightTimeoutError("元素未出现")

            if count == 0:
                raise PlaywrightTimeoutError("元素未出现")

            if count > 1:
                element = _pick_unique_element(base_locator, action, value, desc)
                if element is None:
                    continue
                logger.warning(f"   ⚠️ 选择器匹配到{count}个元素，已自动选取唯一目标")
            else:
                element = base_locator.first

            element.wait_for(state="attached", timeout=3000)

            if action == "click":
                tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                input_type = element.evaluate("el => el.type || ''")
                is_checkable = (tag_name == "input" and input_type in ("checkbox", "radio"))
                if is_checkable:
                    is_checked = element.evaluate("el => el.checked")
                    if is_checked:
                        logger.info("   🔒 复选框/单选框已选中，跳过点击")
                        return selector
                    else:
                        logger.info("   ✅ 复选框/单选框未选中，执行点击")
                        element.click(force=True)
                else:
                    has_inner_select = False
                    try:
                        has_inner_select = element.evaluate(
                            "el => el.querySelector && ("
                            "el.querySelector('.ant-select-selector') !== null || "
                            "el.querySelector('.ant-cascader-picker') !== null || "
                            "el.querySelector('.ant-picker') !== null"
                            ")"
                        )
                    except Exception:
                        pass

                    if has_inner_select:
                        is_select_related = (
                            selector_str.startswith("#form_item_")
                            or "请选择" in desc
                            or "下拉" in desc
                            or "选择" in desc
                            or "select" in desc.lower()
                            or action == "select"
                        )
                        if not is_select_related:
                            logger.info(f"   ⏭️ 跳过 form-item 重定向（非选择器操作）")
                            try:
                                expect(element).to_be_enabled(timeout=5000)
                            except AssertionError:
                                logger.warning("   ⚠️ 按钮可能未启用，尝试强制点击")
                            element.click(force=True, timeout=5000)
                        else:
                            page.keyboard.press("Escape")
                            page.wait_for_timeout(300)
                            logger.info("   ⌨️ 点击选择器前 Escape 关闭遮挡")

                            inner_selectors = [
                                ".ant-select-selector",
                                ".ant-cascader-picker",
                                ".ant-picker",
                            ]
                            clicked_inner = False
                            for inner_sel in inner_selectors:
                                try:
                                    inner_el = element.locator(inner_sel).first
                                    if inner_el.count() > 0:
                                        inner_el.click(timeout=3000)
                                        logger.info(f"   🎯 检测到 form-item 包裹的选择器，改为点击内部 {inner_sel}")
                                        clicked_inner = True
                                        break
                                except Exception:
                                    continue
                            if not clicked_inner:
                                element.click(force=True, timeout=5000)
                    else:
                        if (
                            selector_str.startswith("#form_item_")
                            or "请选择" in desc
                            or "选择" in desc
                        ):
                            page.keyboard.press("Escape")
                            page.wait_for_timeout(300)
                            logger.info("   ⌨️ 点击前 Escape 关闭遮挡（无 inner select）")
                        try:
                            expect(element).to_be_enabled(timeout=5000)
                        except AssertionError:
                            logger.warning("   ⚠️ 按钮可能未启用，尝试强制点击")

                        if tag_name in ("ul", "ol") and desc:
                            desc_text = ""
                            desc_match = re.search(r'【(.*?)】', desc)
                            if desc_match:
                                desc_text = desc_match.group(1).strip()
                            esc_desc = _escape_css_text(desc_text)
                            clicked_menu = False

                            def _is_ant_submenu(el):
                                try:
                                    return el.evaluate(
                                        "el => el.classList.contains('ant-menu-submenu') || el.classList.contains('el-submenu')"
                                    )
                                except Exception:
                                    return False

                            def _click_first_submenu_child(el):
                                for child_sel in ["li.ant-menu-item", "a.ant-menu-item", "li.el-menu-item", "li.ant-menu-item-only-child"]:
                                    try:
                                        fc = el.locator(child_sel).first
                                        if fc.count() > 0:
                                            fc.click(timeout=3000)
                                            return True
                                    except Exception:
                                        continue
                                return False

                            if desc_text:
                                for child_sel in [
                                    "li.ant-menu-item",
                                    "li",
                                    "span.ant-menu-title-content",
                                    "span",
                                    "a",
                                ]:
                                    try:
                                        child = element.locator(
                                            f'{child_sel}:has-text("{esc_desc}")'
                                        ).first
                                        if child.count() > 0:
                                            if _is_ant_submenu(child):
                                                try:
                                                    is_open = child.evaluate(
                                                        "el => el.classList.contains('ant-menu-submenu-open') || el.classList.contains('el-submenu__title') && el.parentElement.classList.contains('is-opened')"
                                                    )
                                                except Exception:
                                                    is_open = False
                                                if is_open:
                                                    if _click_first_submenu_child(child):
                                                        logger.info(
                                                            f"   🎯 子菜单已打开 → 点击第一个子项"
                                                            f"（父级'{desc_text}'）"
                                                        )
                                                        clicked_menu = True
                                                        return selector
                                                else:
                                                    child.click(timeout=2000)
                                                    page.wait_for_timeout(600)
                                                    if _click_first_submenu_child(child):
                                                        logger.info(
                                                            f"   🎯 子菜单已展开 → 点击第一个子项"
                                                            f"（父级'{desc_text}'）"
                                                        )
                                                        clicked_menu = True
                                                        return selector
                                                continue

                                            child.click(timeout=3000)
                                            logger.info(
                                                f"   🎯 菜单容器 UL，改为点击子元素 "
                                                f"{child_sel}（含'{desc_text}'）"
                                            )
                                            clicked_menu = True
                                            return selector
                                    except Exception:
                                        continue
                                try:
                                    text_el = page.locator(f'text="{esc_desc}"').first
                                    if text_el.count() > 0:
                                        if _is_ant_submenu(text_el):
                                            try:
                                                is_open = text_el.evaluate(
                                                    "el => el.classList.contains('ant-menu-submenu-open')"
                                                )
                                            except Exception:
                                                is_open = False
                                            if is_open and _click_first_submenu_child(text_el):
                                                logger.info(
                                                    f"   🎯 全局匹配子菜单已打开 → 点击第一个子项"
                                                    f"（'{desc_text}'）"
                                                )
                                                clicked_menu = True
                                                return selector
                                            elif not is_open:
                                                text_el.click(timeout=2000)
                                                page.wait_for_timeout(600)
                                                if _click_first_submenu_child(text_el):
                                                    logger.info(
                                                        f"   🎯 全局匹配子菜单已展开 → 点击第一个子项"
                                                        f"（'{desc_text}'）"
                                                    )
                                                    clicked_menu = True
                                                    return selector
                                        else:
                                            text_el.click(timeout=3000)
                                            logger.info(f"   🎯 菜单容器，全局 text= 匹配'{desc_text}'")
                                            clicked_menu = True
                                            return selector
                                except Exception:
                                    pass
                            if not clicked_menu:
                                try:
                                    for fallback_sel in [
                                        'li:has([title]:not([title=""])):visible',
                                        "li.ant-menu-item:visible",
                                        "a.ant-menu-item:visible",
                                        "li:has([title]):visible",
                                        "li.menu-item:visible",
                                        "li:visible",
                                        "a:visible",
                                        "[title]:visible",
                                    ]:
                                        visible_items = element.locator(fallback_sel).first
                                        if visible_items.count() > 0:
                                            visible_items.click(timeout=3000)
                                            logger.info(
                                                f"   🎯 菜单容器，fallback 点击第一个可见元素 "
                                                f"({fallback_sel})"
                                            )
                                            return selector
                                except Exception:
                                    pass

                        element.click(force=True, timeout=5000)

            elif action == "fill":
                # 显式等待可见
                try:
                    element.wait_for(state="visible", timeout=10000)
                except Exception:
                    logger.warning(f"   ⚠️ 等待元素可见超时，仍尝试 fill: {selector}")
                tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                input_type = element.evaluate("el => el.type || ''")
                is_checkable = (tag_name == "input" and input_type in ("checkbox", "radio"))
                if is_checkable:
                    logger.info("   🔄 检测到 checkbox/radio，自动将 fill 转换为 click")
                    is_checked = element.evaluate("el => el.checked")
                    if not is_checked:
                        element.click(force=True)
                else:
                    try:
                        element.fill(value, timeout=5000)
                    except Exception:
                        element.click(force=True)
                        page.wait_for_timeout(300)
                        element.fill(value)

                page.wait_for_timeout(200)
                page.keyboard.press("Escape")
                page.wait_for_timeout(400)
                page.keyboard.press("Escape")
                logger.info("   ⌨️ fill 后双重 Escape 关闭 autocomplete")

            elif action == "select":
                # 1. 强制关闭可能存在的遮挡层
                logger.info("   ⌨️ 按下 Escape 关闭遮挡层")
                page.keyboard.press("Escape")
                page.wait_for_timeout(500)
                # 2. 点击 body 角落进一步确保焦点转移
                logger.info("   🖱️ 点击空白处关闭残留浮层")
                page.click("body", position={"x": 10, "y": 10})
                page.wait_for_timeout(300)
                # 3. 点击选择器本身获取焦点
                logger.info(f"   🖱️ 点击选择器: {selector}")
                element.click()
                page.wait_for_timeout(500)
                # 4. 如果是 Ant Design Select，尝试点击内部 input 以确保打开
                if "ant-select" in selector.lower() or "select" in selector.lower():
                    try:
                        input_sel = f"{selector} input"
                        logger.info(f"   🔍 尝试点击内部 input: {input_sel}")
                        page.click(input_sel, timeout=3000)
                        page.wait_for_timeout(500)
                    except:
                        logger.warning("   ⚠️ 未找到内部 input，已通过外层点击打开")
                # 5. 等待下拉容器出现
                try:
                    page.wait_for_selector(".ant-select-dropdown:not(.ant-select-dropdown-hidden)",
                                           state="visible", timeout=5000)
                    logger.info("   ✅ 下拉选项容器已出现")
                except:
                    raise Exception("下拉选项容器未出现，可能仍被遮挡或 Select 未展开")
                # 6. 多策略选择选项（优先title属性，最稳定）
                option_selectors = [
                    f'[title="{_escape_css_text(value) if value else value}"]',
                    f'text={value}',
                    f'.ant-select-item-option:has-text("{_escape_css_text(value) if value else value}")',
                    f'[label="{_escape_css_text(value) if value else value}"]',
                ]
                clicked = False
                for opt_sel in option_selectors:
                    try:
                        logger.info(f"   🔍 尝试选项选择器: {opt_sel}")
                        page.wait_for_selector(opt_sel, state="visible", timeout=3000)
                        page.click(opt_sel)
                        logger.info(f"   ✅ 选择下拉选项: {value}")
                        clicked = True
                        break
                    except:
                        continue
                if not clicked:
                    raise Exception(f"无法在下拉框中找到选项 '{value}'")

            elif action == "wait_for_selector":
                element.wait_for(state="visible", timeout=5000)

            elif action == "hover":
                element.hover(timeout=5000)
                logger.info(f"   ✅ 悬浮成功: {selector}")

            elif action == "assert_text":
                if value is None:
                    logger.warning("   ⚠️ 断言文本为空，自动降级为等待元素可见")
                    element.wait_for(state="visible", timeout=5000)
                else:
                    actual = element.text_content() or ""
                    if value not in actual:
                        raise AssertionError(f"期望「{value}」，实际「{actual}」")

            else:
                raise ValueError(f"未知的动作类型: {action}")

            logger.info(f"   ✅ 执行成功: {selector}")
            return selector

        except Exception as e:
            last_error = e
            logger.warning(f"   ⚠️ 失败: {selector} → {e}")
            continue

    # 本地快速修复（优先于AI调用，节省90%时间）
    if retry_count < MAX_RELOCATE_RETRY and desc:
        logger.info("🔧 尝试本地快速修复选择器...")

        desc_text = ""
        desc_match = re.search(r'【(.*?)】', desc)
        if desc_match:
            desc_text = desc_match.group(1).strip()
        if not desc_text and "点击" in desc:
            desc_text = desc.replace("点击", "").strip()

        is_dropdown_option = (
            "ant-select-item" in selector_str
            or "ant-select-dropdown" in selector_str
            or "title=" in selector_str
            or (desc_text and len(desc_text) <= 10 and "点击" in desc)
        )

        is_date_picker = "ant-picker" in selector_str

        if is_date_picker and action == "click":
            try:
                picker_open = page.locator(
                    ".ant-picker-dropdown:not(.ant-picker-dropdown-hidden)"
                ).count() > 0
                if not picker_open:
                    logger.info("   📅 日期选择器未打开，尝试点击输入框展开...")
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(300)
                    picker_clicked = False
                    for picker_sel in [
                        ".ant-picker-focused input",
                        ".ant-picker-open input",
                        ".ant-picker:not(.ant-picker-disabled) input",
                        ".ant-picker:not(.ant-picker-disabled) .ant-picker-input",
                    ]:
                        try:
                            if page.locator(picker_sel).count() > 0:
                                page.locator(picker_sel).first.click(timeout=2000)
                                picker_clicked = True
                                logger.info(f"   🖱️ 点击 {picker_sel} 展开日期选择器")
                                break
                        except Exception:
                            continue
                    if picker_clicked:
                        page.wait_for_timeout(500)
                        try:
                            page.wait_for_selector(
                                ".ant-picker-dropdown:not(.ant-picker-dropdown-hidden)",
                                state="visible", timeout=4000
                            )
                            logger.info("   ✅ 日期选择器已展开")
                        except Exception:
                            logger.warning("   ⚠️ 日期选择器展开失败")
            except Exception as e:
                logger.warning(f"   ⚠️ 日期选择器展开异常: {e}")

        if is_dropdown_option and action == "click":
            try:
                dropdown_visible = page.locator(
                    ".ant-select-dropdown:not(.ant-select-dropdown-hidden)"
                ).count() > 0
                if not dropdown_visible:
                    logger.info("   🔄 下拉菜单已关闭，尝试重新展开...")
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(300)
                    select_clicked = False
                    for select_sel in [
                        ".ant-select-focused .ant-select-selector",
                        ".ant-select-open .ant-select-selector",
                        ".ant-select:not(.ant-select-disabled) .ant-select-selector",
                    ]:
                        try:
                            if page.locator(select_sel).count() > 0:
                                page.locator(select_sel).first.click(timeout=2000)
                                select_clicked = True
                                logger.info(f"   🖱️ 点击 {select_sel} 重开展开下拉")
                                break
                        except Exception:
                            continue
                    if select_clicked:
                        page.wait_for_timeout(600)
                        try:
                            page.wait_for_selector(
                                ".ant-select-dropdown:not(.ant-select-dropdown-hidden)",
                                state="visible", timeout=4000
                            )
                            logger.info("   ✅ 下拉菜单已重新展开")
                        except Exception:
                            logger.warning("   ⚠️ 下拉菜单展开失败")
            except Exception as e:
                logger.warning(f"   ⚠️ 下拉展开异常: {e}")

        local_fixes = [
            f"{selector_str} >> nth=0",
            re.sub(r'\.css-\w+', '', selector_str),
        ]

        if desc_text:
            esc_desc = _escape_css_text(desc_text)
            local_fixes.extend([
                f'text={desc_text}',
                f'[title="{esc_desc}"]',
                f'text={desc_text} >> nth=0',
            ])

        if is_dropdown_option and desc_text:
            esc_desc2 = _escape_css_text(desc_text)
            local_fixes.extend([
                f'.ant-select-item-option:has-text("{esc_desc2}")',
                f'.ant-select-dropdown:not(.ant-select-dropdown-hidden) >> text={desc_text}',
                f'.ant-select-dropdown >> [title="{esc_desc2}"]',
                f'.ant-select-dropdown >> :has-text("{esc_desc2}")',
            ])

        local_fixes.extend([
            f".ant-layout-sider >> {selector_str}" if "菜单" in desc or "管理" in desc else None,
        ])

        for fix in local_fixes:
            if not fix:
                continue
            fix = clean_selector(fix)
            try:
                logger.info(f"   🛠️ 尝试本地修复: {fix}")
                # 先验证元素是否存在
                page.wait_for_selector(fix, timeout=3000)
                # 递归调用执行操作
                return execute_with_fallback(
                    page=page,
                    selector_str=fix,
                    action=action,
                    value=value,
                    desc=desc,
                    retry_count=retry_count
                )
            except Exception:
                continue

    # AI 自愈重试
    if retry_count < MAX_RELOCATE_RETRY and desc:
        logger.warning(f"⚠️ 全部选择器失效，AI自愈重试 {retry_count + 1}/{MAX_RELOCATE_RETRY}")
        try:
            new_selector = ai_relocate_from_current_page(page, desc)
            return execute_with_fallback(
                page=page,
                selector_str=new_selector,
                action=action,
                value=value,
                desc=desc,
                retry_count=retry_count + 1
            )
        except Exception as ai_err:
            logger.error(f"❌ AI重定位失败: {ai_err}")
            last_error = ai_err

    raise last_error or Exception("重定位失败，已达最大重试次数")


# ======================== 智能断言（异步版，已修复阻塞问题） ========================
async def generate_smart_assertion_async(page: AsyncPage, cleaned_steps: list[dict]) -> dict:
    logger.info("🧠 AI 正在生成智能断言步骤...")
    try:
        dom, screenshot_b64 = await capture_current_page_context_async(page)
        dom_snippet = dom[:5000]

        visible_texts = await page.evaluate("""() => {
            const texts = new Set();
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                { acceptNode: node => {
                    const el = node.parentElement;
                    if (!el) return NodeFilter.FILTER_REJECT;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') return NodeFilter.FILTER_REJECT;
                    const text = node.textContent.trim();
                    if (!text || text.length < 2 || text.length > 100) return NodeFilter.FILTER_REJECT;
                    return NodeFilter.FILTER_ACCEPT;
                }}
            );
            while (walker.nextNode()) {
                texts.add(walker.currentNode.textContent.trim());
            }
            return Array.from(texts).slice(0, 80);
        }""")

        steps_summary = json.dumps([
            {
                "action": s["action"],
                "selector": s.get("selector", ""),
                "value": s.get("value", ""),
                "description": s.get("description", "")
            }
            for s in cleaned_steps[-5:]
        ], ensure_ascii=False)

        system_prompt = """
你是资深测试专家，根据页面截图、DOM片段和已执行的步骤，生成一条 **Playwright assert_text 断言**。

**核心铁律：断言文本必须100%来自截图**
1. 你只能断言**截图中肉眼可见**的文字。用眼睛在截图中找到它，再写进 value。
2. value 必须从下方"页面可见文字"列表中**逐字选取**，一字不改。
3. 严禁根据 DOM 中的 class名、id、placeholder、字段名 来"推测"页面文字。
   - 例如：DOM 中有 `class="totalPrice"`，但截图中显示的是"商品总价"，则 value 必须是"商品总价"，绝不能写成"订单总价"、"总价"、"合计"等。
4. 若截图中找不到任何可验证的文字，返回空断言。

**断言优先级**
1. 错误提示/成功提示（如"提交成功"、"客户已存在，请勿重复添加"）→ 优先
2. 页面标题、表格列名、卡片标题等静态文字（如"商品名称"、"下单时间"）→ 其次
3. 输入框中的值、按钮文字 → 再次
4. 以上都找不到 → 返回空断言

**输出格式**
返回 JSON 对象：
{
  "action": "assert_text",
  "selector": "唯一选择器（优先 text=xxx 或 text=/正\\\\s*则/）",
  "value": "截图中真实存在的文字，逐字复制，不要改动",
  "description": "中文描述"
}
若找不到任何验证目标，返回：
{"action": "assert_text", "selector": "body", "value": "", "description": "验证页面正常加载"}

**正确 vs 错误对照**
✅ 正确：截图中显示"商品总价: ¥1,280.00" → value="商品总价"
✅ 正确：截图中显示"下单时间" → value="下单时间"
❌ 错误：截图中显示"商品总价"，却断言"订单总价"（臆造）
❌ 错误：截图中显示"商品名称"，却断言"产品名"（篡改原文）
❌ 错误：从 DOM `class="orderTotal"` 推测出"订单总价"（应该看截图）
"""
        user_prompt = f"步骤摘要：\n{steps_summary}\n\n页面可见文字（只能从这里选断言文字）：\n{json.dumps(visible_texts, ensure_ascii=False)}\n\n页面DOM片段：\n{dom_snippet}"

        # ✅ 修复：使用异步客户端调用，不阻塞事件循环
        response = await async_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high"
                        }}
                    ]
                }
            ],
            max_tokens=300,
            temperature=0,
            timeout=AI_REQUEST_TIMEOUT
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3]
        assertion = json.loads(result_text)
        assertion["action"] = "assert_text"
        if "selector" not in assertion or not assertion["selector"]:
            assertion["selector"] = "body"
        if "value" not in assertion:
            assertion["value"] = ""
        if "description" not in assertion:
            assertion["description"] = "验证页面操作结果"
        logger.info(f"✅ AI 生成断言: {assertion}")
        return assertion

    except Exception as e:
        logger.error(f"❌ 智能断言生成失败: {e}")
        return {
            "action": "assert_text",
            "selector": "body",
            "value": "",
            "description": "验证页面操作执行完成（AI生成失败）"
        }


# 保留同步占位函数
def generate_smart_assertion(page: Page, cleaned_steps: list[dict]) -> dict:
    logger.warning("generate_smart_assertion 同步版暂不推荐用于录制，请使用 generate_smart_assertion_async")
    return {
        "action": "assert_text",
        "selector": "body",
        "value": "",
        "description": "同步版本占位，请使用异步版本"
    }


# ======================== 需求/文档解析生成测试用例 ========================

def parse_document_file(filename: str, file_bytes: bytes) -> str:
    """解析上传的文档文件，提取文本内容"""
    logger.info(f"📄 解析文档: {filename}, 大小={len(file_bytes)}字节")
    ext = os.path.splitext(filename)[1].lower()

    if ext in ('.txt', '.md', '.markdown'):
        for encoding in ('utf-8', 'gbk', 'gb2312', 'latin-1'):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode('utf-8', errors='replace')

    if ext == '.docx':
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    paragraphs.append(' | '.join(cells))
        return '\n'.join(paragraphs)

    if ext == '.pdf':
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(text.strip())
        return '\n\n'.join(text_parts)

    if ext == '.json':
        content = file_bytes.decode('utf-8')
        data = json.loads(content)
        if isinstance(data, dict):
            swagger_fields = ['paths', 'openapi', 'swagger', 'info']
            if any(k in data for k in swagger_fields):
                return _extract_swagger_text(data)
            return json.dumps(data, ensure_ascii=False, indent=2)
        if isinstance(data, list):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return content

    if ext in ('.yaml', '.yml'):
        try:
            import yaml
            data = yaml.safe_load(file_bytes.decode('utf-8'))
            if isinstance(data, dict):
                swagger_fields = ['paths', 'openapi', 'swagger', 'info']
                if any(k in data for k in swagger_fields):
                    return _extract_swagger_text(data)
                return json.dumps(data, ensure_ascii=False, indent=2)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except ImportError:
            return file_bytes.decode('utf-8')

    try:
        return file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError(f"不支持的文件格式: {ext}，请上传 TXT/DOCX/PDF/MD/JSON/YAML 文件")


def _extract_swagger_text(data: dict) -> str:
    """从 Swagger/OpenAPI JSON 中提取可读的接口描述"""
    lines = []
    info = data.get('info', {})
    if info.get('title'):
        lines.append(f"API名称: {info['title']}")
    if info.get('description'):
        lines.append(f"描述: {info['description']}")

    paths = data.get('paths', {})
    for path, methods in paths.items():
        for method, details in methods.items() if isinstance(methods, dict) else []:
            if not isinstance(details, dict):
                continue
            summary = details.get('summary', '')
            desc = details.get('description', '')
            operation_id = details.get('operationId', '')
            params = details.get('parameters', [])
            request_body = details.get('requestBody', {})
            responses = details.get('responses', {})

            lines.append(f"\n接口: {method.upper()} {path}")
            if summary:
                lines.append(f"  摘要: {summary}")
            if desc:
                lines.append(f"  描述: {desc}")
            if operation_id:
                lines.append(f"  操作ID: {operation_id}")
            if params:
                lines.append(f"  参数:")
                for p in params:
                    lines.append(f"    - {p.get('name', '?')} ({p.get('in', '?')}): {p.get('description', '')} [{'必填' if p.get('required') else '可选'}]")
            if request_body:
                content = request_body.get('content', {})
                for ct, schema in content.items():
                    lines.append(f"  请求体 ({ct}): {json.dumps(schema.get('schema', {}), ensure_ascii=False)}")
            if responses:
                for code, resp in responses.items():
                    lines.append(f"  响应 {code}: {resp.get('description', '')}")

    return '\n'.join(lines)


def fetch_url_content(url: str, login_url: str = "", login_username: str = "",
                      login_password: str = "", login_username_selector: str = "",
                      login_password_selector: str = "", login_submit_selector: str = "") -> str:
    """抓取网页URL，提取可见文本内容和交互元素描述。支持先登录再抓取。"""
    logger.info(f"🌐 抓取网页内容: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()
        try:
            same_page_login = False
            if login_url and login_username and login_password:
                login_parsed = urlparse(login_url)
                target_parsed = urlparse(url)
                if (login_parsed.netloc == target_parsed.netloc and
                        login_parsed.path.rstrip('/') == target_parsed.path.rstrip('/')):
                    same_page_login = True
                    logger.info(f"   🔐 登录URL与目标URL相同，在同一页面执行登录")
                    page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=10000)
                        except Exception:
                            pass
                    page.wait_for_timeout(2000)

                    _perform_login(page, login_username, login_password,
                                  login_username_selector, login_password_selector,
                                  login_submit_selector)

                    page.wait_for_timeout(3000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                else:
                    logger.info(f"   🔐 先执行登录: {login_url}")
                    page.goto(login_url, timeout=PAGE_LOAD_TIMEOUT)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=10000)
                        except Exception:
                            pass
                    page.wait_for_timeout(2000)

                    _perform_login(page, login_username, login_password,
                                  login_username_selector, login_password_selector,
                                  login_submit_selector)

                    page.wait_for_timeout(3000)

            if not same_page_login:
                page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=10000)
                    except Exception:
                        pass
                page.wait_for_timeout(2000)

                if login_url and login_username and login_password:
                    page_text = ""
                    try:
                        page_text = page.evaluate("() => document.body.innerText.slice(0, 500)")
                    except Exception:
                        pass
                    login_indicators = ['请登录', '还没有登录', '没有访问权限', '请先登录', '登录后']
                    if any(ind in page_text for ind in login_indicators):
                        logger.info("   🔍 目标页面需要登录，在当前页面执行登录...")
                        _perform_login(page, login_username, login_password,
                                      login_username_selector, login_password_selector,
                                      login_submit_selector)
                        page.wait_for_timeout(3000)
                        try:
                            page.wait_for_load_state("networkidle", timeout=10000)
                        except Exception:
                            pass

            result = page.evaluate("""() => {
                const extractText = (el, depth = 0) => {
                    if (depth > 3) return [];
                    const texts = [];
                    const tag = el.tagName ? el.tagName.toLowerCase() : '';

                    if (tag === 'script' || tag === 'style' || tag === 'noscript' || tag === 'svg') return [];

                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return [];

                    for (const child of el.childNodes) {
                        if (child.nodeType === 3) {
                            const t = child.textContent.trim();
                            if (t && t.length > 1) texts.push(t);
                        } else if (child.nodeType === 1) {
                            texts.push(...extractText(child, depth + 1));
                        }
                    }
                    return texts;
                };

                const bodyTexts = extractText(document.body);
                const uniqueTexts = [...new Set(bodyTexts)];

                const buttons = [];
                document.querySelectorAll('button, [role="button"], a.btn, .ant-btn, .el-button').forEach(el => {
                    const t = (el.textContent || '').trim().slice(0, 50);
                    if (t) buttons.push(t);
                });

                const inputs = [];
                document.querySelectorAll('input:not([type="hidden"]), textarea').forEach(el => {
                    const ph = el.placeholder || '';
                    const label = (el.closest('.ant-form-item, .el-form-item, .form-group')?.querySelector('label')?.textContent || '').trim();
                    if (ph || label) inputs.push(label || ph);
                });

                const links = [];
                document.querySelectorAll('a[href]:not([href="#"]):not([href=""])').forEach(el => {
                    const t = (el.textContent || '').trim().slice(0, 50);
                    if (t) links.push(t);
                });

                const menus = [];
                document.querySelectorAll('.ant-menu-item, .el-menu-item, nav a, [class*="sidebar"] a, [class*="menu-item"]').forEach(el => {
                    const t = (el.textContent || '').trim().slice(0, 50);
                    if (t) menus.push(t);
                });

                return {
                    title: document.title,
                    url: window.location.href,
                    visibleTexts: uniqueTexts.slice(0, 300),
                    buttons: [...new Set(buttons)].slice(0, 100),
                    inputs: [...new Set(inputs)].slice(0, 50),
                    links: [...new Set(links)].slice(0, 100),
                    menus: [...new Set(menus)].slice(0, 50),
                };
            }""")

            lines = []
            lines.append(f"页面标题: {result.get('title', '')}")
            lines.append(f"页面URL: {result.get('url', '')}")
            lines.append("")

            if result.get('menus'):
                lines.append("## 导航菜单")
                for m in result['menus']:
                    lines.append(f"- {m}")

            if result.get('buttons'):
                lines.append("\n## 按钮")
                for b in result['buttons']:
                    lines.append(f"- {b}")

            if result.get('inputs'):
                lines.append("\n## 输入框/表单字段")
                for i in result['inputs']:
                    lines.append(f"- {i}")

            if result.get('links'):
                lines.append("\n## 链接")
                for l in result['links']:
                    lines.append(f"- {l}")

            lines.append("\n## 页面可见文本")
            for t in result.get('visibleTexts', []):
                lines.append(f"- {t}")

            output = '\n'.join(lines)
            logger.info(f"   ✅ 提取到 {len(output)} 字符")
            logger.info(f"   📝 内容预览: {output[:300]}...")
            if len(output) < 500:
                logger.warning(f"   ⚠️ 提取内容过少（{len(output)}字符），可能页面需要登录或为空")
            return output

        finally:
            browser.close()


DOCUMENT_PARSE_PROMPT = """
你是一名资深测试专家，需要根据用户提供的需求文档（PRD、用户故事或接口文档），生成**全面、高覆盖率**的自动化测试用例集。

【覆盖要求 —— 必须严格遵守】
1. 每个功能点至少生成：1 条正向用例 + 至少 3 条异常/边界用例
2. 必须覆盖以下所有测试场景类型：
   - ✅ 正常流程（Happy Path）
   - ✅ 异常操作：重复提交、并发冲突、非法操作、权限不足
   - ✅ 边界条件：最大值、最小值、空值、超长文本、临界值
   - ✅ 权限校验：无权限访问、越权操作、未登录访问
3. 优先覆盖：高频操作 > 核心链路 > 高危边界

【用例字段规范】
每个测试用例必须包含以下字段：
- id：用例编号，格式 TC-{模块缩写}-{3位序号}，如 TC-ORDER-001
  模块缩写从文档中推断：订单→ORDER、登录→LOGIN、客户→CUST、商品→PROD、支付→PAY
- name：用例名称，清晰描述测试场景（如"正向-创建销售订单"）
- preconditions：前置条件
- steps：符合Playwright格式的步骤数组（action/selector/value/description）
- expected_result：预期结果
- scenario_type：正向流程 / 异常操作 / 边界条件 / 权限校验

【根据文档类型调整策略】
- PRD/需求文档：提取所有功能点、业务流程、异常路径
- 用户故事：按 Given-When-Then 分解，覆盖 Happy Path + 异常
- 接口文档/Swagger：覆盖正常请求、参数校验、认证鉴权、错误码

【输出要求】
返回纯JSON数组，不要任何额外解释、不要markdown、不要代码块。
"""


@retry_api_call
def ai_generate_cases_from_document(document_text: str, doc_type: str = "prd", url: str = "") -> List[Dict]:
    logger.info(f"📄 从文档生成用例: {doc_type}, 文档长度={len(document_text)}...")
    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": DOCUMENT_PARSE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"文档类型：{doc_type}（prd=需求文档, user_story=用户故事, api_doc=接口文档）\n"
                        f"目标URL（如有）：{url}\n"
                        f"文档内容：\n{document_text[:15000]}"
                    )
                }
            ],
            max_tokens=8000,
            temperature=0.1,
            timeout=AI_REQUEST_TIMEOUT
        )
        ai_text = response.choices[0].message.content.strip()
        test_cases = extract_json_from_response(ai_text)
        if isinstance(test_cases, dict):
            test_cases = test_cases.get("test_cases", [])
        valid_cases = []
        for case in test_cases:
            if not case.get("name"):
                continue
            steps = case.get("steps", [])
            valid_steps = []
            for step in steps:
                if not step.get("action") or not step.get("selector"):
                    continue
                if step.get("action") not in ("fill", "click", "wait_for_selector", "assert_text"):
                    continue
                if step.get("action") in ("fill", "assert_text") and not step.get("value"):
                    continue
                valid_steps.append(step)
            case["steps"] = valid_steps if valid_steps else steps
            case["url"] = url
            case["test_type"] = "web" if url else "api"
            valid_cases.append(case)
        logger.info(f"✅ 从文档生成{len(valid_cases)}个用例")
        return valid_cases
    except Exception as e:
        logger.error(f"❌ 文档解析生成失败: {e}", exc_info=True)
        raise


# ======================== 探索性生成（页面遍历识别交互元素） ========================

def crawl_page_interactive_elements(url: str, login_url: str = "", username: str = "",
                                     password: str = "", username_selector: str = "",
                                     password_selector: str = "", submit_selector: str = "") -> Dict:
    """遍历页面，提取所有可交互元素和页面结构。支持先登录再爬取。"""
    logger.info(f"🕷️ 探索性爬取页面: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()
        try:
            same_page_login = False
            if login_url and username and password:
                login_parsed = urlparse(login_url)
                target_parsed = urlparse(url)
                same_page_login = (login_parsed.netloc == target_parsed.netloc and
                            login_parsed.path.rstrip('/') == target_parsed.path.rstrip('/'))

                if same_page_login:
                    logger.info(f"   🔐 登录URL与目标URL相同，在同一页面执行登录")
                    page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=10000)
                        except Exception:
                            pass
                    page.wait_for_timeout(2000)
                    _perform_login(page, username, password, username_selector,
                                  password_selector, submit_selector)
                    page.wait_for_timeout(3000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                else:
                    logger.info(f"   🔐 先执行登录: {login_url}")
                    page.goto(login_url, timeout=PAGE_LOAD_TIMEOUT)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=10000)
                        except Exception:
                            pass
                    page.wait_for_timeout(2000)
                    _perform_login(page, username, password, username_selector,
                                  password_selector, submit_selector)
                    page.wait_for_timeout(3000)

            if not same_page_login:
                page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=10000)
                    except Exception:
                        pass
                page.wait_for_timeout(2000)

            title = page.title()

            elements_data = page.evaluate("""() => {
                const results = {
                    buttons: [],
                    inputs: [],
                    selects: [],
                    links: [],
                    tabs: [],
                    menus: [],
                    tables: [],
                    forms: [],
                    modals: [],
                    pageTitle: document.title
                };

                document.querySelectorAll('button, [role="button"], a.btn, .ant-btn, .el-button').forEach(el => {
                    const text = (el.textContent || '').trim().slice(0, 50);
                    const id = el.id || '';
                    const cls = (el.className || '').slice(0, 100);
                    if (text && !text.startsWith('<')) {
                        results.buttons.push({text, id, cls, tag: el.tagName});
                    }
                });

                document.querySelectorAll('input:not([type="hidden"]), textarea').forEach(el => {
                    const placeholder = el.placeholder || '';
                    const name = el.name || '';
                    const id = el.id || '';
                    const type = el.type || 'text';
                    const label = (el.closest('.ant-form-item, .el-form-item, .form-group, .field')?.querySelector('label')?.textContent || '').trim();
                    results.inputs.push({placeholder, name, id, type, label});
                });

                document.querySelectorAll('.ant-select, .el-select, select').forEach(el => {
                    const label = (el.closest('.ant-form-item, .el-form-item, .form-group, .field')?.querySelector('label')?.textContent || '').trim();
                    const id = el.id || '';
                    results.selects.push({label, id});
                });

                document.querySelectorAll('a[href]:not([href="#"]):not([href=""]), .ant-menu-item, .el-menu-item, [class*="menu-item"]').forEach(el => {
                    const text = (el.textContent || '').trim().slice(0, 50);
                    const href = el.href || '';
                    if (text) results.links.push({text, href});
                });

                document.querySelectorAll('.ant-tabs-tab, .el-tabs__item, [role="tab"]').forEach(el => {
                    const text = (el.textContent || '').trim().slice(0, 30);
                    if (text) results.tabs.push({text});
                });

                document.querySelectorAll('.ant-menu, .el-menu, [class*="sidebar"], [class*="side-menu"], nav[class*="menu"]').forEach(el => {
                    const items = Array.from(el.querySelectorAll('.ant-menu-item, .el-menu-item, li, [class*="menu-item"]'))
                        .map(i => (i.textContent || '').trim().slice(0, 50))
                        .filter(t => t);
                    if (items.length > 0) results.menus.push({items});
                });

                document.querySelectorAll('table, .ant-table, .el-table').forEach(el => {
                    const headers = Array.from(el.querySelectorAll('th'))
                        .map(th => (th.textContent || '').trim())
                        .filter(t => t);
                    const rowCount = el.querySelectorAll('tbody tr').length;
                    if (headers.length > 0) results.tables.push({headers, rowCount});
                });

                document.querySelectorAll('form, .ant-form, .el-form').forEach(el => {
                    const inputs = el.querySelectorAll('input:not([type="hidden"]), textarea, select').length;
                    const buttons = el.querySelectorAll('button, [role="button"]').length;
                    if (inputs > 0) results.forms.push({inputs, buttons});
                });

                document.querySelectorAll('.ant-modal, .el-dialog, [role="dialog"], .modal').forEach(el => {
                    const visible = el.offsetParent !== null;
                    const title = (el.querySelector('.ant-modal-title, .el-dialog__title, .modal-title')?.textContent || '').trim();
                    if (title) results.modals.push({title, visible});
                });

                return results;
            }""")

            logger.info(f"   📊 发现: {len(elements_data.get('buttons',[]))}按钮, "
                       f"{len(elements_data.get('inputs',[]))}输入框, "
                       f"{len(elements_data.get('selects',[]))}下拉框, "
                       f"{len(elements_data.get('tables',[]))}表格, "
                       f"{len(elements_data.get('forms',[]))}表单, "
                       f"{len(elements_data.get('menus',[]))}菜单")
            return {"url": url, "title": title, "elements": elements_data}
        finally:
            browser.close()


def _find_login_frame(page, require_password=False):
    """查找包含登录表单的iframe，返回frame对象或None。
    require_password=False时只要有输入框即可（因为可能默认是验证码登录，密码框需要切换后才出现）
    """
    try:
        frames = page.frames
        if len(frames) <= 1:
            return None
        for frame in frames[1:]:
            try:
                result = frame.evaluate("""() => {
                    const inputs = document.querySelectorAll('input:not([type="hidden"])');
                    let hasText = false, hasPassword = false, inputCount = 0;
                    inputs.forEach(inp => {
                        const style = window.getComputedStyle(inp);
                        if (style.display === 'none' || style.visibility === 'hidden') return;
                        inputCount++;
                        if (inp.type === 'password') hasPassword = true;
                        if (['text', 'tel', 'email', 'number'].includes(inp.type)) hasText = true;
                    });
                    const bodyText = (document.body.innerText || '').slice(0, 200);
                    const buttons = [];
                    document.querySelectorAll('button, [role="button"]').forEach(el => {
                        const t = (el.textContent || '').trim();
                        if (t) buttons.push(t);
                    });
                    return {hasText, hasPassword, inputCount, bodyText, buttons: buttons.slice(0, 10)};
                }""")
                logger.info(f"   🔍 iframe {frame.url[:60]}: 输入框={result.get('inputCount',0)}, 有文本框={result.get('hasText')}, 有密码框={result.get('hasPassword')}, 按钮={result.get('buttons',[])}, 文本={result.get('bodyText','')[:80]}")

                if require_password:
                    if result.get('hasText') and result.get('hasPassword'):
                        return frame
                else:
                    if result.get('inputCount', 0) > 0:
                        logger.info(f"   ✅ 找到包含输入框的iframe: {frame.url[:80]}")
                        return frame
                    login_keywords = ['登录', '密码', '手机', '账号', 'login', 'password', 'phone']
                    body_text = result.get('bodyText', '').lower()
                    if any(kw in body_text for kw in login_keywords):
                        logger.info(f"   ✅ 找到包含登录关键词的iframe: {frame.url[:80]}")
                        return frame
            except Exception as e:
                logger.info(f"   ⚡ iframe访问失败(可能跨域): {frame.url[:60]} - {e}")
                continue
    except Exception:
        pass
    return None


def _extract_frame_info(frame) -> str:
    """从iframe中提取登录页面信息"""
    try:
        info = frame.evaluate("""() => {
            const results = {
                buttons: [],
                inputs: [],
                checkboxes: [],
                customCheckboxes: [],
                tabs: [],
                links: [],
                clickableSpans: [],
                allText: []
            };

            const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
            };

            document.querySelectorAll('button, [role="button"], a.btn, span[class*="btn"], div[class*="btn"], a[class*="login"], span[class*="login"]').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                if (text && text.length > 1) results.buttons.push({text, tag: el.tagName, cls: (el.className || '').slice(0, 60)});
            });

            document.querySelectorAll('a').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                if (text && text.length > 1) results.links.push({text, href: (el.href || '').slice(0, 100)});
            });

            document.querySelectorAll('span, div').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 60);
                const cls = (el.className || '').toString().toLowerCase();
                const isClickable = el.onclick || cls.includes('click') || cls.includes('tab') ||
                    cls.includes('switch') || cls.includes('toggle') || cls.includes('link') ||
                    cls.includes('btn') || cls.includes('active') || cls.includes('select') ||
                    el.getAttribute('role') === 'tab' || el.getAttribute('role') === 'button' ||
                    cls.includes('mode') || cls.includes('type') || cls.includes('method');
                if (isClickable && text && text.length > 1 && text.length < 30) {
                    const alreadyInButtons = results.buttons.some(b => b.text === text);
                    const alreadyInLinks = results.links.some(l => l.text === text);
                    if (!alreadyInButtons && !alreadyInLinks) {
                        results.clickableSpans.push({text, tag: el.tagName, cls: cls.slice(0, 60)});
                    }
                }
            });

            document.querySelectorAll('input:not([type="hidden"]), textarea').forEach(el => {
                const style = window.getComputedStyle(el);
                const hidden = style.display === 'none' || style.visibility === 'hidden';
                results.inputs.push({
                    placeholder: el.placeholder || '', name: el.name || '',
                    id: el.id || '', type: el.type || 'text', hidden
                });
            });

            document.querySelectorAll('input[type="checkbox"]').forEach(el => {
                const label = (el.closest('label')?.textContent || el.parentElement?.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                results.checkboxes.push({label, checked: el.checked, id: el.id, name: el.name});
            });

            document.querySelectorAll('[class*="checkbox"], [class*="check-box"], [class*="agree"], [class*="protocol"]').forEach(el => {
                if (el.tagName === 'INPUT') return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 100);
                if (text && (text.includes('同意') || text.includes('协议') || text.includes('隐私') || text.includes('agree'))) {
                    results.customCheckboxes.push({text, tag: el.tagName, cls: (el.className || '').slice(0, 80)});
                }
            });

            document.querySelectorAll('.ant-tabs-tab, .el-tabs__item, [role="tab"], .tab-item, [class*="tab"], [class*="Tab"]').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().slice(0, 50);
                if (text && text.length > 1) results.tabs.push({text, cls: (el.className || '').slice(0, 60)});
            });

            const loginMethods = [];
            document.querySelectorAll('.signway-container li, .signway-container a, [class*="signway"] li, [class*="signway"] a, [class*="login-method"] li, [class*="login-method"] a, ul[class*="method"] li, ul[class*="way"] li').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().slice(0, 40);
                if (text && text.length > 1 && text.length < 30) {
                    loginMethods.push({text, tag: el.tagName, cls: (el.className || '').slice(0, 60)});
                }
            });
            if (loginMethods.length > 0) results.loginMethods = loginMethods;

            const bodyText = document.body.innerText || '';
            const texts = bodyText.split('\\n').map(t => t.trim()).filter(t => t.length > 1 && t.length < 200);
            results.allText = [...new Set(texts)].slice(0, 80);

            return results;
        }""")

        lines = []
        lines.append("[iframe中的登录表单]")

        if info.get('tabs'):
            tabs_text = ', '.join([t['text'] for t in info['tabs']])
            lines.append(f"标签页/切换选项: {tabs_text}")

        if info.get('links'):
            lines.append(f"链接 ({len(info['links'])}个):")
            for l in info['links'][:20]:
                lines.append(f"  - {l['text']}")

        if info.get('clickableSpans'):
            lines.append(f"可点击的span/div ({len(info['clickableSpans'])}个):")
            for s in info['clickableSpans'][:20]:
                lines.append(f"  - [{s['tag']}] {s['text']} (class={s.get('cls','')})")

        if info.get('buttons'):
            lines.append(f"可点击按钮 ({len(info['buttons'])}个):")
            for b in info['buttons'][:30]:
                lines.append(f"  - [{b['tag']}] {b['text']}")

        visible_inputs = [i for i in info.get('inputs', []) if not i.get('hidden')]
        if visible_inputs:
            lines.append(f"可见输入框 ({len(visible_inputs)}个):")
            for inp in visible_inputs[:20]:
                parts = [f"type={inp['type']}"]
                if inp.get('placeholder'): parts.append(f"placeholder={inp['placeholder']}")
                if inp.get('name'): parts.append(f"name={inp['name']}")
                if inp.get('id'): parts.append(f"id={inp['id']}")
                lines.append(f"  - {', '.join(parts)}")

        if info.get('checkboxes'):
            lines.append(f"复选框 ({len(info['checkboxes'])}个):")
            for c in info['checkboxes']:
                status = "已勾选" if c['checked'] else "未勾选"
                lines.append(f"  - {c['label']} [{status}]")

        if info.get('customCheckboxes'):
            lines.append(f"自定义复选框/协议 ({len(info['customCheckboxes'])}个):")
            for c in info['customCheckboxes']:
                lines.append(f"  - [{c['tag']}] {c['text']}")

        if info.get('loginMethods'):
            lines.append(f"登录方式选项 ({len(info['loginMethods'])}个):")
            for m in info['loginMethods']:
                lines.append(f"  - [{m['tag']}] {m['text']}")

        if info.get('allText'):
            lines.append(f"页面可见文本片段:")
            for t in info['allText'][:30]:
                lines.append(f"  - {t}")

        return '\n'.join(lines)
    except Exception as e:
        logger.warning(f"   ⚠️ 提取iframe信息失败: {e}")
        return ""


def _perform_login(page, username: str, password: str, username_selector: str = "",
                   password_selector: str = "", submit_selector: str = ""):
    """在页面上执行登录操作，AI智能识别页面内容并自主完成登录流程"""
    logger.info("   🔍 AI智能分析登录页面...")

    if username_selector and password_selector:
        logger.info("   📌 使用用户指定的选择器登录")
        _perform_simple_login(page, username, password, username_selector,
                             password_selector, submit_selector)
        return

    page_info = _extract_login_page_info(page)
    if not page_info:
        logger.warning("   ⚠️ 无法提取页面信息，回退到简单登录")
        _perform_simple_login(page, username, password)
        return

    logger.info(f"   📊 页面信息: {len(page_info)} 字符，发送给AI分析...")

    login_frame = None
    has_login_form = _page_has_login_form(page)
    login_dialog_opened = False

    if not has_login_form:
        login_frame = _find_login_frame(page)
        if login_frame:
            logger.info("   ✅ 检测到登录表单在iframe中")
            has_login_form = True
            login_dialog_opened = True
            frame_info = _extract_frame_info(login_frame)
            if frame_info:
                page_info = frame_info
                page_info += "\n\n【重要提示】登录表单在iframe中，系统会自动在iframe中执行操作。不要再次点击页面上的登录入口按钮，直接在登录表单内操作即可。"
                logger.info(f"   📊 iframe页面信息: {len(page_info)} 字符")
        else:
            logger.info("   🔍 未检测到登录表单，尝试点击登录按钮打开登录弹窗...")
            clicked = _try_click_login_entry(page)
            if clicked:
                page.wait_for_timeout(3000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                login_frame = _find_login_frame(page)
                if login_frame:
                    logger.info("   ✅ 点击后检测到登录表单在iframe中")
                    login_dialog_opened = True
                    frame_info = _extract_frame_info(login_frame)
                    if frame_info:
                        page_info = frame_info
                        page_info += "\n\n【重要提示】登录表单在iframe中，系统会自动在iframe中执行操作。不要再次点击页面上的登录入口按钮，直接在登录表单内操作即可。"
                        logger.info(f"   📊 iframe页面信息: {len(page_info)} 字符")

                if not login_frame:
                    try:
                        page.wait_for_selector('input[type="text"], input[type="tel"], input[type="password"], input[placeholder*="手机"], input[placeholder*="账号"], input[placeholder*="用户"]', timeout=5000)
                        login_dialog_opened = True
                        logger.info("   ✅ 登录弹窗已打开，检测到输入框")
                    except Exception:
                        pass

                    if not login_dialog_opened:
                        login_frame = _find_login_frame(page)
                        if login_frame:
                            login_dialog_opened = True
                            frame_info = _extract_frame_info(login_frame)
                            if frame_info:
                                page_info = frame_info
                                page_info += "\n\n【重要提示】登录表单在iframe中，系统会自动在iframe中执行操作。不要再次点击页面上的登录入口按钮，直接在登录表单内操作即可。"
                        elif _page_has_login_form(page):
                            login_dialog_opened = True
                            logger.info("   ✅ 登录弹窗已打开（延迟检测）")

                    page_info_new = _extract_login_page_info(page)
                    if page_info_new and len(page_info_new) > len(page_info):
                        page_info = page_info_new
                        logger.info(f"   📊 弹窗后页面信息: {len(page_info)} 字符")

                if not login_dialog_opened:
                    logger.info("   🔄 弹窗可能未正确打开，尝试更广泛的选择器...")
                    _try_open_login_dialog_advanced(page)
                    page.wait_for_timeout(3000)
                    login_frame = _find_login_frame(page)
                    if login_frame:
                        login_dialog_opened = True
                        frame_info = _extract_frame_info(login_frame)
                        if frame_info:
                            page_info = frame_info
                            page_info += "\n\n【重要提示】登录表单在iframe中，系统会自动在iframe中执行操作。不要再次点击页面上的登录入口按钮，直接在登录表单内操作即可。"
                    else:
                        page_info_new = _extract_login_page_info(page)
                        if page_info_new and len(page_info_new) > len(page_info):
                            page_info = page_info_new
                        if _page_has_login_form(page):
                            login_dialog_opened = True

    if login_dialog_opened and not login_frame:
        page_info = _extract_login_page_info(page)
        if page_info:
            page_info += "\n\n【重要提示】登录弹窗/对话框已经打开，不要再次点击页面上的登录入口按钮，直接在弹窗内操作即可。"

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": LOGIN_PAGE_ANALYZE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户名: {username}\n密码: {password}\n\n"
                        f"页面结构:\n{page_info}"
                    )
                }
            ],
            max_tokens=2000,
            temperature=0.1,
            timeout=AI_REQUEST_TIMEOUT
        )
        ai_text = response.choices[0].message.content.strip()
        actions = extract_json_from_response(ai_text)
        if isinstance(actions, dict):
            actions = actions.get("actions", actions.get("steps", []))
        if not isinstance(actions, list) or len(actions) == 0:
            logger.warning("   ⚠️ AI未返回有效登录步骤，回退到简单登录")
            _perform_simple_login(page, username, password)
            return

        logger.info(f"   🤖 AI生成 {len(actions)} 个登录步骤")
        for i, action in enumerate(actions):
            _execute_login_action(page, action, i + 1, login_frame)
            page.wait_for_timeout(500)

            if login_frame and action.get('action', action.get('type', '')) == 'click':
                page.wait_for_timeout(1000)
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass
                new_frame = _find_login_frame(page)
                if new_frame:
                    login_frame = new_frame
                    try:
                        new_info = _extract_frame_info(login_frame)
                        if new_info:
                            logger.info(f"   📊 iframe内容已更新: {len(new_info)} 字符")
                    except Exception:
                        pass

            if i < len(actions) - 1:
                try:
                    if login_frame:
                        debug_text = login_frame.evaluate("() => document.body.innerText.slice(0, 200)")
                    else:
                        debug_text = page.evaluate("() => document.body.innerText.slice(0, 200)")
                    logger.info(f"   🔍 步骤{i+1}后页面文本: {debug_text[:100]}...")
                except Exception:
                    pass

            if action.get('action', action.get('type', '')) == 'click':
                page.wait_for_timeout(500)
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass

        page.wait_for_timeout(3000)
        current_url = page.url
        logger.info(f"   🔐 AI登录流程完成，当前页面: {current_url}")

        try:
            debug_info = page.evaluate("""() => {
                const inputs = document.querySelectorAll('input:not([type="hidden"])');
                const inputInfo = [];
                inputs.forEach(inp => {
                    inputInfo.push({type: inp.type, value: inp.value.slice(0,20), placeholder: inp.placeholder});
                });
                const checkboxes = document.querySelectorAll('input[type="checkbox"]');
                const cbInfo = [];
                checkboxes.forEach(cb => { cbInfo.push({checked: cb.checked}); });
                return {inputs: inputInfo, checkboxes: cbInfo, url: window.location.href};
            }""")
            logger.info(f"   🔍 登录后状态: URL={debug_info.get('url','')}, 输入框={debug_info.get('inputs',[])}, 复选框={debug_info.get('checkboxes',[])}")
        except Exception:
            pass

        login_keywords = ['login', 'signin', 'auth', 'passport', 'sso']
        still_on_login = any(kw in current_url.lower() for kw in login_keywords)

        if still_on_login:
            logger.info("   🔄 仍在登录页面，尝试按Enter提交表单...")
            try:
                page.keyboard.press('Enter')
                page.wait_for_timeout(3000)
                current_url = page.url
                still_on_login = any(kw in current_url.lower() for kw in login_keywords)
                if not still_on_login:
                    logger.info(f"   ✅ Enter提交成功，已跳转: {current_url}")
            except Exception:
                pass

        if still_on_login:
            modal_handled = _try_handle_post_login_modal(page)
            if modal_handled:
                page.wait_for_timeout(3000)
                current_url = page.url
                still_on_login = any(kw in current_url.lower() for kw in login_keywords)
                if not still_on_login:
                    logger.info(f"   ✅ 处理弹窗后已跳转: {current_url}")

        if still_on_login:
            logger.warning("   ⚠️ 登录可能未成功，仍在登录页面")
            _close_blocking_modals(page)
            try:
                _perform_simple_login(page, username, password)
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"   ⚠️ AI登录分析失败: {e}，回退到简单登录")
        try:
            _perform_simple_login(page, username, password)
        except Exception:
            pass


def _page_has_login_form(page) -> bool:
    """检测页面是否有可见的登录表单（输入框+密码框）"""
    try:
        result = page.evaluate("""() => {
            const inputs = document.querySelectorAll('input:not([type="hidden"])');
            let hasText = false, hasPassword = false;
            inputs.forEach(inp => {
                const style = window.getComputedStyle(inp);
                if (style.display === 'none' || style.visibility === 'hidden') return;
                if (inp.type === 'password') hasPassword = true;
                if (inp.type === 'text' || inp.type === 'tel' || inp.type === 'email') hasText = true;
            });
            return hasText && hasPassword;
        }""")
        return bool(result)
    except Exception:
        return False


def _try_handle_post_login_modal(page) -> bool:
    """登录后如果出现弹窗（如切换组织、选择角色等），尝试处理"""
    try:
        modal_info = page.evaluate("""() => {
            const modals = document.querySelectorAll('.ant-modal-wrap, .el-dialog__wrapper, [role="dialog"], .ant-modal');
            for (const modal of modals) {
                const style = window.getComputedStyle(modal);
                if (style.display === 'none') continue;
                const title = (modal.querySelector('.ant-modal-title, .el-dialog__title, .modal-title')?.textContent || '').trim();
                const buttons = [];
                modal.querySelectorAll('button, [role="button"], .ant-btn').forEach(btn => {
                    const t = (btn.textContent || '').trim();
                    if (t) buttons.push(t);
                });
                const text = (modal.textContent || '').trim().slice(0, 500);
                return {title, buttons, text: text.slice(0, 300)};
            }
            return null;
        }""")

        if not modal_info:
            return False

        logger.info(f"   🔍 检测到弹窗: 标题={modal_info.get('title','')}, 按钮={modal_info.get('buttons',[])}")

        confirm_keywords = ['确定', '确认', '进入', '选择', '继续', 'OK', 'Confirm', 'Enter', 'ok']
        close_keywords = ['关闭', '取消', '跳过', 'Close', 'Cancel', 'Skip']

        for keyword in confirm_keywords:
            for btn_text in modal_info.get('buttons', []):
                if keyword in btn_text:
                    try:
                        btn = page.locator(f'.ant-modal-wrap button:has-text("{btn_text}"), .ant-modal button:has-text("{btn_text}")').first
                        if btn.count() > 0:
                            btn.click()
                            page.wait_for_timeout(1000)
                            logger.info(f"   ✅ 点击弹窗按钮: {btn_text}")
                            return True
                    except Exception:
                        continue

        try:
            first_btn = page.locator('.ant-modal-wrap .ant-btn-primary:visible, .ant-modal .ant-btn-primary:visible').first
            if first_btn.count() > 0:
                first_btn.click()
                page.wait_for_timeout(1000)
                logger.info("   ✅ 点击弹窗主按钮")
                return True
        except Exception:
            pass

        try:
            first_item = page.locator('.ant-modal-wrap .ant-list-item:visible, .ant-modal-wrap .ant-card:visible, .ant-modal-wrap [class*="org"]:visible, .ant-modal-wrap [class*="item"]:visible').first
            if first_item.count() > 0:
                first_item.click()
                page.wait_for_timeout(500)
                logger.info("   ✅ 点击弹窗列表第一项")
                try:
                    confirm_btn = page.locator('.ant-modal-wrap .ant-btn-primary:visible').first
                    if confirm_btn.count() > 0:
                        confirm_btn.click()
                        page.wait_for_timeout(1000)
                except Exception:
                    pass
                return True
        except Exception:
            pass

        return False
    except Exception as e:
        logger.info(f"   ⚡ 处理弹窗异常: {e}")
        return False


def _close_blocking_modals(page):
    """关闭阻挡操作的弹窗"""
    try:
        close_selectors = [
            '.ant-modal-close',
            '.ant-modal-wrap .ant-modal-close',
            '.el-dialog__close',
            '[class*="modal"] [class*="close"]',
            'button[aria-label="Close"]',
            '.ant-modal-wrap .ant-modal-close-x',
        ]
        for sel in close_selectors:
            try:
                btn = page.locator(sel).first
                if btn.count() > 0:
                    btn.click()
                    page.wait_for_timeout(500)
                    logger.info(f"   ✅ 关闭阻挡弹窗: {sel}")
                    return
            except Exception:
                continue

        try:
            page.keyboard.press('Escape')
            page.wait_for_timeout(500)
            logger.info("   ✅ 按Escape关闭弹窗")
        except Exception:
            pass
    except Exception:
        pass


def _try_click_login_entry(page) -> bool:
    """尝试点击页面上的登录入口按钮（打开登录弹窗）"""
    entry_selectors = [
        'a:has-text("登录")',
        'button:has-text("登录")',
        'text=登录',
        'span:has-text("登录")',
        'div:has-text("登录")',
        '[class*="login"]',
        'a:has-text("Login")',
        'button:has-text("Login")',
    ]
    for sel in entry_selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.click()
                page.wait_for_timeout(1000)
                logger.info(f"   ✅ 点击登录入口: {sel}")
                return True
        except Exception:
            continue
    logger.warning("   ⚠️ 未找到登录入口按钮")
    return False


def _try_open_login_dialog_advanced(page):
    """更高级的登录弹窗打开策略：尝试点击更广泛的选择器，处理iframe等"""
    advanced_selectors = [
        '[class*="signin"]', '[class*="sign-in"]', '[class*="login-btn"]',
        '[class*="loginBtn"]', '[class*="login_button"]',
        'a[href*="login"]', 'a[href*="signin"]',
        '[data-action="login"]', '[data-type="login"]',
    ]
    for sel in advanced_selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.click()
                page.wait_for_timeout(1500)
                logger.info(f"   ✅ 高级选择器点击: {sel}")
                return
        except Exception:
            continue

    try:
        frames = page.frames
        if len(frames) > 1:
            logger.info(f"   🔍 检测到 {len(frames)} 个iframe，尝试在iframe中查找登录入口")
            for frame in frames[1:]:
                try:
                    login_link = frame.locator('a:has-text("登录"), button:has-text("登录")').first
                    if login_link.count() > 0:
                        login_link.click()
                        page.wait_for_timeout(2000)
                        logger.info("   ✅ 在iframe中点击登录入口")
                        return
                except Exception:
                    continue
    except Exception:
        pass


def _extract_login_page_info(page) -> str:
    """提取登录页面的结构化信息，供AI分析。包括弹窗/对话框内容"""
    try:
        info = page.evaluate("""() => {
            const results = {
                title: document.title,
                url: window.location.href,
                buttons: [],
                links: [],
                inputs: [],
                checkboxes: [],
                customCheckboxes: [],
                tabs: [],
                modals: [],
                allText: []
            };

            const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
            };

            document.querySelectorAll('.ant-modal, .el-dialog, [role="dialog"], .modal, [class*="dialog"], [class*="popup"], [class*="overlay"], [class*="Drawer"], [class*="drawer"], [class*="sheet"], [class*="lightbox"], [class*="mask"]').forEach(el => {
                if (!isVisible(el)) return;
                const title = (el.querySelector('.ant-modal-title, .el-dialog__title, .modal-title, [class*="title"], h1, h2, h3')?.textContent || '').trim();
                const text = (el.textContent || '').trim().slice(0, 800);
                results.modals.push({title, text: text.slice(0, 500)});
            });

            document.querySelectorAll('[class*="login"], [class*="signin"], [class*="sign-in"], [id*="login"], [id*="signin"]').forEach(el => {
                if (!isVisible(el)) return;
                if (results.modals.some(m => m.text.includes((el.textContent || '').trim().slice(0, 50)))) return;
                const text = (el.textContent || '').trim().slice(0, 800);
                if (text.length > 20 && (text.includes('密码') || text.includes('手机') || text.includes('账号') || text.includes('登录') || text.includes('password') || text.includes('phone'))) {
                    results.modals.push({title: '登录区域', text: text.slice(0, 500)});
                }
            });

            document.querySelectorAll('button, [role="button"], a.btn, .ant-btn, .el-button, span[class*="btn"], div[class*="btn"], a[class*="login"], span[class*="login"]').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                const cls = (el.className || '').slice(0, 100);
                if (text && text.length > 1) {
                    results.buttons.push({text, cls, tag: el.tagName});
                }
            });

            document.querySelectorAll('a[href]:not([href="#"]):not([href=""])').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                if (text && text.length > 1) {
                    results.links.push({text, href: el.href});
                }
            });

            document.querySelectorAll('input:not([type="hidden"]), textarea').forEach(el => {
                const style = window.getComputedStyle(el);
                const visuallyHidden = style.display === 'none' || style.visibility === 'hidden';
                const placeholder = el.placeholder || '';
                const name = el.name || '';
                const id = el.id || '';
                const type = el.type || 'text';
                const label = (el.closest('.ant-form-item, .el-form-item, .form-group, .field, label')?.querySelector('label')?.textContent || '').trim();
                results.inputs.push({placeholder, name, id, type, label, hidden: visuallyHidden});
            });

            document.querySelectorAll('input[type="checkbox"]').forEach(el => {
                const label = (el.closest('label')?.textContent || el.parentElement?.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                const checked = el.checked;
                results.checkboxes.push({label, checked, id: el.id, name: el.name});
            });

            document.querySelectorAll('[class*="checkbox"], [class*="check-box"], [class*="agree"], [class*="protocol"]').forEach(el => {
                if (el.tagName === 'INPUT') return;
                const text = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 100);
                if (text && (text.includes('同意') || text.includes('协议') || text.includes('隐私') || text.includes('agree'))) {
                    results.customCheckboxes.push({text, tag: el.tagName, cls: (el.className || '').slice(0, 80)});
                }
            });

            document.querySelectorAll('.ant-tabs-tab, .el-tabs__item, [role="tab"], .tab-item, [class*="tab"], span[class*="tab"], div[class*="tab"]').forEach(el => {
                if (!isVisible(el)) return;
                const text = (el.textContent || '').trim().slice(0, 50);
                if (text && text.length > 1) results.tabs.push({text});
            });

            const bodyText = document.body.innerText || '';
            const texts = bodyText.split('\\n').map(t => t.trim()).filter(t => t.length > 1 && t.length < 200);
            results.allText = [...new Set(texts)].slice(0, 100);

            return results;
        }""")

        lines = []
        lines.append(f"页面标题: {info.get('title', '')}")
        lines.append(f"当前URL: {info.get('url', '')}")

        if info.get('modals'):
            lines.append(f"\n弹窗/对话框 ({len(info['modals'])}个):")
            for m in info['modals']:
                lines.append(f"  - 标题: {m.get('title', '无')}")
                lines.append(f"    内容: {m.get('text', '')[:200]}")

        if info.get('tabs'):
            tabs_text = ', '.join([t['text'] for t in info['tabs']])
            lines.append(f"\n标签页/切换选项: {tabs_text}")

        if info.get('buttons'):
            lines.append(f"\n可点击按钮 ({len(info['buttons'])}个):")
            for b in info['buttons'][:30]:
                lines.append(f"  - [{b['tag']}] {b['text']}")

        if info.get('links'):
            lines.append(f"\n链接 ({len(info['links'])}个):")
            for l in info['links'][:20]:
                lines.append(f"  - {l['text']}")

        if info.get('inputs'):
            visible_inputs = [i for i in info['inputs'] if not i.get('hidden')]
            hidden_inputs = [i for i in info['inputs'] if i.get('hidden')]
            if visible_inputs:
                lines.append(f"\n可见输入框 ({len(visible_inputs)}个):")
                for inp in visible_inputs[:20]:
                    parts = [f"type={inp['type']}"]
                    if inp.get('placeholder'): parts.append(f"placeholder={inp['placeholder']}")
                    if inp.get('name'): parts.append(f"name={inp['name']}")
                    if inp.get('id'): parts.append(f"id={inp['id']}")
                    if inp.get('label'): parts.append(f"label={inp['label']}")
                    lines.append(f"  - {', '.join(parts)}")
            if hidden_inputs:
                lines.append(f"\n隐藏输入框 ({len(hidden_inputs)}个):")
                for inp in hidden_inputs[:10]:
                    parts = [f"type={inp['type']}"]
                    if inp.get('placeholder'): parts.append(f"placeholder={inp['placeholder']}")
                    if inp.get('id'): parts.append(f"id={inp['id']}")
                    lines.append(f"  - {', '.join(parts)}")

        if info.get('checkboxes'):
            lines.append(f"\n复选框 ({len(info['checkboxes'])}个):")
            for c in info['checkboxes']:
                status = "已勾选" if c['checked'] else "未勾选"
                lines.append(f"  - {c['label']} [{status}]")

        if info.get('customCheckboxes'):
            lines.append(f"\n自定义复选框/协议 ({len(info['customCheckboxes'])}个):")
            for c in info['customCheckboxes']:
                lines.append(f"  - [{c['tag']}] {c['text']}")

        if info.get('allText'):
            lines.append(f"\n页面可见文本片段:")
            for t in info['allText'][:50]:
                lines.append(f"  - {t}")

        try:
            frames = page.frames
            if len(frames) > 1:
                for frame in frames[1:]:
                    try:
                        frame_inputs = frame.evaluate("""() => {
                            const inputs = [];
                            document.querySelectorAll('input:not([type="hidden"]), textarea').forEach(el => {
                                const style = window.getComputedStyle(el);
                                if (style.display === 'none') return;
                                inputs.push({placeholder: el.placeholder || '', name: el.name || '', id: el.id || '', type: el.type || 'text'});
                            });
                            const buttons = [];
                            document.querySelectorAll('button, [role="button"]').forEach(el => {
                                const t = (el.textContent || '').trim();
                                if (t) buttons.push(t);
                            });
                            const text = (document.body.innerText || '').slice(0, 500);
                            return {inputs, buttons, text};
                        }""")
                        if frame_inputs.get('inputs') or frame_inputs.get('buttons'):
                            lines.append(f"\niframe内容 ({frame.url[:80]}):")
                            if frame_inputs.get('inputs'):
                                lines.append(f"  输入框: {frame_inputs['inputs'][:10]}")
                            if frame_inputs.get('buttons'):
                                lines.append(f"  按钮: {frame_inputs['buttons'][:10]}")
                            if frame_inputs.get('text'):
                                lines.append(f"  文本: {frame_inputs['text'][:200]}")
                    except Exception:
                        continue
        except Exception:
            pass

        return '\n'.join(lines)

    except Exception as e:
        logger.warning(f"   ⚠️ 提取页面信息失败: {e}")
        return ""


def _execute_login_action(page, action: dict, step_num: int, login_frame=None):
    """执行AI返回的单个登录步骤，带智能回退。支持在iframe中执行"""
    action_type = action.get('action', action.get('type', 'click'))
    target = action.get('target', action.get('selector', ''))
    value = action.get('value', '')
    desc = action.get('description', f'{action_type} {target}')

    logger.info(f"   📍 步骤{step_num}: {desc} ({action_type}: {target})" + (f" = {value}" if value else ""))

    if not target and action_type not in ('wait', 'press', 'wait_for_navigation'):
        logger.warning(f"   ⚠️ 步骤{step_num} 缺少目标，跳过")
        return

    try:
        if action_type == 'click':
            _try_click_with_fallback(page, target, step_num, login_frame)

        elif action_type == 'fill':
            _try_fill_with_fallback(page, target, value, step_num, login_frame)

        elif action_type == 'check':
            _try_check_with_fallback(page, target, step_num, login_frame)

        elif action_type == 'wait':
            page.wait_for_timeout(int(value) if value else 1000)

        elif action_type == 'press':
            if login_frame:
                login_frame.evaluate("() => document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter'}))")
            page.keyboard.press(value or 'Enter')
            page.wait_for_timeout(500)

        elif action_type == 'wait_for_navigation':
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(2000)

        else:
            logger.warning(f"   ⚠️ 未知操作类型: {action_type}")

    except Exception as e:
        logger.warning(f"   ⚠️ 步骤{step_num}执行异常: {e}")


def _try_click_with_fallback(page, target: str, step_num: int, login_frame=None):
    """点击操作，带多级回退策略。支持在iframe中查找"""
    import re
    candidates = [target]

    if 'has-text' in target:
        m = re.search(r'has-text\("([^"]*)"\)', target)
        if m:
            text = m.group(1)
            clean = text.replace(' ', '')
            if clean != text:
                candidates.append(target.replace(f'has-text("{text}")', f'has-text("{clean}")'))
            clean2 = re.sub(r'\s+', ' ', text).strip()
            if clean2 != text:
                candidates.append(target.replace(f'has-text("{text}")', f'has-text("{clean2}")'))
            if '/' in text:
                parts = [p.strip() for p in text.split('/') if p.strip()]
                for part in parts:
                    candidates.append(f'button:has-text("{part}")')
                    candidates.append(f'text="{part}"')
            candidates.append(f'text="{text}"')
            candidates.append(f'text="{clean}"')
            if text in ('登录', '登 录', 'Login', 'Sign in'):
                candidates.append(f'[type="submit"]')
                candidates.append(f'.ant-btn:has-text("{text}")')
                candidates.append(f'a:has-text("{text}")')
                candidates.append(f'span:has-text("{text}")')
                candidates.append(f'[class*="login"]:has-text("{text}")')
                candidates.append(f'[class*="submit"]:has-text("{text}")')
                candidates.append(f'button:has-text("登录/注册")')
                candidates.append(f'button:has-text("登 录/注 册")')
                candidates.append(f'text="登录/注册"')
                candidates.append(f'text="登 录/注 册"')
                candidates.append(f'button:has-text("立即登录/注册")')
                candidates.append(f'text="立即登录/注册"')

    if 'text=' in target and 'text=/' not in target:
        m = re.search(r'text="([^"]*)"', target)
        if m:
            text = m.group(1)
            clean = text.replace(' ', '')
            if clean != text:
                candidates.append(f'text="{clean}"')
            if text in ('登录', '登 录', 'Login', 'Sign in'):
                candidates.append(f'button:has-text("{text}")')
                candidates.append(f'button:has-text("登录/注册")')
                candidates.append(f'button:has-text("登 录/注 册")')
                candidates.append(f'text="登录/注册"')
                candidates.append(f'text="登 录/注 册"')
                candidates.append(f'button:has-text("立即登录/注册")')
                candidates.append(f'text="立即登录/注册"')
                candidates.append(f'[type="submit"]')
            tab_keywords = ['密码登录', '短信登录', '验证码登录', '账号登录', '扫码登录',
                          '手机登录', '邮箱登录', '微信登录', '其他登录',
                          'password', 'sms', 'phone', 'email', 'wechat']
            if any(kw in text.lower() for kw in tab_keywords):
                candidates.append(f'span:has-text("{text}")')
                candidates.append(f'div:has-text("{text}")')
                candidates.append(f'a:has-text("{text}")')
                candidates.append(f'[class*="tab"]:has-text("{text}")')
                candidates.append(f'[class*="switch"]:has-text("{text}")')
                candidates.append(f'[class*="mode"]:has-text("{text}")')
                candidates.append(f'[class*="type"]:has-text("{text}")')
                candidates.append(f'[role="tab"]:has-text("{text}")')
                candidates.append(f'[class*="item"]:has-text("{text}")')

    if 'placeholder' in target or 'input' in target.lower():
        pass

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if login_frame:
        for i, sel in enumerate(unique):
            try:
                loc = login_frame.locator(sel).first
                count = loc.count()
                if count > 0:
                    logger.info(f"   ✅ 在iframe中找到元素: {sel}")
                    clicked = False
                    try:
                        old_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                    except Exception:
                        old_text = ""
                    try:
                        loc.click()
                        clicked = True
                    except Exception as click_err:
                        if 'intercepts pointer events' in str(click_err):
                            logger.info(f"   🔄 iframe元素被遮挡，尝试force click: {sel}")
                            try:
                                loc.click(force=True)
                                clicked = True
                            except Exception:
                                pass

                    if clicked:
                        page.wait_for_timeout(1500)
                        try:
                            page.wait_for_load_state("networkidle", timeout=3000)
                        except Exception:
                            pass

                        try:
                            login_frame.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            pass

                        try:
                            new_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                        except Exception:
                            new_text = ""
                        if new_text and new_text != old_text:
                            logger.info(f"   ✅ iframe内容已通过click更新")
                            return
                        try:
                            input_count = login_frame.evaluate("""() => {
                                const inputs = document.querySelectorAll('input:not([type="hidden"])');
                                let count = 0;
                                inputs.forEach(inp => {
                                    const style = window.getComputedStyle(inp);
                                    if (style.display !== 'none' && style.visibility !== 'hidden') count++;
                                });
                                return count;
                            }""")
                            if input_count > 0:
                                logger.info(f"   ✅ iframe中检测到 {input_count} 个可见输入框，内容已更新")
                                return
                        except Exception:
                            pass
                        logger.info(f"   🔄 iframe内容未变化，尝试JS click...")

                    try:
                        loc.evaluate("el => { el.click(); el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, composed: true})); el.dispatchEvent(new PointerEvent('click', {bubbles: true, cancelable: true, composed: true})); }")
                        page.wait_for_timeout(1500)
                        try:
                            login_frame.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            pass
                        new_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                        if new_text and new_text != old_text:
                            logger.info(f"   ✅ iframe内容已通过JS click更新")
                            return
                    except Exception:
                        pass

                    logger.info(f"   🔄 Playwright click无效，使用evaluate遍历祖先+多种事件类型+React Fiber...")
                    click_result = login_frame.evaluate("""(selector) => {
                        function findElement(sel) {
                            if (sel.startsWith('text=')) {
                                const text = sel.slice(5).replace(/^["']|["']$/g, '');
                                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                                let node;
                                while (node = walker.nextNode()) {
                                    if (node.textContent.trim().includes(text)) {
                                        return node.parentElement;
                                    }
                                }
                                return null;
                            }
                            try { return document.querySelector(sel); } catch(e) { return null; }
                        }
                        function isVisible(el) {
                            const style = window.getComputedStyle(el);
                            return style.display !== 'none' && style.visibility !== 'hidden';
                        }
                        const el = findElement(selector);
                        if (!el) return {found: false, reason: 'element not found'};
                        const info = {
                            found: true,
                            tag: el.tagName,
                            className: (el.className || '').toString(),
                            id: el.id || '',
                            href: el.href || '',
                            outerHTML: (el.outerHTML || '').slice(0, 300),
                            parentTag: el.parentElement?.tagName || '',
                            parentClass: (el.parentElement?.className || '').toString(),
                            grandParentTag: el.parentElement?.parentElement?.tagName || '',
                            hasReactFiber: false,
                            clickHandlers: [],
                            isDivider: (el.tagName === 'SPAN' && (el.className || '').toString().includes('divider')),
                            siblingsClicked: []
                        };
                        try {
                            const fiberKey = Object.keys(el).find(k => k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'));
                            if (fiberKey) {
                                info.hasReactFiber = true;
                                let fiber = el[fiberKey];
                                let depth = 0;
                                while (fiber && depth < 10) {
                                    const props = fiber.memoizedProps || fiber.pendingProps || {};
                                    if (props.onClick) info.clickHandlers.push('onClick');
                                    if (props.onPointerDown) info.clickHandlers.push('onPointerDown');
                                    if (props.onMouseDown) info.clickHandlers.push('onMouseDown');
                                    if (props.onTouchEnd) info.clickHandlers.push('onTouchEnd');
                                    if (info.clickHandlers.length > 0) break;
                                    fiber = fiber.return;
                                    depth++;
                                }
                            }
                        } catch(e) {}
                        let current = el;
                        for (let i = 0; i < 6 && current && current !== document.body; i++) {
                            const evtInit = {bubbles: true, cancelable: true, composed: true, view: window};
                            ['click', 'pointerdown', 'pointerup', 'mousedown', 'mouseup', 'touchend'].forEach(type => {
                                try {
                                    current.dispatchEvent(new MouseEvent(type, evtInit));
                                    current.dispatchEvent(new PointerEvent(type, evtInit));
                                } catch(e) {}
                            });
                            try { current.click(); } catch(e) {}
                            current = current.parentElement;
                        }
                        if (info.isDivider && info.clickHandlers.length === 0) {
                            let container = el.parentElement;
                            for (let i = 0; i < 3 && container; i++) {
                                const children = container.children;
                                for (let j = 0; j < children.length; j++) {
                                    const child = children[j];
                                    if (child === el) continue;
                                    if (!isVisible(child)) continue;
                                    const childTag = child.tagName;
                                    const childCls = (child.className || '').toString();
                                    const childText = (child.textContent || '').trim().slice(0, 40);
                                    if (childTag === 'LI' || childTag === 'A' || childTag === 'BUTTON' ||
                                        childCls.includes('item') || childCls.includes('option') || childCls.includes('method') ||
                                        childCls.includes('login') || child.onclick) {
                                        try {
                                            child.click();
                                            child.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, composed: true, view: window}));
                                            child.dispatchEvent(new PointerEvent('click', {bubbles: true, cancelable: true, composed: true, view: window}));
                                        } catch(e) {}
                                        info.siblingsClicked.push({tag: childTag, cls: childCls.slice(0, 60), text: childText});
                                    }
                                }
                                container = container.parentElement;
                            }
                        }
                        return info;
                    }""", sel)
                    logger.info(f"   📋 元素结构: {click_result}")
                    if isinstance(click_result, dict) and click_result.get('siblingsClicked'):
                        logger.info(f"   🔄 检测到divider，已点击 {len(click_result['siblingsClicked'])} 个兄弟元素: {click_result['siblingsClicked']}")
                    page.wait_for_timeout(2000)
                    try:
                        login_frame.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                    try:
                        new_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                        if new_text and new_text != old_text[:500]:
                            logger.info(f"   ✅ iframe内容已通过evaluate点击更新")
                            return
                    except Exception:
                        pass
                    try:
                        input_count = login_frame.evaluate("""() => {
                            const inputs = document.querySelectorAll('input:not([type="hidden"])');
                            let count = 0;
                            inputs.forEach(inp => {
                                const style = window.getComputedStyle(inp);
                                if (style.display !== 'none' && style.visibility !== 'hidden') count++;
                            });
                            return count;
                        }""")
                        if input_count > 0:
                            logger.info(f"   ✅ iframe中检测到 {input_count} 个可见输入框，内容已更新")
                            return
                    except Exception:
                        pass

                    try:
                        href = loc.evaluate("el => el.href || el.closest('a')?.href || ''")
                        if href:
                            logger.info(f"   🔄 尝试直接导航iframe到: {href[:100]}")
                            login_frame.goto(href, timeout=15000)
                            try:
                                login_frame.wait_for_load_state("domcontentloaded", timeout=10000)
                            except Exception:
                                pass
                            page.wait_for_timeout(1000)
                            return
                    except Exception:
                        pass

                    try:
                        current_url = login_frame.evaluate("() => window.location.href")
                        parsed = urlparse(current_url)
                        params = parse_qs(parsed.query)
                        for alt_type in ['phone_login', 'password_login', 'email_login', 'sms_login']:
                            params['type'] = [alt_type]
                            new_query = urlencode(params, doseq=True)
                            alt_url = urlunparse(parsed._replace(query=new_query))
                            logger.info(f"   🔄 尝试导航iframe到: {alt_url[:100]}")
                            try:
                                login_frame.goto(alt_url, timeout=10000)
                                try:
                                    login_frame.wait_for_load_state("domcontentloaded", timeout=8000)
                                except Exception:
                                    pass
                                page.wait_for_timeout(1000)
                                return
                            except Exception:
                                continue
                    except Exception:
                        pass

                    return
            except Exception:
                continue

    for i, sel in enumerate(unique):
        try:
            loc = page.locator(sel).first
            count = loc.count()
            if count > 0:
                if i > 0:
                    logger.info(f"   🔄 回退选择器成功: {sel}")
                try:
                    loc.click()
                except Exception as click_err:
                    if 'intercepts pointer events' in str(click_err):
                        logger.info(f"   🔄 元素被遮挡，尝试force click: {sel}")
                        loc.click(force=True)
                page.wait_for_timeout(800)
                return
        except Exception as e:
            logger.info(f"   ⚡ 选择器 {sel} 失败: {e}")
            continue

    if login_frame:
        logger.info(f"   🔄 主页面未找到元素，尝试在iframe中查找...")
        for i, sel in enumerate(unique):
            try:
                loc = login_frame.locator(sel).first
                count = loc.count()
                if count > 0:
                    logger.info(f"   ✅ 在iframe中找到元素: {sel}")
                    clicked = False
                    try:
                        old_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                    except Exception:
                        old_text = ""
                    try:
                        loc.click()
                        clicked = True
                    except Exception as click_err:
                        if 'intercepts pointer events' in str(click_err):
                            logger.info(f"   🔄 iframe元素被遮挡，尝试force click: {sel}")
                            try:
                                loc.click(force=True)
                                clicked = True
                            except Exception:
                                pass

                    if clicked:
                        page.wait_for_timeout(1500)
                        try:
                            page.wait_for_load_state("networkidle", timeout=3000)
                        except Exception:
                            pass

                        try:
                            login_frame.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            pass

                        try:
                            new_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                        except Exception:
                            new_text = ""
                        if new_text and new_text != old_text:
                            logger.info(f"   ✅ iframe内容已通过click更新")
                            return
                        try:
                            input_count = login_frame.evaluate("""() => {
                                const inputs = document.querySelectorAll('input:not([type="hidden"])');
                                let count = 0;
                                inputs.forEach(inp => {
                                    const style = window.getComputedStyle(inp);
                                    if (style.display !== 'none' && style.visibility !== 'hidden') count++;
                                });
                                return count;
                            }""")
                            if input_count > 0:
                                logger.info(f"   ✅ iframe中检测到 {input_count} 个可见输入框，内容已更新")
                                return
                        except Exception:
                            pass
                        logger.info(f"   🔄 iframe内容未变化，尝试JS click...")

                    try:
                        loc.evaluate("el => { el.click(); el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, composed: true})); el.dispatchEvent(new PointerEvent('click', {bubbles: true, cancelable: true, composed: true})); }")
                        page.wait_for_timeout(1500)
                        try:
                            login_frame.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            pass
                        new_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                        if new_text and new_text != old_text:
                            logger.info(f"   ✅ iframe内容已通过JS click更新")
                            return
                    except Exception:
                        pass

                    logger.info(f"   🔄 Playwright click无效，使用evaluate遍历祖先+多种事件类型+React Fiber...")
                    click_result = login_frame.evaluate("""(selector) => {
                        function findElement(sel) {
                            if (sel.startsWith('text=')) {
                                const text = sel.slice(5).replace(/^["']|["']$/g, '');
                                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                                let node;
                                while (node = walker.nextNode()) {
                                    if (node.textContent.trim().includes(text)) {
                                        return node.parentElement;
                                    }
                                }
                                return null;
                            }
                            try { return document.querySelector(sel); } catch(e) { return null; }
                        }
                        function isVisible(el) {
                            const style = window.getComputedStyle(el);
                            return style.display !== 'none' && style.visibility !== 'hidden';
                        }
                        const el = findElement(selector);
                        if (!el) return {found: false, reason: 'element not found'};
                        const info = {
                            found: true,
                            tag: el.tagName,
                            className: (el.className || '').toString(),
                            id: el.id || '',
                            href: el.href || '',
                            outerHTML: (el.outerHTML || '').slice(0, 300),
                            parentTag: el.parentElement?.tagName || '',
                            parentClass: (el.parentElement?.className || '').toString(),
                            grandParentTag: el.parentElement?.parentElement?.tagName || '',
                            hasReactFiber: false,
                            clickHandlers: [],
                            isDivider: (el.tagName === 'SPAN' && (el.className || '').toString().includes('divider')),
                            siblingsClicked: []
                        };
                        try {
                            const fiberKey = Object.keys(el).find(k => k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'));
                            if (fiberKey) {
                                info.hasReactFiber = true;
                                let fiber = el[fiberKey];
                                let depth = 0;
                                while (fiber && depth < 10) {
                                    const props = fiber.memoizedProps || fiber.pendingProps || {};
                                    if (props.onClick) info.clickHandlers.push('onClick');
                                    if (props.onPointerDown) info.clickHandlers.push('onPointerDown');
                                    if (props.onMouseDown) info.clickHandlers.push('onMouseDown');
                                    if (props.onTouchEnd) info.clickHandlers.push('onTouchEnd');
                                    if (info.clickHandlers.length > 0) break;
                                    fiber = fiber.return;
                                    depth++;
                                }
                            }
                        } catch(e) {}
                        let current = el;
                        for (let i = 0; i < 6 && current && current !== document.body; i++) {
                            const evtInit = {bubbles: true, cancelable: true, composed: true, view: window};
                            ['click', 'pointerdown', 'pointerup', 'mousedown', 'mouseup', 'touchend'].forEach(type => {
                                try {
                                    current.dispatchEvent(new MouseEvent(type, evtInit));
                                    current.dispatchEvent(new PointerEvent(type, evtInit));
                                } catch(e) {}
                            });
                            try { current.click(); } catch(e) {}
                            current = current.parentElement;
                        }
                        if (info.isDivider && info.clickHandlers.length === 0) {
                            let container = el.parentElement;
                            for (let i = 0; i < 3 && container; i++) {
                                const children = container.children;
                                for (let j = 0; j < children.length; j++) {
                                    const child = children[j];
                                    if (child === el) continue;
                                    if (!isVisible(child)) continue;
                                    const childTag = child.tagName;
                                    const childCls = (child.className || '').toString();
                                    const childText = (child.textContent || '').trim().slice(0, 40);
                                    if (childTag === 'LI' || childTag === 'A' || childTag === 'BUTTON' ||
                                        childCls.includes('item') || childCls.includes('option') || childCls.includes('method') ||
                                        childCls.includes('login') || child.onclick) {
                                        try {
                                            child.click();
                                            child.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, composed: true, view: window}));
                                            child.dispatchEvent(new PointerEvent('click', {bubbles: true, cancelable: true, composed: true, view: window}));
                                        } catch(e) {}
                                        info.siblingsClicked.push({tag: childTag, cls: childCls.slice(0, 60), text: childText});
                                    }
                                }
                                container = container.parentElement;
                            }
                        }
                        return info;
                    }""", sel)
                    logger.info(f"   📋 元素结构: {click_result}")
                    if isinstance(click_result, dict) and click_result.get('siblingsClicked'):
                        logger.info(f"   🔄 检测到divider，已点击 {len(click_result['siblingsClicked'])} 个兄弟元素: {click_result['siblingsClicked']}")
                    page.wait_for_timeout(2000)
                    try:
                        login_frame.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                    try:
                        new_text = login_frame.evaluate("() => document.body.innerText.slice(0, 500)")
                        if new_text and new_text != old_text[:500]:
                            logger.info(f"   ✅ iframe内容已通过evaluate点击更新")
                            return
                    except Exception:
                        pass
                    try:
                        input_count = login_frame.evaluate("""() => {
                            const inputs = document.querySelectorAll('input:not([type="hidden"])');
                            let count = 0;
                            inputs.forEach(inp => {
                                const style = window.getComputedStyle(inp);
                                if (style.display !== 'none' && style.visibility !== 'hidden') count++;
                            });
                            return count;
                        }""")
                        if input_count > 0:
                            logger.info(f"   ✅ iframe中检测到 {input_count} 个可见输入框，内容已更新")
                            return
                    except Exception:
                        pass

                    try:
                        href = loc.evaluate("el => el.href || el.closest('a')?.href || ''")
                        if href:
                            logger.info(f"   🔄 尝试直接导航iframe到: {href[:100]}")
                            login_frame.goto(href, timeout=15000)
                            try:
                                login_frame.wait_for_load_state("domcontentloaded", timeout=10000)
                            except Exception:
                                pass
                            page.wait_for_timeout(1000)
                            return
                    except Exception:
                        pass

                    try:
                        current_url = login_frame.evaluate("() => window.location.href")
                        parsed = urlparse(current_url)
                        params = parse_qs(parsed.query)
                        for alt_type in ['phone_login', 'password_login', 'email_login', 'sms_login']:
                            params['type'] = [alt_type]
                            new_query = urlencode(params, doseq=True)
                            alt_url = urlunparse(parsed._replace(query=new_query))
                            logger.info(f"   🔄 尝试导航iframe到: {alt_url[:100]}")
                            try:
                                login_frame.goto(alt_url, timeout=10000)
                                try:
                                    login_frame.wait_for_load_state("domcontentloaded", timeout=8000)
                                except Exception:
                                    pass
                                page.wait_for_timeout(1000)
                                return
                            except Exception:
                                continue
                    except Exception:
                        pass

                    return
            except Exception:
                continue

    logger.warning(f"   ⚠️ 未找到可点击元素，已尝试 {len(unique)} 个选择器")


def _try_fill_with_fallback(page, target: str, value: str, step_num: int, login_frame=None):
    """填写操作，带回退。支持在iframe中查找"""
    candidates = [target]

    is_phone = any(kw in (target + value).lower() for kw in ['手机', 'phone', 'tel', 'mobile'])
    is_password = any(kw in (target + value).lower() for kw in ['密码', 'password', 'pwd'])
    is_username = any(kw in (target + value).lower() for kw in ['用户', '账号', 'account', 'username', 'email'])

    if is_phone:
        candidates.extend([
            'input[type="tel"]', 'input[name*="phone"]', 'input[name*="mobile"]',
            'input[placeholder*="手机"]', 'input[placeholder*="号码"]',
            'input[placeholder*="phone"]', 'input[placeholder*="mobile"]',
        ])
    if is_password:
        candidates.extend([
            'input[type="password"]', 'input[name*="password"]', 'input[name*="pwd"]',
            'input[placeholder*="密码"]', 'input[placeholder*="password"]',
        ])
    if is_username:
        candidates.extend([
            'input[type="text"]:first-of-type', 'input[name*="user"]', 'input[name*="account"]',
            'input[name*="email"]', 'input[placeholder*="账号"]', 'input[placeholder*="邮箱"]',
        ])

    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                try:
                    loc.click()
                    loc.fill(value)
                except Exception as fill_err:
                    if 'intercepts pointer events' in str(fill_err):
                        logger.info(f"   🔄 输入框被遮挡，尝试force fill: {sel}")
                        loc.fill(value, force=True)
                    else:
                        raise
                page.wait_for_timeout(300)
                return
        except Exception:
            continue

    if login_frame:
        logger.info(f"   🔄 主页面未找到输入框，尝试在iframe中查找...")
        for sel in candidates:
            try:
                loc = login_frame.locator(sel).first
                if loc.count() > 0:
                    logger.info(f"   ✅ 在iframe中找到输入框: {sel}")
                    loc.click()
                    loc.fill(value)
                    page.wait_for_timeout(300)
                    return
            except Exception:
                continue

        logger.info(f"   🔄 iframe中指定选择器均失败，尝试通用输入框查找...")
        try:
            visible_inputs = login_frame.evaluate("""() => {
                const inputs = document.querySelectorAll('input:not([type="hidden"])');
                const result = [];
                inputs.forEach(inp => {
                    const style = window.getComputedStyle(inp);
                    if (style.display === 'none' || style.visibility === 'hidden') return;
                    result.push({type: inp.type, placeholder: inp.placeholder || '', name: inp.name || '', id: inp.id || ''});
                });
                return result;
            }""")
            logger.info(f"   📊 iframe中可见输入框: {visible_inputs}")

            if is_phone and visible_inputs:
                for inp in visible_inputs:
                    if inp.get('type') in ('tel', 'text', 'number') and inp.get('type') != 'password':
                        sel = f'input[type="{inp["type"]}"]'
                        if inp.get('name'):
                            sel = f'input[name="{inp["name"]}"]'
                        elif inp.get('id'):
                            sel = f'input#{inp["id"]}'
                        elif inp.get('placeholder'):
                            sel = f'input[placeholder="{inp["placeholder"]}"]'
                        try:
                            loc = login_frame.locator(sel).first
                            if loc.count() > 0:
                                loc.click()
                                loc.fill(value)
                                page.wait_for_timeout(300)
                                logger.info(f"   ✅ 在iframe中通过通用查找填入手机号: {sel}")
                                return
                        except Exception:
                            continue

            if is_password and visible_inputs:
                for inp in visible_inputs:
                    if inp.get('type') == 'password':
                        sel = 'input[type="password"]'
                        if inp.get('name'):
                            sel = f'input[name="{inp["name"]}"]'
                        elif inp.get('id'):
                            sel = f'input#{inp["id"]}'
                        try:
                            loc = login_frame.locator(sel).first
                            if loc.count() > 0:
                                loc.click()
                                loc.fill(value)
                                page.wait_for_timeout(300)
                                logger.info(f"   ✅ 在iframe中通过通用查找填入密码: {sel}")
                                return
                        except Exception:
                            continue
        except Exception as e:
            logger.info(f"   ⚡ iframe通用输入框查找失败: {e}")

    page.wait_for_timeout(300)
    try:
        active = page.locator('input:focus, textarea:focus').first
        if active.count() > 0:
            active.fill(value)
            page.wait_for_timeout(300)
            logger.info(f"   🔄 回退: 填入当前焦点输入框")
            return
    except Exception:
        pass

    if login_frame:
        try:
            active = login_frame.locator('input:focus, textarea:focus').first
            if active.count() > 0:
                active.fill(value)
                page.wait_for_timeout(300)
                logger.info(f"   🔄 回退: 填入iframe中当前焦点输入框")
                return
        except Exception:
            pass

    logger.warning(f"   ⚠️ 未找到输入框: {target}")


def _try_check_with_fallback(page, target: str, step_num: int, login_frame=None):
    """勾选复选框，带回退。自定义组件用click而非check。支持在iframe中查找"""
    import re
    candidates = [target]

    if 'has-text' in target:
        m = re.search(r'has-text\("([^"]*)"\)', target)
        if m:
            text = m.group(1)
            candidates.append(f'text="{text}"')
            candidates.append(f'label:has-text("{text}") input[type="checkbox"]')
            candidates.append(f'input[type="checkbox"]:near(:text("{text}"))')

    if 'label:has-text' in target:
        m = re.search(r'label:has-text\("([^"]*)"\)', target)
        if m:
            text = m.group(1)
            candidates.append(f'text="{text}"')

    click_candidates = []
    for keyword in ['我已阅读', '同意', '协议', '隐私', 'agree', 'accept']:
        click_candidates.append(f'text={keyword}')
        click_candidates.append(f'span:has-text("{keyword}")')
        click_candidates.append(f'label:has-text("{keyword}")')
        click_candidates.append(f'div:has-text("{keyword}") >> input[type="checkbox"]')
        click_candidates.append(f'[class*="checkbox"]:near(:text("{keyword}"))')
        click_candidates.append(f'[class*="check"]:near(:text("{keyword}"))')

    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                try:
                    if not loc.is_checked():
                        loc.check()
                        page.wait_for_timeout(300)
                        logger.info(f"   ☑️ check()成功: {sel}")
                    return
                except Exception:
                    logger.info(f"   🔄 check()失败，尝试click(): {sel}")
                    try:
                        loc.click()
                        page.wait_for_timeout(300)
                        logger.info(f"   ☑️ click()代替check()成功: {sel}")
                        return
                    except Exception:
                        continue
        except Exception:
            continue

    logger.info("   🔄 标准选择器均失败，尝试点击协议文本...")
    for sel in click_candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.click()
                page.wait_for_timeout(300)
                logger.info(f"   ☑️ 点击协议文本成功: {sel}")
                return
        except Exception:
            continue

    if login_frame:
        logger.info(f"   🔄 主页面未找到复选框，尝试在iframe中查找...")
        for sel in candidates + click_candidates:
            try:
                loc = login_frame.locator(sel).first
                if loc.count() > 0:
                    try:
                        loc.click()
                        page.wait_for_timeout(300)
                        logger.info(f"   ☑️ 在iframe中点击成功: {sel}")
                        return
                    except Exception:
                        continue
            except Exception:
                continue

    logger.warning(f"   ⚠️ 未找到复选框: {target}")


def _perform_simple_login(page, username: str, password: str, username_selector: str = "",
                          password_selector: str = "", submit_selector: str = ""):
    """简单登录：直接查找用户名/密码输入框并填写提交"""
    logger.info("   🔍 简单登录模式：自动检测登录表单...")

    username_input = None
    password_input = None
    submit_button = None

    if username_selector:
        try:
            username_input = page.locator(username_selector).first
        except Exception:
            pass
    if not username_input:
        username_candidates = [
            'input[type="text"]', 'input[name="username"]', 'input[name="account"]',
            'input[name="mobile"]', 'input[name="phone"]', 'input[name="email"]',
            'input[placeholder*="账号"]', 'input[placeholder*="用户名"]',
            'input[placeholder*="手机"]', 'input[placeholder*="邮箱"]',
            'input[placeholder*="Username"]', 'input[placeholder*="Email"]',
            '#username', '#account', '#mobile', '#phone',
            'input:not([type="password"]):not([type="hidden"]):not([type="submit"]):not([type="checkbox"]):not([type="radio"])'
        ]
        for sel in username_candidates:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    username_input = loc
                    logger.info(f"   ✅ 自动检测到用户名输入框: {sel}")
                    break
            except Exception:
                continue

    if password_selector:
        try:
            password_input = page.locator(password_selector).first
        except Exception:
            pass
    if not password_input:
        for sel in ['input[type="password"]', 'input[name="password"]', '#password']:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    password_input = loc
                    logger.info(f"   ✅ 自动检测到密码输入框: {sel}")
                    break
            except Exception:
                continue

    if submit_selector:
        try:
            submit_button = page.locator(submit_selector).first
        except Exception:
            pass
    if not submit_button:
        for sel in [
            'button[type="submit"]', 'input[type="submit"]',
            'button:has-text("登录/注册")', 'button:has-text("登 录/注 册")',
            'button:has-text("登录")', 'button:has-text("登 录")',
            'button:has-text("Sign in")', 'button:has-text("Login")',
            '.ant-btn-primary', '.el-button--primary',
            'button', '[role="button"]'
        ]:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    submit_button = loc
                    logger.info(f"   ✅ 自动检测到登录按钮: {sel}")
                    break
            except Exception:
                continue

    if not username_input or not password_input:
        if not password_input and username_input:
            logger.info("   🔄 未找到密码输入框，尝试切换到密码登录...")
            try:
                for switch_sel in [
                    'text=密码登录', 'text=账号登录', 'text=帐号登录',
                    '.ant-tabs-tab:has-text("密码")', '[class*="tab"]:has-text("密码")',
                    'text=Password', 'text=Account'
                ]:
                    loc = page.locator(switch_sel).first
                    if loc.count() > 0:
                        loc.click()
                        page.wait_for_timeout(500)
                        break
                for sel in ['input[type="password"]', 'input[name="password"]', '#password']:
                    try:
                        loc = page.locator(sel).first
                        if loc.count() > 0:
                            password_input = loc
                            logger.info(f"   ✅ 切换后找到密码输入框: {sel}")
                            break
                    except Exception:
                        continue
            except Exception:
                pass

    if not username_input or not password_input:
        logger.warning("   ⚠️ 未能自动检测到登录表单，跳过登录")
        return

    try:
        username_input.click()
        username_input.fill(username)
        page.wait_for_timeout(300)
        logger.info(f"   ⌨️ 已填写用户名")
    except Exception as e:
        logger.warning(f"   ⚠️ 填写用户名失败: {e}")

    try:
        password_input.click()
        password_input.fill(password)
        page.wait_for_timeout(300)
        logger.info(f"   ⌨️ 已填写密码")
    except Exception as e:
        logger.warning(f"   ⚠️ 填写密码失败: {e}")

    try:
        for cb_sel in [
            'input[type="checkbox"]:visible', 'input[type="checkbox"]:not([disabled])',
            'text=我已阅读并同意', 'text=同意', 'text=协议',
            'label:has-text("协议") >> input[type="checkbox"]',
            'label:has-text("隐私") >> input[type="checkbox"]',
            'span:has-text("同意")', 'span:has-text("协议")',
            'div:has-text("我已阅读") >> input[type="checkbox"]',
            '[class*="checkbox"]:near(:text("同意"))',
            '[class*="check"]:near(:text("协议"))',
        ]:
            try:
                cb = page.locator(cb_sel).first
                if cb.count() > 0:
                    try:
                        if not cb.is_checked():
                            cb.check(force=True)
                            page.wait_for_timeout(300)
                        logger.info(f"   ☑️ 已勾选协议复选框: {cb_sel}")
                        break
                    except Exception:
                        logger.info(f"   🔄 check()失败，尝试click(): {cb_sel}")
                        try:
                            cb.click()
                            page.wait_for_timeout(300)
                            logger.info(f"   ☑️ click()勾选协议成功: {cb_sel}")
                            break
                        except Exception:
                            continue
            except Exception:
                continue
    except Exception:
        pass

    if submit_button:
        try:
            submit_button.click()
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(2000)
            logger.info(f"   🖱️ 已点击登录按钮")
        except Exception as e:
            logger.warning(f"   ⚠️ 点击登录按钮失败: {e}")
            try:
                page.keyboard.press("Enter")
                page.wait_for_load_state("networkidle", timeout=15000)
                page.wait_for_timeout(2000)
                logger.info(f"   ⌨️ 已按回车提交登录")
            except Exception:
                pass
    else:
        try:
            page.keyboard.press("Enter")
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(2000)
            logger.info(f"   ⌨️ 已按回车提交登录")
        except Exception:
            pass

    logger.info(f"   🔐 简单登录完成，当前页面: {page.url}")


LOGIN_PAGE_ANALYZE_PROMPT = """
你是一个网页自动化专家。你需要根据登录页面的结构信息，生成一组Playwright操作步骤来完成登录。

【背景】
用户已经提供了用户名和密码，你需要分析页面结构，找出完成登录所需的全部操作步骤。

【常见情况处理】
1. 如果页面默认是"短信登录"/"验证码登录"，需要先点击切换到"密码登录"或"账号登录"
2. 如果有"同意隐私协议"/"用户协议"等复选框，需要先勾选（使用click而非check）
3. 如果有"记住密码"/"自动登录"等复选框，可以勾选
4. 填写用户名和密码
5. 点击"登录"按钮

【输出格式】
返回纯JSON数组，每个元素是一个操作步骤：
[
  {
    "action": "click" | "fill" | "check" | "wait" | "press" | "wait_for_navigation",
    "target": "Playwright选择器",
    "value": "填写内容（fill时必填）",
    "description": "操作描述"
  }
]

【选择器生成规则 —— 非常重要】
- 按钮类：优先使用 button:has-text() 或 [type="submit"]
  ⚠️ 登录按钮文字可能是"登录/注册"、"登 录/注 册"等，使用 button:has-text("登录") 匹配
  ⚠️ 禁止对登录按钮使用 text=登录，因为会误匹配到"密码登录"tab文字
  ⚠️ 切换tab才用 text=精确匹配，如 text=密码登录
- 输入框类：优先使用 placeholder，如 input[placeholder*="手机号"]、input[placeholder*="密码"]
- 复选框类：优先使用 click 点击协议文本，如 text=我已阅读并同意
  不要用 label:has-text() >> input[type="checkbox"]，因为自定义组件可能没有真实checkbox
- 标签切换：使用 text= 精确匹配，如 text=密码登录
- 如果有 name 或 id 属性，优先使用：如 #username、input[name="password"]
- 选择器必须基于页面结构中实际出现的文本和属性，不要凭空编造

【注意事项】
- 如果页面已经有用户名/密码输入框可见，直接填写即可，不需要切换
- fill 操作前不需要先 click，直接 fill
- 勾选协议用 action: "check"，系统会自动尝试 click 作为回退
- 最后一步必须是 wait_for_navigation，等待登录成功跳转
- 最多返回10个步骤

【重要：弹窗已打开的情况】
如果页面信息中包含"登录弹窗/对话框已经打开"的提示，说明登录弹窗已经打开，你不需要再点击登录入口按钮。
直接在弹窗内操作即可（切换tab、填写输入框、勾选协议、点击登录按钮）。
绝对不要生成"点击登录按钮打开登录弹窗"这样的步骤，那会关闭已经打开的弹窗！
"""


EXPLORATORY_CASE_PROMPT = """
你是一名资深测试专家，需要根据页面探索结果（可交互元素列表），生成自动化测试用例集。

【核心任务】
你拿到的是一个Web页面的全部可交互元素清单（按钮、输入框、下拉框、链接、表格、表单、菜单等）。
请推断出该页面的业务流程，并为每个流程生成测试用例。

【覆盖要求】
1. 每个功能点生成1条正向用例 + 1-2条关键异常用例
2. 优先覆盖核心业务流程，不要过度生成
3. 场景类型：正向流程 / 异常操作 / 边界条件

【用例字段规范】
- id：TC-{模块缩写}-{3位序号}，如 TC-ORDER-001
- name：用例名称（简洁）
- preconditions：前置条件（一句话）
- steps：Playwright格式步骤（action/selector/value/description）
  选择器要使用从元素列表中提取的真实属性（placeholder、id、文本等）
- expected_result：预期结果（一句话）
- scenario_type：正向流程 / 异常操作 / 边界条件

【步骤选择器生成规则】
- 按钮：text=按钮文字
- 输入框：placeholder或id
- 下拉框：text=选项文字
- 链接/菜单：text=链接文字
- 表单验证：assert_text

【输出要求】
返回纯JSON数组，不要额外解释，不要markdown代码块。确保JSON格式完整有效。
"""


@retry_api_call
def ai_generate_exploratory_cases(url: str, login_url: str = "", username: str = "",
                                   password: str = "", username_selector: str = "",
                                   password_selector: str = "", submit_selector: str = "") -> List[Dict]:
    logger.info(f"🔍 探索性生成用例: {url}")
    try:
        page_data = crawl_page_interactive_elements(
            url, login_url, username, password,
            username_selector, password_selector, submit_selector
        )
        elements_json = json.dumps(page_data["elements"], ensure_ascii=False, indent=2)

        if len(elements_json) > 12000:
            elements_json = elements_json[:12000]

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": EXPLORATORY_CASE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"页面标题：{page_data['title']}\n"
                        f"页面URL：{url}\n"
                        f"可交互元素清单：\n{elements_json}\n"
                        f"请根据以上元素清单，推断业务流程并生成测试用例。"
                    )
                }
            ],
            max_tokens=8000,
            temperature=0.1,
            timeout=AI_REQUEST_TIMEOUT
        )
        ai_text = response.choices[0].message.content.strip()
        test_cases = extract_json_from_response(ai_text)
        if isinstance(test_cases, dict):
            test_cases = test_cases.get("test_cases", [])
        valid_cases = []
        for case in test_cases:
            if not case.get("name"):
                continue
            steps = case.get("steps", [])
            valid_steps = []
            for step in steps:
                if not step.get("action") or not step.get("selector"):
                    continue
                if step.get("action") not in ("fill", "click", "wait_for_selector", "assert_text"):
                    continue
                if step.get("action") in ("fill", "assert_text") and not step.get("value"):
                    continue
                valid_steps.append(step)
            case["steps"] = valid_steps if valid_steps else steps
            case["url"] = url
            case["test_type"] = "web"
            valid_cases.append(case)
        logger.info(f"✅ 探索性生成{len(valid_cases)}个用例")
        return valid_cases
    except Exception as e:
        logger.error(f"❌ 探索性生成失败: {e}", exc_info=True)
        raise