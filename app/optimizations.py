import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page

load_dotenv()
logger = logging.getLogger(__name__)

# ======================== 优化开关（默认全部关闭，保持向后兼容） ========================
OPTIMIZATIONS = {
    "browser_pool": os.getenv("OPT_BROWSER_POOL", "false").lower() == "true",
    "ai_cache": os.getenv("OPT_AI_CACHE", "false").lower() == "true",
    "dom_preprocess": os.getenv("OPT_DOM_PREPROCESS", "false").lower() == "true",
    "page_wait_unified": os.getenv("OPT_PAGE_WAIT_UNIFIED", "false").lower() == "true",
    "dom_extract_unified": os.getenv("OPT_DOM_EXTRACT_UNIFIED", "false").lower() == "true",
}

VIEWPORT = {"width": 1920, "height": 1080}
PAGE_LOAD_TIMEOUT = 60000
BROWSER_POOL_MAX_SIZE = int(os.getenv("BROWSER_POOL_MAX_SIZE", "3"))
BROWSER_POOL_IDLE_TIMEOUT = int(os.getenv("BROWSER_POOL_IDLE_TIMEOUT", "300"))


# ======================== 浏览器池管理 ========================
class BrowserPoolManager:
    """浏览器实例池管理器，复用浏览器实例避免频繁启动关闭"""

    def __init__(self, max_size: int = BROWSER_POOL_MAX_SIZE, idle_timeout: int = BROWSER_POOL_IDLE_TIMEOUT):
        self._playwright = None
        self._pool: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._idle_timeout = idle_timeout

    def _ensure_playwright(self):
        if self._playwright is None:
            self._playwright = sync_playwright().start()

    def _create_key(self, headless: bool = True, viewport: dict = None) -> str:
        vp = viewport or VIEWPORT
        return f"headless={headless}_vp={vp['width']}x{vp['height']}"

    def acquire(self, headless: bool = True, viewport: dict = None) -> Tuple[any, any, any]:
        """获取浏览器、context、page 三元组。复用已有实例或创建新的。"""
        if not OPTIMIZATIONS["browser_pool"]:
            p = sync_playwright().start()
            browser = p.chromium.launch(headless=headless)
            vp = viewport or VIEWPORT
            context = browser.new_context(viewport=vp)
            page = context.new_page()
            return p, browser, context, page

        self._ensure_playwright()
        key = self._create_key(headless, viewport)

        with self._lock:
            entry = self._pool.get(key)
            if entry and entry["page"] and not entry["page"].is_closed():
                entry["last_used"] = time.time()
                logger.debug(f"♻️ 浏览器池复用: {key}")
                return None, entry["browser"], entry["context"], entry["page"]

            if len(self._pool) >= self._max_size:
                oldest_key = min(self._pool, key=lambda k: self._pool[k]["last_used"])
                self._release(oldest_key)

            vp = viewport or VIEWPORT
            browser = self._playwright.chromium.launch(headless=headless)
            context = browser.new_context(viewport=vp)
            page = context.new_page()
            self._pool[key] = {
                "browser": browser,
                "context": context,
                "page": page,
                "last_used": time.time()
            }
            logger.debug(f"🆕 浏览器池新建: {key}, 当前池大小: {len(self._pool)}")
            return None, browser, context, page

    def release(self, page):
        """释放页面，保留浏览器实例供下次复用"""
        if not OPTIMIZATIONS["browser_pool"]:
            return
        try:
            if page and not page.is_closed():
                page.goto("about:blank")
        except Exception:
            pass

    def _release(self, key: str):
        entry = self._pool.pop(key, None)
        if entry:
            try:
                if entry["browser"]:
                    entry["browser"].close()
            except Exception:
                pass

    def shutdown(self):
        with self._lock:
            for key in list(self._pool.keys()):
                self._release(key)
            self._pool.clear()
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def cleanup_idle(self):
        """清理空闲超时的浏览器实例"""
        if not self._pool:
            return
        now = time.time()
        with self._lock:
            for key in list(self._pool.keys()):
                entry = self._pool[key]
                if now - entry["last_used"] > self._idle_timeout:
                    self._release(key)
                    logger.debug(f"🗑️ 浏览器池清理空闲实例: {key}")


browser_pool = BrowserPoolManager()


# ======================== AI响应缓存 ========================
class AICacheManager:
    """AI响应本地缓存，避免相同请求重复调用API"""

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_size = int(os.getenv("AI_CACHE_MAX_SIZE", "200"))

    def _make_key(self, model: str, prompt_hash: str, user_content: str) -> str:
        content_hash = hashlib.md5(user_content.encode("utf-8")).hexdigest()
        return f"{model}:{prompt_hash}:{content_hash}"

    def get(self, model: str, prompt_hash: str, user_content: str) -> Optional[dict]:
        if not OPTIMIZATIONS["ai_cache"]:
            return None
        key = self._make_key(model, prompt_hash, user_content)
        with self._lock:
            entry = self._pool.get(key) if hasattr(self, "_pool") else self._cache.get(key)
            if entry:
                entry["hits"] = entry.get("hits", 0) + 1
                logger.info(f"🎯 AI缓存命中 (hits={entry['hits']}): {key[:40]}...")
                return entry["result"]
        return None

    def set(self, model: str, prompt_hash: str, user_content: str, result: dict):
        if not OPTIMIZATIONS["ai_cache"]:
            return
        key = self._make_key(model, prompt_hash, user_content)
        with self._lock:
            target = self._pool if hasattr(self, "_pool") else self._cache
            if len(target) >= self._max_size:
                oldest = min(target, key=lambda k: target[k].get("timestamp", 0))
                del target[oldest]
            target[key] = {
                "result": result,
                "hits": 1,
                "timestamp": time.time()
            }

    def clear(self):
        with self._lock:
            target = self._pool if hasattr(self, "_pool") else self._cache
            target.clear()

    def stats(self) -> dict:
        with self._lock:
            target = self._pool if hasattr(self, "_pool") else self._cache
            total_hits = sum(e.get("hits", 0) for e in target.values())
            return {
                "total_entries": len(target),
                "total_hits": total_hits,
                "enabled": OPTIMIZATIONS["ai_cache"]
            }


ai_cache = AICacheManager()


# ======================== DOM预处理 ========================
def preprocess_dom_for_ai(html_content: str, max_length: int = 20000) -> str:
    """预处理DOM内容，过滤无用标签和属性，减少AI Token消耗"""
    if not OPTIMIZATIONS["dom_preprocess"]:
        return html_content[:max_length]

    import re

    # 移除 script 和 style 标签及其内容
    cleaned = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html_content)
    cleaned = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', cleaned)
    cleaned = re.sub(r'<noscript[^>]*>[\s\S]*?</noscript>', '', cleaned)

    # 移除注释
    cleaned = re.sub(r'<!--[\s\S]*?-->', '', cleaned)

    # 移除SVG
    cleaned = re.sub(r'<svg[^>]*>[\s\S]*?</svg>', '', cleaned)

    # 移除无用属性（保留关键属性）
    useless_attrs = r'\s+(?:data-v-[a-f0-9]+|data-reactid|data-reactroot|aria-\w+|on\w+|role|tabindex|style|class)="[^"]*"'
    cleaned = re.sub(useless_attrs, '', cleaned)

    # 移除多余空白
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = re.sub(r'>\s+<', '><', cleaned)

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    logger.info(f"   🧹 DOM预处理: {len(html_content)} → {len(cleaned)} 字符 " +
                f"({(1 - len(cleaned)/len(html_content))*100:.0f}% 减少)")
    return cleaned


# ======================== 统一页面加载等待 ========================
WAIT_NETWORKIDLE_TIMEOUT = 15000
WAIT_DOMCONTENTLOADED_TIMEOUT = 10000
WAIT_EXTRA_TIMEOUT = 2000


def wait_for_page_load(page: Page,
                       networkidle_timeout: int = WAIT_NETWORKIDLE_TIMEOUT,
                       dom_timeout: int = WAIT_DOMCONTENTLOADED_TIMEOUT,
                       extra_wait: int = WAIT_EXTRA_TIMEOUT):
    """统一的页面加载等待策略：networkidle → domcontentloaded → 固定等待"""
    try:
        page.wait_for_load_state("networkidle", timeout=networkidle_timeout)
    except Exception:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=dom_timeout)
        except Exception:
            pass
    page.wait_for_timeout(extra_wait)


# ======================== 统一DOM元素提取 ========================
DOM_EXTRACT_SCRIPT = """
() => {
    const results = {
        buttons: [],
        inputs: [],
        links: [],
        selects: [],
        headings: [],
        labels: [],
        forms: [],
        dialogs: []
    };
    try {
        document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]').forEach(el => {
            const info = {
                tag: el.tagName.toLowerCase(),
                text: (el.textContent || '').trim().substring(0, 80),
                id: el.id || '',
                className: (el.className && typeof el.className === 'string') ? el.className.substring(0, 100) : '',
                type: el.getAttribute('type') || '',
                placeholder: el.getAttribute('placeholder') || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                visible: el.offsetParent !== null
            };
            if (info.text || info.id || info.ariaLabel) results.buttons.push(info);
        });
        document.querySelectorAll('input:not([type="button"]):not([type="submit"]), textarea').forEach(el => {
            const info = {
                tag: el.tagName.toLowerCase(),
                id: el.id || '',
                name: el.getAttribute('name') || '',
                type: el.getAttribute('type') || 'text',
                placeholder: el.getAttribute('placeholder') || '',
                value: el.getAttribute('value') || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                className: (el.className && typeof el.className === 'string') ? el.className.substring(0, 100) : '',
                visible: el.offsetParent !== null
            };
            if (info.id || info.name || info.placeholder || info.ariaLabel) results.inputs.push(info);
        });
        document.querySelectorAll('a[href]').forEach(el => {
            const info = {
                text: (el.textContent || '').trim().substring(0, 80),
                href: el.getAttribute('href') || '',
                id: el.id || '',
                className: (el.className && typeof el.className === 'string') ? el.className.substring(0, 100) : '',
                visible: el.offsetParent !== null
            };
            if (info.text && info.href) results.links.push(info);
        });
        document.querySelectorAll('select').forEach(el => {
            const options = [];
            el.querySelectorAll('option').forEach(opt => {
                options.push({text: opt.textContent.trim().substring(0, 50), value: opt.value});
            });
            results.selects.push({
                id: el.id || '', name: el.getAttribute('name') || '',
                options: options.slice(0, 20), visible: el.offsetParent !== null
            });
        });
        document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(el => {
            const t = (el.textContent || '').trim();
            if (t) results.headings.push({tag: el.tagName.toLowerCase(), text: t.substring(0, 100)});
        });
        document.querySelectorAll('label').forEach(el => {
            const t = (el.textContent || '').trim();
            if (t) results.labels.push({for: el.getAttribute('for') || '', text: t.substring(0, 80)});
        });
        document.querySelectorAll('form').forEach(el => {
            results.forms.push({
                id: el.id || '', action: el.getAttribute('action') || '',
                method: el.getAttribute('method') || 'get', visible: el.offsetParent !== null
            });
        });
        document.querySelectorAll('.ant-modal, .el-dialog, [role="dialog"], .modal, .login-dialog, [class*="login"]').forEach(el => {
            const t = (el.textContent || '').trim().substring(0, 200);
            const cls = (el.className && typeof el.className === 'string') ? el.className.substring(0, 100) : '';
            if (t) results.dialogs.push({className: cls, text: t, visible: el.offsetParent !== null});
        });
    } catch(e) {}
    return results;
}
"""


def extract_interactive_elements(page: Page) -> dict:
    """提取页面中所有可交互元素的结构化信息"""
    try:
        return page.evaluate(DOM_EXTRACT_SCRIPT)
    except Exception as e:
        logger.warning(f"⚠️ DOM元素提取失败: {e}")
        return {}


def format_extracted_elements(elements: dict) -> str:
    """将提取的元素格式化为结构化文本，供AI分析"""
    lines = []
    if elements.get("forms"):
        lines.append("📋 表单:")
        for f in elements["forms"][:5]:
            lines.append(f'  - id={f["id"]} action={f["action"]} method={f["method"]}')
    if elements.get("dialogs"):
        lines.append("🪟 弹窗/对话框:")
        for d in elements["dialogs"][:5]:
            lines.append(f'  - [{d["className"]}] {d["text"][:100]}')
    if elements.get("inputs"):
        lines.append("📝 输入框:")
        for inp in elements["inputs"][:30]:
            lines.append(f'  - id={inp["id"]} name={inp["name"]} type={inp["type"]} ' +
                         f'placeholder={inp["placeholder"]}')
    if elements.get("buttons"):
        lines.append("🔘 按钮:")
        for btn in elements["buttons"][:20]:
            lines.append(f'  - id={btn["id"]} text={btn["text"]}')
    if elements.get("selects"):
        lines.append("📊 下拉框:")
        for sel in elements["selects"][:5]:
            opts = ", ".join([o["text"] for o in sel["options"][:10]])
            lines.append(f'  - id={sel["id"]} options=[{opts}]')
    if elements.get("links"):
        lines.append("🔗 链接:")
        for link in elements["links"][:15]:
            lines.append(f'  - href={link["href"]} text={link["text"]}')
    if elements.get("headings"):
        lines.append("📌 标题:")
        for h in elements["headings"][:10]:
            lines.append(f'  - {h["tag"]}: {h["text"]}')
    return "\n".join(lines)


# ======================== 登录参数数据类 ========================
@dataclass
class LoginConfig:
    """登录相关参数封装"""
    login_url: str = ""
    username: str = ""
    password: str = ""
    username_selector: str = ""
    password_selector: str = ""
    submit_selector: str = ""

    def to_dict(self) -> dict:
        return {
            "login_url": self.login_url,
            "username": self.username,
            "password": self.password,
            "username_selector": self.username_selector,
            "password_selector": self.password_selector,
            "submit_selector": self.submit_selector,
        }

    def is_configured(self) -> bool:
        return bool(self.login_url and self.username and self.password)


# ======================== JSON解析增强 ========================
def robust_json_parse(json_str: str) -> tuple:
    """增强的JSON解析，返回 (parsed_result, success)"""
    import re as _re

    if not json_str or not json_str.strip():
        return [], False


# ======================== 【新增】Session持久化管理 ========================

class SessionManager:
    """【新增】登录Session持久化管理器，支持按域名+用户保存/恢复/清理"""

    def __init__(self, session_dir: str = None):
        self._session_dir = session_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")
        os.makedirs(self._session_dir, exist_ok=True)

    def _session_path(self, domain: str, username: str = "default") -> str:
        safe_domain = domain.replace("://", "_").replace("/", "_").replace(":", "_")
        safe_user = username.replace("@", "_").replace("/", "_")
        return os.path.join(self._session_dir, f"{safe_domain}__{safe_user}.json")

    def save(self, domain: str, context, username: str = "default"):
        """保存浏览器Session到文件"""
        filepath = self._session_path(domain, username)
        try:
            context.storage_state(path=filepath)
            logger.info(f"💾 Session已保存: {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ Session保存失败: {e}")

    def load(self, domain: str, username: str = "default") -> Optional[dict]:
        """加载Session，返回storage_state或None"""
        filepath = self._session_path(domain, username)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info(f"📂 Session已加载: {filepath}")
            return state
        except Exception as e:
            logger.warning(f"⚠️ Session加载失败: {e}")
            return None

    def is_valid(self, domain: str, username: str = "default", max_age_hours: int = 24) -> bool:
        """检查Session是否有效（文件存在且未过期）"""
        filepath = self._session_path(domain, username)
        if not os.path.exists(filepath):
            return False
        file_age = time.time() - os.path.getmtime(filepath)
        is_valid = file_age < max_age_hours * 3600
        if not is_valid:
            logger.info(f"⏰ Session已过期（{file_age/3600:.1f}小时前）: {filepath}")
        return is_valid

    def clear(self, domain: str = None, username: str = None):
        """清除Session：清除指定域名/用户的Session，或清除全部"""
        if domain is None:
            import glob
            for f in glob.glob(os.path.join(self._session_dir, "*.json")):
                try:
                    os.remove(f)
                    logger.info(f"🗑️ 已清除Session: {f}")
                except Exception as e:
                    logger.warning(f"⚠️ Session清除失败: {f} -> {e}")
        else:
            filepath = self._session_path(domain, username or "default")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"🗑️ 已清除Session: {filepath}")
                except Exception as e:
                    logger.warning(f"⚠️ Session清除失败: {filepath} -> {e}")

    def list_sessions(self) -> List[dict]:
        """列出所有已保存的Session"""
        sessions = []
        import glob
        for f in glob.glob(os.path.join(self._session_dir, "*.json")):
            try:
                stat = os.stat(f)
                age_hours = (time.time() - stat.st_mtime) / 3600
                sessions.append({
                    "file": os.path.basename(f),
                    "size_kb": stat.st_size / 1024,
                    "age_hours": round(age_hours, 1),
                    "valid": age_hours < 24
                })
            except Exception:
                pass
        return sorted(sessions, key=lambda s: s["age_hours"])


# 模块级单例
session_manager = SessionManager()
# ======================== 【新增结束】 ========================

# ======================== 【新增】结构化日志 ========================

USE_STRUCTURED_LOGGING = os.getenv("OPT_STRUCTURED_LOGGING", "false").lower() == "true"


class JSONLogFormatter(logging.Formatter):
    """【新增】JSON格式日志输出器，方便接入日志分析系统"""

    def format(self, record: logging.LogRecord) -> str:
        import datetime as _dt
        log_entry = {
            "timestamp": _dt.datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging():
    """【新增】配置结构化日志：将根logger的handler替换为JSON格式"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info("📋 结构化日志已启用（JSON格式）")


if USE_STRUCTURED_LOGGING:
    setup_structured_logging()

# ======================== 【新增结束】 ========================