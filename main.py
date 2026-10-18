"""团学事务智能沟通小助手（astrbot_plugin_student_assistant）。

设计定位：学生自助问答由 AstrBot 原生知识库（RAG）完成，本插件只补齐
原生能力做不到的业务闭环，以"旁路监听"方式工作，不干扰原生问答体验。
群聊与私聊（单聊）均可用，私聊场景数据以 private:<用户ID> 归档：

1. 未答收集 —— 挂载 LLM 请求钩子，检查本次请求是否注入了知识库上下文。
   未注入（即知识库没答上）、消息像提问且未命中已学会的问答时，自动
   记录到数据库，并实时推送给已订阅推送的负责人，避免"机器人答不上来
   的问题石沉大海"。
2. 解答学会 —— /未答解决 <编号> <标准答案> 把解答登记为 FAQ；之后
   相似问题会自动把参考答案注入 LLM 请求，实现"问过一次，下次就会"。
3. 链接入库 —— 管理员发送 /加入知识库 <网页链接>，自动抓取网页内容
   写入配置的知识库，无需打开 WebUI 上传文件。
4. 沟通摘要 —— /整理（引用消息或回溯最近群聊），AI 整理老师意见、部门
   反馈、外院对接等多方沟通，提炼关键要求与待办事项，待办自动登记入清单。
5. 待办同步 —— /记待办、/待办列表、/完成待办、/删待办、/同步待办 [老师]，
   支持按同步对象（老师/同学）自动变换语气播报。
"""

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.message.message_event_result import MessageChain

PLUGIN_NAME = "astrbot_plugin_student_assistant"

# 统一使用东八区时间，避免 Docker 容器时区不一致
CST = timezone(timedelta(hours=8))

# 知识库命中标记：AstrBot 内置知识库检索到内容后，会把带该固定前缀的
# 上下文注入 LLM 请求（见 astrbot/core/knowledge_base/kb_mgr.py 的
# _format_context）。据此判断本次回答是否基于知识库。
KB_CONTEXT_MARKER = "以下是相关的知识库内容"

# 同群同一问题/待办在该小时数内不重复记录
DEDUP_WINDOW_HOURS = 24

# 单条问题/待办/单条消息的最长保留字符数
MAX_TEXT_LEN = 500

# 判定"像提问"的最短消息长度（过滤寒暄与误触）
MIN_QUESTION_LEN = 4

# 群消息缓存策略：QQ 官方机器人无法拉取历史消息，由插件实时缓存
CACHE_KEEP_COUNT = 200  # 每个群最多保留多少条
CACHE_TTL_HOURS = 48  # 超过此时长的缓存自动删除

# /同步待办 单次最多播报条数
SYNC_TODO_LIMIT = 20

# FAQ 相似匹配阈值（问题 2-gram Jaccard 相似度），达到即视为"相似问题"
FAQ_SIM_THRESHOLD = 0.4

# FAQ 答案最长保留字符数
MAX_ANSWER_LEN = 2000

# 允许出现的截止时间字符（不含数字直接判空；超宽泛字符集做粗校验）
_DEADLINE_PATTERN = re.compile(r"^[0-9A-Za-z\-/月日周点分秒时:：. ]{1,24}$")


def _now_cst() -> datetime:
    return datetime.now(CST)


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _data_dir() -> str:
    """持久化数据目录。

    优先使用 AstrBot/data/plugins_data/<plugin>/（Docker 挂载卷内，
    升级/重装插件不会丢失）；找不到 data 目录时回退到插件目录内。
    """
    if os.path.isdir(os.path.join(os.getcwd(), "data")):
        target = os.path.join(os.getcwd(), "data", "plugins_data", PLUGIN_NAME)
    else:
        target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(target, exist_ok=True)
    return target


def _normalize_text(text: str) -> str:
    """去重规范化：去掉空白与常用中英文标点，转小写。"""
    return re.sub(
        r"[\s，。！？?!\.,:：;；、~～'\"“”‘’()（）【】\[\]{}]+", "", text
    ).lower()


def _bigrams(text: str) -> set[str]:
    """提取文本的 2-gram 集合（中文短文本相似度用）。"""
    text = _normalize_text(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """两个集合的 Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clean_deadline(value: str) -> str:
    """校验并规范化 AI 返回的截止时间：不合法时返回空字符串。"""
    text = str(value or "").strip().replace("：", ":")
    if not text or not re.search(r"\d", text):
        return ""
    if not _DEADLINE_PATTERN.fullmatch(text):
        return ""
    return text[:24]


def _parse_todos(data: dict | None) -> list[dict]:
    """从 AI 返回的 JSON 中提取待办列表，剔除非法项。"""
    if not data:
        return []
    todos = data.get("todos")
    if not isinstance(todos, list):
        return []
    out: list[dict] = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        content = str(t.get("content", "")).strip()
        if not content:
            continue
        out.append(
            {
                "content": content[:MAX_TEXT_LEN],
                "deadline": _clean_deadline(str(t.get("deadline", ""))),
                "assignee": str(t.get("assignee", "")).strip()[:50],
            }
        )
    return out


class AssistantStore:
    """未答问题 / 待办 / 群消息缓存的 SQLite 存储层。

    写入量很小，单连接 + 异步锁即可。
    """

    def __init__(self, db_path: str) -> None:
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unanswered (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                sender_id TEXT DEFAULT '',
                sender_name TEXT DEFAULT '',
                question TEXT NOT NULL,
                q_norm TEXT NOT NULL DEFAULT '',
                asked_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT DEFAULT ''
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unanswered_lookup "
            "ON unanswered(group_id, status)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_norm TEXT NOT NULL DEFAULT '',
                assignee TEXT DEFAULT '',
                deadline TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                source TEXT DEFAULT '',
                creator_name TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                done_at TEXT DEFAULT ''
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_todos_lookup "
            "ON todos(group_id, status)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cached_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                msg_id TEXT NOT NULL,
                sender_id TEXT DEFAULT '',
                sender_name TEXT DEFAULT '',
                content TEXT DEFAULT '',
                ts INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_cached_msg "
            "ON cached_messages(group_id, msg_id)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_id TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL,
                q_norm TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faq_norm ON faq(q_norm)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS push_targets (
                umo TEXT PRIMARY KEY,
                platform_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # ---------- 未答问题 ----------

    async def add_unanswered(
        self,
        *,
        group_id: str,
        sender_id: str,
        sender_name: str,
        question: str,
        asked_at: datetime,
    ) -> int | None:
        """记录一条未答问题。同群同问题在去重窗口内已存在时返回 None。"""
        question = question.strip()[:MAX_TEXT_LEN]
        q_norm = _normalize_text(question)
        if not q_norm:
            return None
        since = _fmt_time(asked_at - timedelta(hours=DEDUP_WINDOW_HOURS))
        async with self._lock:
            dup = self._conn.execute(
                "SELECT 1 FROM unanswered WHERE group_id = ? AND q_norm = ? "
                "AND status = 'pending' AND asked_at >= ? LIMIT 1",
                (group_id, q_norm, since),
            ).fetchone()
            if dup:
                return None
            cur = self._conn.execute(
                "INSERT INTO unanswered (group_id, sender_id, sender_name, "
                "question, q_norm, asked_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    group_id,
                    sender_id,
                    sender_name,
                    question,
                    q_norm,
                    _fmt_time(asked_at),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    async def list_pending_unanswered(
        self, group_id: str | None, limit: int
    ) -> list[sqlite3.Row]:
        """列出待处理的未答问题；group_id 为 None 时列出所有群。"""
        async with self._lock:
            if group_id is None:
                return self._conn.execute(
                    "SELECT * FROM unanswered WHERE status = 'pending' "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM unanswered WHERE group_id = ? AND status = 'pending' "
                "ORDER BY id DESC LIMIT ?",
                (group_id, limit),
            ).fetchall()

    async def resolve_unanswered(
        self, record_id: int, group_id: str | None = None
    ) -> sqlite3.Row | None:
        """把一条未答问题标记为已处理，返回被处理的记录。

        group_id 为 None（私聊管理员操作）时不校验归属，可处理任意来源；
        群聊操作时仅允许处理本群记录。
        """
        async with self._lock:
            if group_id is None:
                row = self._conn.execute(
                    "SELECT * FROM unanswered "
                    "WHERE id = ? AND status = 'pending'",
                    (record_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM unanswered "
                    "WHERE id = ? AND group_id = ? AND status = 'pending'",
                    (record_id, group_id),
                ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE unanswered SET status = 'resolved', resolved_at = ? "
                "WHERE id = ?",
                (_fmt_time(_now_cst()), record_id),
            )
            self._conn.commit()
            return row

    # ---------- 已学会的问答（FAQ） ----------

    async def add_faq(
        self,
        *,
        scope_id: str,
        question: str,
        answer: str,
        source_id: int = 0,
        created_at: datetime,
    ) -> tuple[int, bool]:
        """登记/更新一条问答。同一问题已存在时更新答案。

        返回 (编号, 是否为更新已有问答)。
        """
        question = question.strip()[:MAX_TEXT_LEN]
        q_norm = _normalize_text(question)
        answer = answer.strip()[:MAX_ANSWER_LEN]
        if not q_norm or not answer:
            return 0, False
        async with self._lock:
            row = self._conn.execute(
                "SELECT id FROM faq WHERE q_norm = ?", (q_norm,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE faq SET answer = ?, scope_id = ?, source_id = ? "
                    "WHERE id = ?",
                    (answer, scope_id, source_id, row["id"]),
                )
                self._conn.commit()
                return row["id"], True
            cur = self._conn.execute(
                "INSERT INTO faq (scope_id, question, q_norm, answer, "
                "source_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    scope_id,
                    question,
                    q_norm,
                    answer,
                    source_id,
                    _fmt_time(created_at),
                ),
            )
            self._conn.commit()
            return cur.lastrowid, False

    async def list_faq(self, limit: int = 200) -> list[sqlite3.Row]:
        """列出已学会的问答（新到旧）。"""
        async with self._lock:
            return self._conn.execute(
                "SELECT * FROM faq ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    async def delete_faq(self, record_id: int) -> str | None:
        """删除一条问答，返回被删问题（未找到返回 None）。"""
        async with self._lock:
            row = self._conn.execute(
                "SELECT question FROM faq WHERE id = ?", (record_id,)
            ).fetchone()
            if not row:
                return None
            self._conn.execute("DELETE FROM faq WHERE id = ?", (record_id,))
            self._conn.commit()
            return row["question"]

    # ---------- 未答推送订阅 ----------

    async def add_push_target(
        self, umo: str, platform_name: str, created_at: datetime
    ) -> bool:
        """订阅未答推送；已订阅时返回 False。"""
        async with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO push_targets (umo, platform_name, "
                "created_at) VALUES (?, ?, ?)",
                (umo, platform_name, _fmt_time(created_at)),
            )
            self._conn.commit()
            return cur.rowcount > 0

    async def remove_push_target(self, umo: str) -> bool:
        """取消订阅未答推送。"""
        async with self._lock:
            cur = self._conn.execute(
                "DELETE FROM push_targets WHERE umo = ?", (umo,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    async def list_push_targets(self) -> list[str]:
        """列出所有推送目标会话（unified_msg_origin）。"""
        async with self._lock:
            rows = self._conn.execute(
                "SELECT umo FROM push_targets"
            ).fetchall()
            return [r["umo"] for r in rows]

    # ---------- 待办清单 ----------

    async def add_todo(
        self,
        *,
        group_id: str,
        content: str,
        assignee: str = "",
        deadline: str = "",
        source: str = "manual",
        creator_name: str = "",
        created_at: datetime,
    ) -> int | None:
        """登记一条待办。同群同内容待办已存在（待处理）时返回 None。"""
        content = content.strip()[:MAX_TEXT_LEN]
        c_norm = _normalize_text(content)
        if not c_norm:
            return None
        async with self._lock:
            dup = self._conn.execute(
                "SELECT 1 FROM todos WHERE group_id = ? AND content_norm = ? "
                "AND status = 'pending' LIMIT 1",
                (group_id, c_norm),
            ).fetchone()
            if dup:
                return None
            cur = self._conn.execute(
                "INSERT INTO todos (group_id, content, content_norm, assignee, "
                "deadline, source, creator_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    group_id,
                    content,
                    c_norm,
                    assignee,
                    deadline,
                    source,
                    creator_name,
                    _fmt_time(created_at),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    async def list_pending_todos(
        self, group_id: str | None, limit: int
    ) -> list[sqlite3.Row]:
        """列出待处理待办（按登记时间正序）；group_id 为 None 时列出所有群。"""
        async with self._lock:
            if group_id is None:
                return self._conn.execute(
                    "SELECT * FROM todos WHERE status = 'pending' "
                    "ORDER BY id ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM todos WHERE group_id = ? AND status = 'pending' "
                "ORDER BY id ASC LIMIT ?",
                (group_id, limit),
            ).fetchall()

    async def finish_todo(self, record_id: int, group_id: str) -> bool:
        """把一条待办标记为已完成。"""
        async with self._lock:
            cur = self._conn.execute(
                "UPDATE todos SET status = 'done', done_at = ? "
                "WHERE id = ? AND group_id = ? AND status = 'pending'",
                (_fmt_time(_now_cst()), record_id, group_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    async def delete_todo(self, record_id: int, group_id: str) -> str | None:
        """删除一条待办，返回被删内容（未找到返回 None）。"""
        async with self._lock:
            row = self._conn.execute(
                "SELECT content FROM todos WHERE id = ? AND group_id = ?",
                (record_id, group_id),
            ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "DELETE FROM todos WHERE id = ? AND group_id = ?",
                (record_id, group_id),
            )
            self._conn.commit()
            return row["content"]

    # ---------- 群消息缓存 ----------

    async def cache_message(
        self,
        *,
        group_id: str,
        msg_id: str,
        sender_id: str,
        sender_name: str,
        content: str,
        ts: int,
    ) -> bool:
        """写入一条群消息缓存。同群同消息 ID 自动忽略（事件可能重复推送）。

        返回是否为新写入的消息。
        """
        if not msg_id:
            return False
        async with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO cached_messages "
                "(group_id, msg_id, sender_id, sender_name, content, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (group_id, msg_id, sender_id, sender_name, content, ts),
            )
            inserted = cur.rowcount > 0
            if inserted:
                self._prune_cache_locked(group_id, ts)
            self._conn.commit()
            return inserted

    def _prune_cache_locked(self, group_id: str, now_ts: int) -> None:
        """删除超龄消息，并把每群缓存条数控制在 CACHE_KEEP_COUNT 以内。"""
        expire_before = now_ts - CACHE_TTL_HOURS * 3600
        self._conn.execute(
            "DELETE FROM cached_messages WHERE group_id = ? AND ts < ?",
            (group_id, expire_before),
        )
        self._conn.execute(
            """
            DELETE FROM cached_messages
            WHERE group_id = ? AND id NOT IN (
                SELECT id FROM cached_messages
                WHERE group_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (group_id, group_id, CACHE_KEEP_COUNT),
        )

    async def get_cached(self, group_id: str, count: int) -> list[sqlite3.Row]:
        """读取该群最近 count 条缓存消息（按时间正序返回）。"""
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM cached_messages WHERE group_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (group_id, count),
            ).fetchall()
            return list(reversed(rows))

    async def get_cached_by_msg_id(self, group_id: str, msg_id: str) -> str | None:
        """按消息 ID 查缓存内容，用于解析引用消息。"""
        async with self._lock:
            row = self._conn.execute(
                "SELECT content FROM cached_messages "
                "WHERE group_id = ? AND msg_id = ? LIMIT 1",
                (group_id, msg_id),
            ).fetchone()
            return (row["content"] or "").strip() if row else None

    def close(self) -> None:
        self._conn.close()


class StudentAssistant(Star):
    """团学事务智能沟通小助手：未答收集 + 链接入库 + 沟通摘要 + 待办同步。"""

    SUMMARIZE_PROMPT = """你是学院团学事务智能沟通助手。下面是一段多方工作沟通记录（可能包含老师意见、各部门反馈、外院对接信息等）。

【沟通记录】（来源：{source}）
{transcript}

【任务】
1. 通读记录，整理成一份面向团学工作人员的结构化摘要；
2. 摘要按「关键要求」「待办事项」「其他信息」分节，用换行分隔；待办事项逐条列出，并标注截止时间与负责人（仅当记录中有）；
3. 只输出一个 JSON 对象，禁止输出任何解释、客套话或多余文字，格式如下：
{{
  "summary": "完整摘要文本，用\\n换行分节",
  "todos": [
    {{
      "content": "待办事项，一句话",
      "deadline": "记录中明确的截止时间，原样摘录（如 10-15 18:00）；没有则填空字符串",
      "assignee": "负责人或对接部门；没有则填空字符串"
    }}
  ]
}}

【要求】
- summary 里也要包含待办事项的文字描述，todos 是其中可执行项的结构化清单；
- 记录中没有的信息一律填空字符串，严禁编造；
- 拿不准是否算待办的不列入 todos；没有待办时 todos 为空数组。"""

    EXTRACT_TODO_PROMPT = """你是学院团学事务智能沟通助手。请从下面的消息中提取需要跟进的待办事项。

【消息内容】
{transcript}

【任务】只输出一个 JSON 对象，禁止输出任何解释或多余文字：
{{
  "todos": [
    {{
      "content": "待办事项，一句话",
      "deadline": "截止时间原文（如 10-15 18:00、本周五前）；没有则填空字符串",
      "assignee": "负责人或对接部门；没有则填空字符串"
    }}
  ]
}}

【要求】没有待办时输出 {{"todos": []}}；拿不准的不列入；严禁编造。"""

    SYNC_PROMPT = """你是学院团学事务智能沟通助手。请把下面的待办清单改写成一条即将发到群聊里的同步消息。

【待办清单】
{plain}

【同步对象】{audience}

【要求】
- 同步对象是老师时：语气正式礼貌、条理清晰，称呼用「您」，结尾可加一句恳请老师过目指示的话；
- 同步对象是同学或部门时：语气轻松直接，突出事项、负责人和截止时间；
- 保留全部待办事项与截止时间，不得遗漏、不得新增；
- 直接输出消息正文，不要任何解释、标题引号或代码块。"""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        db_path = os.path.join(_data_dir(), "assistant.db")
        self.store = AssistantStore(db_path)
        logger.info(f"[团学助手] 已加载，数据库: {db_path}")

    # ---------- 配置读取 ----------

    @property
    def kb_name(self) -> str:
        return str(self.config.get("kb_name", "")).strip()

    @property
    def allowed_groups(self) -> list[str]:
        return [str(x) for x in self.config.get("allowed_groups", []) if x]

    @property
    def question_keywords(self) -> list[str]:
        kws = [str(k) for k in self.config.get("question_keywords", []) if k]
        return kws or ["？", "?"]

    @property
    def list_limit(self) -> int:
        try:
            return max(5, min(100, int(self.config.get("list_limit", 20))))
        except (TypeError, ValueError):
            return 20

    @property
    def summary_count(self) -> int:
        try:
            return max(10, min(100, int(self.config.get("summary_count", 30))))
        except (TypeError, ValueError):
            return 30

    @property
    def auto_sync(self) -> bool:
        return bool(self.config.get("auto_sync", False))

    def _is_group_allowed(self, group_id: str) -> bool:
        """群白名单：未配置时所有群生效，配置后仅名单内的群生效。"""
        if not self.allowed_groups:
            return True
        return group_id in self.allowed_groups

    def _is_onebot(self, event: AstrMessageEvent) -> bool:
        return str(event.platform_meta.name) == "aiocqhttp"

    @staticmethod
    def _check_scope(event: AstrMessageEvent) -> tuple[str, str]:
        """确定事件归属：返回 (scope_id, "group"|"private")。

        群聊返回群号（与历史数据兼容）；私聊返回 private:<用户ID>。
        白名单校验用 _is_scope_allowed，私聊一律放行。
        """
        gid = str(event.get_group_id() or "").strip()
        if gid:
            return gid, "group"
        sid = str(event.get_sender_id() or "").strip()
        return f"private:{sid}", "private"

    def _is_scope_allowed(self, event: AstrMessageEvent) -> bool:
        """群聊校验白名单，私聊放行。"""
        gid = str(event.get_group_id() or "").strip()
        if not gid:
            return True
        return self._is_group_allowed(gid)

    # ---------- 未答收集（旁路监听，不影响原生问答） ----------

    @staticmethod
    def _kb_hit(req: ProviderRequest) -> bool:
        """判断本次 LLM 请求是否注入了知识库上下文。

        检查 prompt / system_prompt / contexts 中是否带有内置知识库
        检索结果的固定前缀。判定过程出错时按"命中"处理（宁可漏记，
        不可把闲聊误记成未答问题）。
        """
        try:
            blob = "\n".join(
                str(getattr(req, field, "") or "")
                for field in ("prompt", "system_prompt", "contexts")
            )
            return KB_CONTEXT_MARKER in blob
        except Exception:  # noqa: BLE001
            return True

    @filter.on_llm_request()
    async def inject_faq(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """LLM 请求钩子：命中已学会的问答时，把参考答案注入请求。

        注意：只做旁路增强，绝不覆盖知识库检索结果，也不阻止事件流转。
        """
        try:
            if not self._is_scope_allowed(event):
                return
            text = (event.message_str or "").strip()
            if len(text) < MIN_QUESTION_LEN or text.startswith("/"):
                return
            row = await self._match_faq(text)
            if not row:
                return
            ref = (
                "\n\n【历史解答参考】此前有人问过类似问题："
                f"「{row['question']}」，当时的标准解答如下，"
                "回答时请优先参考：\n" + row["answer"]
            )
            req.system_prompt = (getattr(req, "system_prompt", "") or "") + ref
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[团学助手] FAQ 注入异常: {e}")

    @filter.on_llm_request()
    async def collect_unanswered(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """LLM 请求钩子：知识库未命中且消息像提问时，记录为未解答问题。

        群聊与私聊均生效；命中已学会的问答（FAQ）的问题视为已覆盖，
        不再记录。记录成功后实时推送给已订阅的负责人会话。
        注意：这里只做旁路记录，绝不修改请求，也不阻止事件流转。
        """
        try:
            if not self._is_scope_allowed(event):
                return

            text = (event.message_str or "").strip()
            if len(text) < MIN_QUESTION_LEN or text.startswith("/"):
                return
            if not any(k in text for k in self.question_keywords):
                return
            if await self._match_faq(text):
                return  # 已学会的相似问题，视为能解答
            if self._kb_hit(req):
                return

            scope_id, scope_kind = self._check_scope(event)
            record_id = await self.store.add_unanswered(
                group_id=scope_id,
                sender_id=str(event.get_sender_id() or ""),
                sender_name=event.get_sender_name() or "",
                question=text,
                asked_at=_now_cst(),
            )
            if record_id:
                logger.info(
                    f"[团学助手] 记录未答问题 #{record_id}"
                    f"（{scope_id}）: {text[:50]}"
                )
                await self._push_unanswered(
                    record_id,
                    event.get_sender_name() or "",
                    text,
                    scope_id,
                    scope_kind,
                )
        except Exception as e:  # noqa: BLE001
            # 收集失败不能影响机器人的正常回答流程
            logger.debug(f"[团学助手] 未答收集异常: {e}")

    async def _match_faq(self, question: str) -> sqlite3.Row | None:
        """在已学会的问答中找相似问题：先精确匹配，再做 2-gram 相似度。"""
        q_norm = _normalize_text(question)
        if len(q_norm) < 2:
            return None
        rows = await self.store.list_faq()
        if not rows:
            return None
        grams_q = _bigrams(q_norm)
        best, best_sim = None, 0.0
        for r in rows:
            f_norm = _normalize_text(r["question"])
            if not f_norm:
                continue
            if q_norm == f_norm:
                return r
            sim = _jaccard(grams_q, _bigrams(f_norm))
            if sim > best_sim:
                best, best_sim = r, sim
        return best if best_sim >= FAQ_SIM_THRESHOLD else None

    async def _push_unanswered(
        self,
        record_id: int,
        sender_name: str,
        question: str,
        scope_id: str,
        scope_kind: str,
    ) -> None:
        """把新未答问题实时推送给已订阅的负责人会话。

        单个目标推送失败（如官方机器人主动消息配额不足）只记日志，
        不影响其他目标。
        """
        try:
            targets = await self.store.list_push_targets()
            if not targets:
                return
            source = "私聊" if scope_kind == "private" else f"群 {scope_id}"
            text = (
                f"📩 收到新的未答问题 #{record_id}（来源：{source}）\n"
                f"提问人：{sender_name or '未知'}\n"
                f"问题：{question}\n\n"
                f"解答后回复：/未答解决 {record_id} <标准答案>\n"
                f"（附上答案会被记住，下次相似问题可直接回答）\n"
                f"查看全部：/未答列表"
            )
            chain = MessageChain(chain=[Plain(text)])
            for umo in targets:
                try:
                    await self.context.send_message(umo, chain)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[团学助手] 推送未答问题到 {umo} 失败: {e}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[团学助手] 未答推送异常: {e}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def cache_group_message(self, event: AstrMessageEvent):
        """实时缓存群文本消息，供 /整理 在 QQ 官方机器人上回溯上下文。

        只做旁路缓存：不 stop_event、不回复。
        """
        try:
            group_id = str(event.get_group_id() or "").strip()
            if not group_id or not self._is_group_allowed(group_id):
                return
            text = (event.message_str or "").strip()
            if not text or text.startswith("/"):
                return
            try:
                ts = int(event.message_obj.timestamp)
            except (TypeError, ValueError):
                ts = int(_now_cst().timestamp())
            await self.store.cache_message(
                group_id=group_id,
                msg_id=str(event.message_obj.message_id),
                sender_id=str(event.get_sender_id() or ""),
                sender_name=event.get_sender_name() or "",
                content=text,
                ts=ts,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[团学助手] 消息缓存失败: {e}")

    # ---------- LLM 与内容解析辅助 ----------

    async def _get_provider_id(self, event: AstrMessageEvent) -> str | None:
        """获取 LLM 提供商 ID：当前会话配置优先，其次会话在用提供商，
        最后兜底取任意一个可用提供商。"""
        umo = event.unified_msg_origin
        try:
            pid = await self.context.get_current_chat_provider_id(umo=umo)
            if pid:
                return str(pid)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[团学助手] 获取当前会话提供商失败: {e}")
        try:
            provider = self.context.get_using_provider(umo)
            pid = getattr(provider, "id", None)
            if pid:
                return str(pid)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[团学助手] 获取在用提供商失败: {e}")
        try:
            for provider in self.context.get_all_providers():
                pid = getattr(provider, "id", None)
                if pid:
                    return str(pid)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[团学助手] 兜底获取提供商失败: {e}")
        return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """解析 LLM 返回的 JSON（容忍 ```json 代码块包裹）。"""
        text = re.sub(r"^```(?:json)?\s*", "", (text or "").strip())
        text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    async def _summarize(
        self, provider_id: str, transcript: str, source: str
    ) -> tuple[str | None, list[dict]]:
        """AI 整理沟通记录。返回 (摘要文本, 待办列表)；失败返回 (None, [])。"""
        prompt = self.SUMMARIZE_PROMPT.format(
            source=source, transcript=transcript[:8000]
        )
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id, prompt=prompt
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[团学助手] 摘要 LLM 调用失败: {e}")
            return None, []
        raw = (getattr(resp, "completion_text", "") or "").strip()
        data = self._parse_json(raw)
        if data is None:
            # JSON 解析失败时退化为纯文本摘要，不登记待办
            return (raw or None), []
        summary = str(data.get("summary", "") or "").strip()
        if not summary:
            summary = raw.strip() or None
        return summary, _parse_todos(data)

    async def _extract_todos(
        self, provider_id: str, transcript: str
    ) -> list[dict] | None:
        """AI 从引用内容提取待办；LLM 调用失败返回 None（区别于空列表）。"""
        prompt = self.EXTRACT_TODO_PROMPT.format(transcript=transcript[:8000])
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id, prompt=prompt
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[团学助手] 待办提取 LLM 调用失败: {e}")
            return None
        return _parse_todos(self._parse_json(resp.completion_text or ""))

    async def _register_todos(
        self,
        event: AstrMessageEvent,
        group_id: str,
        todos: list[dict],
        source: str,
    ) -> list[tuple[int, dict]]:
        """把 AI 提取的待办批量登记入库，返回 (编号, 待办) 列表。"""
        registered: list[tuple[int, dict]] = []
        now = _now_cst()
        for t in todos:
            record_id = await self.store.add_todo(
                group_id=group_id,
                content=t["content"],
                assignee=t["assignee"],
                deadline=t["deadline"],
                source=source,
                creator_name=event.get_sender_name() or "",
                created_at=now,
            )
            if record_id:
                registered.append((record_id, t))
        return registered

    @staticmethod
    def _format_todo_line(record_id: int, todo: dict) -> str:
        line = f"#{record_id} {todo['content']}"
        extra = []
        if todo["deadline"]:
            extra.append(f"截止 {todo['deadline']}")
        if todo["assignee"]:
            extra.append(f"负责 {todo['assignee']}")
        if extra:
            line += f"（{'｜'.join(extra)}）"
        return line

    # ---------- 引用消息与群聊回溯 ----------

    async def _get_quoted_text(self, event: AstrMessageEvent, group_id: str) -> str:
        """提取事件中引用（回复）的消息文本。

        三级回退：Reply 组件自带文本 -> OneBot get_msg 接口 -> 本地消息缓存。
        """
        chain = event.message_obj.message or []
        reply_seg = None
        for seg in chain:
            if isinstance(seg, Reply):
                reply_seg = seg
                break
        if reply_seg is not None:
            text = (getattr(reply_seg, "message_str", "") or "").strip()
            if not text:
                parts = [
                    s.text
                    for s in (getattr(reply_seg, "chain", None) or [])
                    if isinstance(s, Plain)
                ]
                text = "".join(parts).strip()
            if text:
                return text
            ref_id = str(getattr(reply_seg, "id", "") or "")
            if self._is_onebot(event) and ref_id:
                try:
                    ret = await event.bot.call_action(
                        action="get_msg", message_id=int(ref_id)
                    )
                    if isinstance(ret, dict) and isinstance(ret.get("data"), dict):
                        ret = ret["data"]
                    text = self._raw_msg_text(ret if isinstance(ret, dict) else {})
                    if text.strip():
                        return text.strip()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[团学助手] get_msg(引用消息) 失败: {e}")
            if ref_id:
                cached = await self.store.get_cached_by_msg_id(group_id, ref_id)
                if cached:
                    return cached
        return ""

    @staticmethod
    def _raw_msg_text(m: dict) -> str:
        """从 OneBot 消息对象中提取纯文本。"""
        parts: list[str] = []
        payload = m.get("message")
        if isinstance(payload, list):
            for seg in payload:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    parts.append(seg.get("data", {}).get("text", ""))
        text = "".join(parts).strip()
        return text or (m.get("raw_message") or "").strip()

    async def _build_recent_transcript(
        self, event: AstrMessageEvent, group_id: str, count: int
    ) -> str:
        """回溯近期群聊文本：OneBot 优先拉历史，失败或官方机器人走本地缓存。"""
        lines: list[str] = []
        if self._is_onebot(event):
            ret = None
            for gid_val in (
                int(group_id) if group_id.isdigit() else group_id,
                group_id,
            ):
                try:
                    ret = await event.bot.call_action(
                        action="get_group_msg_history",
                        group_id=gid_val,
                        count=count,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        f"[团学助手] get_group_msg_history(group_id={gid_val}) 失败: {e}"
                    )
                    ret = None
            if isinstance(ret, dict) and isinstance(ret.get("data"), dict):
                ret = ret["data"]
            messages = ret.get("messages", []) if isinstance(ret, dict) else []
            messages.sort(key=lambda m: m.get("time", 0))
            for m in messages:
                try:
                    content = self._raw_msg_text(m)
                    if not content:
                        continue
                    sender = m.get("sender", {})
                    name = (
                        sender.get("card")
                        or sender.get("nickname")
                        or str(sender.get("user_id", ""))
                    )
                    t = datetime.fromtimestamp(int(m.get("time", 0)), CST).strftime(
                        "%H:%M"
                    )
                    if len(content) > 600:
                        content = content[:600] + "……"
                    lines.append(f"[{t}] {name}：{content}")
                except Exception:  # noqa: BLE001
                    continue
        if not lines:
            rows = await self.store.get_cached(group_id, count)
            for r in rows:
                content = (r["content"] or "").strip()
                if not content:
                    continue
                t = datetime.fromtimestamp(int(r["ts"]), CST).strftime("%H:%M")
                if len(content) > 600:
                    content = content[:600] + "……"
                lines.append(f"[{t}] {r['sender_name'] or r['sender_id']}：{content}")
        return "\n".join(lines)

    # ---------- 管理员指令：未答收集 ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("未答列表")
    async def list_unanswered_cmd(self, event: AstrMessageEvent):
        """查看知识库无法解答的问题：/未答列表 [全部]（私聊默认全部）"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, scope_kind = self._check_scope(event)
        scope_all = (
            "全部" in (event.message_str or "") or scope_kind == "private"
        )
        rows = await self.store.list_pending_unanswered(
            None if scope_all else scope_id, self.list_limit
        )
        if not rows:
            yield event.plain_result("暂无待处理的未答问题，知识库覆盖良好。")
            return

        scope = "全部会话" if scope_all else "本群"
        lines = [f"未解答问题（{scope}，待处理 {len(rows)} 条）："]
        for r in rows:
            asked = (r["asked_at"] or "")[5:16]  # MM-DD HH:MM
            origin = "私聊" if r["group_id"].startswith("private:") else "群聊"
            lines.append(
                f"#{r['id']} [{asked}][{origin}] "
                f"{r['sender_name'] or r['sender_id']}：{r['question']}"
            )
        lines.append(
            "处理：/未答解决 <编号> [标准答案]；手动登记：/加入未答 <问题>"
        )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("未答解决")
    async def resolve_unanswered_cmd(self, event: AstrMessageEvent):
        """标记未答问题已处理并可学会答案：/未答解决 <编号> [标准答案]"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, scope_kind = self._check_scope(event)
        parts = (event.message_str or "").strip().split(maxsplit=2)
        if len(parts) < 2 or not parts[1].isdigit():
            yield event.plain_result(
                "用法：/未答解决 <编号> [标准答案]\n"
                "附上标准答案会被记住，下次相似问题直接回答。\n"
                "编号可用 /未答列表 查看。"
            )
            return
        record_id = int(parts[1])
        answer = parts[2].strip() if len(parts) > 2 else ""
        row = await self.store.resolve_unanswered(
            record_id, None if scope_kind == "private" else scope_id
        )
        if not row:
            yield event.plain_result(
                f"未找到待处理的 #{record_id}（群聊操作仅限本群记录，"
                f"跨群问题请在私聊中处理）。"
            )
            return
        if not answer:
            yield event.plain_result(f"已标记 #{record_id} 为已处理。")
            return
        faq_id, updated = await self.store.add_faq(
            scope_id=row["group_id"],
            question=row["question"],
            answer=answer,
            source_id=record_id,
            created_at=_now_cst(),
        )
        if updated:
            yield event.plain_result(
                f"✅ #{record_id} 已处理，问答 #{faq_id} 的答案已更新，"
                f"下次相似问题将自动参考。"
            )
        else:
            yield event.plain_result(
                f"✅ #{record_id} 已处理，并已学会问答 #{faq_id}：\n"
                f"问：{row['question']}\n"
                f"下次遇到相似问题将自动参考该解答。"
            )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("问答列表")
    async def list_faq_cmd(self, event: AstrMessageEvent):
        """查看已学会的问答：/问答列表"""
        event.stop_event()
        rows = await self.store.list_faq(self.list_limit)
        if not rows:
            yield event.plain_result(
                "还没有学会任何问答。用 /未答解决 <编号> <标准答案> 教机器人。"
            )
            return
        lines = [f"已学会的问答（共 {len(rows)} 条，相似问题自动参考）："]
        for r in rows:
            q = r["question"]
            a = r["answer"]
            if len(q) > 40:
                q = q[:40] + "……"
            if len(a) > 60:
                a = a[:60] + "……"
            lines.append(f"#{r['id']} 问：{q}\n　答：{a}")
        lines.append("删除：/删问答 <编号>")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("删问答")
    async def delete_faq_cmd(self, event: AstrMessageEvent):
        """删除一条已学会的问答：/删问答 <编号>"""
        event.stop_event()
        parts = (event.message_str or "").split()
        if len(parts) != 2 or not parts[1].isdigit():
            yield event.plain_result(
                "用法：/删问答 <编号>，编号可用 /问答列表 查看。"
            )
            return
        record_id = int(parts[1])
        question = await self.store.delete_faq(record_id)
        if question:
            yield event.plain_result(f"已删除问答 #{record_id}：{question}")
        else:
            yield event.plain_result(f"未找到 #{record_id} 的问答。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("加入未答")
    async def add_unanswered_cmd(self, event: AstrMessageEvent):
        """手动登记一条待补资料的问题：/加入未答 <问题文字>"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, _ = self._check_scope(event)
        parts = (event.message_str or "").strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            yield event.plain_result("用法：/加入未答 <问题文字>")
            return
        record_id = await self.store.add_unanswered(
            group_id=scope_id,
            sender_id=str(event.get_sender_id() or ""),
            sender_name=event.get_sender_name() or "",
            question=parts[1].strip(),
            asked_at=_now_cst(),
        )
        if record_id is None:
            yield event.plain_result("该问题已在待处理列表中，无需重复登记。")
        else:
            yield event.plain_result(
                f"已登记未答问题 #{record_id}。\n"
                f"订阅推送的负责人会收到提醒（/开启未答推送）。"
            )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("开启未答推送")
    async def enable_push_cmd(self, event: AstrMessageEvent):
        """订阅未答问题实时推送：/开启未答推送（推荐在私聊中使用）"""
        event.stop_event()
        added = await self.store.add_push_target(
            event.unified_msg_origin,
            str(event.platform_meta.name),
            _now_cst(),
        )
        if added:
            yield event.plain_result(
                "✅ 已订阅：学生提出机器人答不上的问题时，"
                "本会话将实时收到提醒。\n"
                "解答时回复 /未答解决 <编号> <标准答案> 即可让机器人学会。\n"
                "取消订阅：/关闭未答推送"
            )
        else:
            yield event.plain_result("本会话已订阅过未答推送，无需重复开启。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("关闭未答推送")
    async def disable_push_cmd(self, event: AstrMessageEvent):
        """取消订阅未答问题推送：/关闭未答推送"""
        event.stop_event()
        if await self.store.remove_push_target(event.unified_msg_origin):
            yield event.plain_result("已取消本会话的未答推送。")
        else:
            yield event.plain_result("本会话尚未订阅未答推送。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("加入知识库")
    async def ingest_url_cmd(self, event: AstrMessageEvent):
        """抓取网页内容加入知识库：/加入知识库 <网页链接>"""
        event.stop_event()
        parts = (event.message_str or "").split()
        if len(parts) != 2 or not re.match(r"^https?://", parts[1]):
            yield event.plain_result(
                "用法：/加入知识库 <网页链接>\n"
                "示例：/加入知识库 https://example.com/zhongce-xize"
            )
            return
        url = parts[1]

        if not self.kb_name:
            yield event.plain_result(
                "请先在 WebUI 插件配置中填写「知识库名称」并重载插件。"
            )
            return
        kb_mgr = getattr(self.context, "kb_manager", None)
        if kb_mgr is None:
            yield event.plain_result(
                "当前 AstrBot 版本没有内置知识库功能，请升级到 4.5.0 及以上。"
            )
            return
        helper = await kb_mgr.get_kb_by_name(self.kb_name)
        if not helper:
            yield event.plain_result(
                f"未找到知识库「{self.kb_name}」，"
                f"请先在 WebUI「知识库」页面创建，或用 /知识库状态 查看现有知识库。"
            )
            return

        yield event.plain_result("正在抓取网页并入库，请稍候……")
        try:
            doc = await kb_mgr.upload_from_url(helper.kb.kb_id, url)
            doc_name = str(getattr(doc, "doc_name", "") or url)
            yield event.plain_result(
                f"✅ 已加入知识库「{self.kb_name}」：《{doc_name}》\n"
                f"内容已切片并向量化，学生提问即可检索到。"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[团学助手] 链接入库失败: {e}")
            yield event.plain_result(
                f"入库失败：{e}\n请确认链接可公开访问（无需登录内网）。"
            )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("知识库状态")
    async def kb_status_cmd(self, event: AstrMessageEvent):
        """查看知识库配置状态：/知识库状态"""
        event.stop_event()
        kb_mgr = getattr(self.context, "kb_manager", None)
        if kb_mgr is None:
            yield event.plain_result(
                "当前 AstrBot 版本没有内置知识库功能，请升级到 4.5.0 及以上。"
            )
            return

        kbs = await kb_mgr.list_kbs()
        if not kbs:
            yield event.plain_result(
                "还没有任何知识库。请先在 WebUI「知识库」页面：\n"
                "1. 配置嵌入模型（推荐硅基流动 BAAI/bge-m3）\n"
                "2. 创建知识库，并在插件配置中填写知识库名称"
            )
            return
        names = "、".join(kb.kb_name for kb in kbs)
        if not self.kb_name:
            yield event.plain_result(
                f"现有知识库：{names}\n"
                f"请在插件配置中填写「知识库名称」后重载插件。"
            )
            return
        helper = await kb_mgr.get_kb_by_name(self.kb_name)
        if not helper:
            yield event.plain_result(
                f"未找到知识库「{self.kb_name}」。现有知识库：{names}\n"
                f"请核对插件配置中的知识库名称是否完全一致。"
            )
            return

        kb = helper.kb
        err = getattr(helper, "init_error", None)
        status = f"不可用：{err}" if err else "可用"
        yield event.plain_result(
            f"知识库「{kb.kb_name}」\n"
            f"描述：{kb.description or '—'}\n"
            f"嵌入模型：{kb.embedding_provider_id or '—'}\n"
            f"切片大小：{kb.chunk_size}（重叠 {kb.chunk_overlap}）\n"
            f"召回条数：{kb.top_m_final}\n"
            f"状态：{status}\n\n"
            f"提示：还需在 WebUI「配置文件 → AI 配置 → 能力 → 知识库」中"
            f"勾选该知识库，学生提问才会走知识库回答。"
        )

    # ---------- 管理员指令：沟通摘要 ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("整理")
    async def summarize_cmd(self, event: AstrMessageEvent):
        """AI 整理沟通记录：引用消息发 /整理，群聊也可直接 /整理 回溯最近群聊"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, scope_kind = self._check_scope(event)
        quoted = await self._get_quoted_text(event, scope_id)
        if quoted:
            transcript, source = quoted, "引用消息"
        elif scope_kind == "private":
            yield event.plain_result(
                "私聊整理仅支持引用消息：请先引用要分析的消息，再发送 /整理。"
            )
            return
        else:
            transcript = await self._build_recent_transcript(
                event, scope_id, self.summary_count
            )
            source = f"最近群聊（≤{self.summary_count} 条）"
            if not transcript:
                yield event.plain_result(
                    "没有可整理的内容：既没有引用消息，也没有可回溯的群聊记录。"
                )
                return

        provider_id = await self._get_provider_id(event)
        if not provider_id:
            yield event.plain_result(
                "没有可用的 LLM 提供商，请先在管理面板配置大模型。"
            )
            return

        yield event.plain_result("正在整理沟通内容，请稍候……")
        summary, todos = await self._summarize(provider_id, transcript, source)
        if not summary:
            yield event.plain_result("AI 整理失败，请稍后重试或查看日志。")
            return

        registered = await self._register_todos(event, scope_id, todos, "summary")

        text = f"📋 沟通摘要（来源：{source}）\n\n{summary}"
        if registered:
            lines = ["", f"—— 已自动登记 {len(registered)} 条待办 ——"]
            for record_id, todo in registered:
                lines.append(self._format_todo_line(record_id, todo))
            lines.append("查看：/待办列表 ｜ 同步：/同步待办")
            text += "\n" + "\n".join(lines)
        yield event.plain_result(text)

        if registered and self.auto_sync:
            sync_msg = await self._build_sync(event, scope_id, "同学")
            if sync_msg:
                yield event.plain_result(sync_msg)

    # ---------- 管理员指令：待办清单 ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("记待办")
    async def add_todo_cmd(self, event: AstrMessageEvent):
        """登记待办：/记待办 <事项文字>；或引用消息发 /记待办 由 AI 提取"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, _ = self._check_scope(event)
        arg = ""
        parts = (event.message_str or "").strip().split(maxsplit=1)
        if len(parts) > 1:
            arg = parts[1].strip()
        quoted = await self._get_quoted_text(event, scope_id)

        if not arg and not quoted:
            yield event.plain_result(
                "用法：/记待办 <事项文字>\n"
                "或：引用一条含任务安排的消息后发送 /记待办，由 AI 自动提取。"
            )
            return

        # 引用消息：AI 提取（可能得到多条待办）
        if quoted and not arg:
            provider_id = await self._get_provider_id(event)
            if not provider_id:
                yield event.plain_result(
                    "没有可用的 LLM 提供商，请先在管理面板配置大模型。"
                )
                return
            yield event.plain_result("正在从引用内容中提取待办，请稍候……")
            todos = await self._extract_todos(provider_id, quoted)
            if todos is None:
                yield event.plain_result("AI 提取失败，请稍后重试或查看日志。")
                return
            if not todos:
                yield event.plain_result("引用内容中没有识别到待办事项。")
                return
            registered = await self._register_todos(
                event, scope_id, todos, "quoted"
            )
            if not registered:
                yield event.plain_result("提取到的待办已在清单中，未重复登记。")
                return
            lines = [f"✅ 已登记 {len(registered)} 条待办："]
            for record_id, todo in registered:
                lines.append(self._format_todo_line(record_id, todo))
            lines.append("查看：/待办列表 ｜ 同步：/同步待办")
            yield event.plain_result("\n".join(lines))
            if self.auto_sync:
                sync_msg = await self._build_sync(event, scope_id, "同学")
                if sync_msg:
                    yield event.plain_result(sync_msg)
            return

        # 直接文字登记
        record_id = await self.store.add_todo(
            group_id=scope_id,
            content=arg,
            source="manual",
            creator_name=event.get_sender_name() or "",
            created_at=_now_cst(),
        )
        if record_id is None:
            yield event.plain_result("该待办已在清单中，未重复登记。")
            return
        yield event.plain_result(f"✅ 已登记待办 #{record_id}：{arg}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("待办列表")
    async def list_todo_cmd(self, event: AstrMessageEvent):
        """查看待办清单：/待办列表 [全部]（私聊默认本会话）"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, scope_kind = self._check_scope(event)
        scope_all = "全部" in (event.message_str or "")
        rows = await self.store.list_pending_todos(
            None if scope_all else scope_id, self.list_limit
        )
        if not rows:
            yield event.plain_result("当前没有待处理的待办事项。")
            return
        scope = "全部会话" if scope_all else (
            "私聊" if scope_kind == "private" else "本群"
        )
        lines = [f"待办清单（{scope}，共 {len(rows)} 项）："]
        for r in rows:
            extra = []
            if r["deadline"]:
                extra.append(f"截止 {r['deadline']}")
            if r["assignee"]:
                extra.append(f"负责 {r['assignee']}")
            suffix = f"（{'｜'.join(extra)}）" if extra else ""
            lines.append(f"#{r['id']} {r['content']}{suffix}")
        lines.append(
            "完成：/完成待办 <编号> ｜ 删除：/删待办 <编号> ｜ 同步：/同步待办"
        )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("完成待办")
    async def finish_todo_cmd(self, event: AstrMessageEvent):
        """把一条待办标记为已完成：/完成待办 <编号>"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, _ = self._check_scope(event)
        parts = (event.message_str or "").split()
        if len(parts) != 2 or not parts[1].isdigit():
            yield event.plain_result(
                "用法：/完成待办 <编号>，编号可用 /待办列表 查看。"
            )
            return
        record_id = int(parts[1])
        if await self.store.finish_todo(record_id, scope_id):
            yield event.plain_result(f"✅ 待办 #{record_id} 已完成。")
        else:
            yield event.plain_result(f"本会话未找到待处理的 #{record_id}。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("删待办")
    async def delete_todo_cmd(self, event: AstrMessageEvent):
        """删除一条待办：/删待办 <编号>"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, _ = self._check_scope(event)
        parts = (event.message_str or "").split()
        if len(parts) != 2 or not parts[1].isdigit():
            yield event.plain_result(
                "用法：/删待办 <编号>，编号可用 /待办列表 查看。"
            )
            return
        record_id = int(parts[1])
        content = await self.store.delete_todo(record_id, scope_id)
        if content:
            yield event.plain_result(f"已删除待办 #{record_id}：{content}")
        else:
            yield event.plain_result(f"本会话未找到 #{record_id} 的待办。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.command("同步待办")
    async def sync_todo_cmd(self, event: AstrMessageEvent):
        """把待办清单按对象同步播报到群里：/同步待办 [老师]"""
        event.stop_event()
        if not self._is_scope_allowed(event):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_id, _ = self._check_scope(event)
        parts = (event.message_str or "").split()
        audience = (
            "老师" if any("老师" in p for p in parts[1:]) else "同学"
        )
        msg = await self._build_sync(event, scope_id, audience)
        if not msg:
            yield event.plain_result("当前没有待同步的待办事项。")
            return
        yield event.plain_result(msg)

    async def _build_sync(
        self, event: AstrMessageEvent, group_id: str, audience: str
    ) -> str | None:
        """生成同步消息：先取待办清单，AI 按对象改写语气，失败退回纯文本。"""
        rows = await self.store.list_pending_todos(group_id, SYNC_TODO_LIMIT)
        if not rows:
            return None
        lines = [f"【待办同步】共 {len(rows)} 项"]
        for r in rows:
            extra = []
            if r["deadline"]:
                extra.append(f"截止 {r['deadline']}")
            if r["assignee"]:
                extra.append(f"负责 {r['assignee']}")
            suffix = f"（{'｜'.join(extra)}）" if extra else ""
            lines.append(f"{r['id']}. {r['content']}{suffix}")
        plain = "\n".join(lines)

        provider_id = await self._get_provider_id(event)
        if provider_id:
            try:
                resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=self.SYNC_PROMPT.format(plain=plain, audience=audience),
                )
                text = (getattr(resp, "completion_text", "") or "").strip()
                if text:
                    return text
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[团学助手] 同步语气改写失败，使用纯文本: {e}")
        return plain

    async def terminate(self):
        """插件卸载/停用时释放资源。"""
        self.store.close()
