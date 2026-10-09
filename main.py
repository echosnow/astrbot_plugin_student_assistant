"""团学事务智能沟通小助手（astrbot_plugin_student_assistant）。

设计定位：学生自助问答由 AstrBot 原生知识库（RAG）完成，本插件只补齐
原生能力做不到的业务闭环，以"旁路监听"方式工作，不干扰原生问答体验：

1. 未答收集 —— 挂载 LLM 请求钩子，检查本次请求是否注入了知识库上下文。
   未注入（即知识库没答上）且消息像提问时，自动记录到数据库，供管理员
   通过 /未答列表 汇总处理，避免"机器人答不上来的问题石沉大海"。
2. 链接入库 —— 管理员在群内发送 /加入知识库 <网页链接>，自动抓取网页
   内容写入配置的知识库，无需打开 WebUI 上传文件。
3. 管理指令 —— /未答列表、/未答解决、/加入未答、/知识库状态。
"""

import asyncio
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

PLUGIN_NAME = "astrbot_plugin_student_assistant"

# 统一使用东八区时间，避免 Docker 容器时区不一致
CST = timezone(timedelta(hours=8))

# 知识库命中标记：AstrBot 内置知识库检索到内容后，会把带该固定前缀的
# 上下文注入 LLM 请求（见 astrbot/core/knowledge_base/kb_mgr.py 的
# _format_context）。据此判断本次回答是否基于知识库。
KB_CONTEXT_MARKER = "以下是相关的知识库内容"

# 同群同一问题在该小时数内不重复记录
DEDUP_WINDOW_HOURS = 24

# 单个问题最长保留字符数
MAX_QUESTION_LEN = 500

# 判定"像提问"的最短消息长度（过滤寒暄与误触）
MIN_QUESTION_LEN = 4


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


def _normalize_question(text: str) -> str:
    """问题去重规范化：去掉空白与常用中英文标点，转小写。"""
    return re.sub(
        r"[\s，。！？?!\.,:：;；、~～'\"“”‘’()（）【】\[\]{}]+", "", text
    ).lower()


class UnansweredStore:
    """未答问题的 SQLite 存储层。写入量很小，单连接 + 异步锁即可。"""

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
        self._conn.commit()

    async def add(
        self,
        *,
        group_id: str,
        sender_id: str,
        sender_name: str,
        question: str,
        asked_at: datetime,
    ) -> int | None:
        """记录一条未答问题。同群同问题在去重窗口内已存在时返回 None。"""
        question = question.strip()[:MAX_QUESTION_LEN]
        q_norm = _normalize_question(question)
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

    async def list_pending(
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

    async def resolve(self, record_id: int, group_id: str) -> bool:
        """把一条未答问题标记为已处理。"""
        async with self._lock:
            cur = self._conn.execute(
                "UPDATE unanswered SET status = 'resolved', resolved_at = ? "
                "WHERE id = ? AND group_id = ? AND status = 'pending'",
                (_fmt_time(_now_cst()), record_id, group_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


class StudentAssistant(Star):
    """团学事务智能沟通小助手：未答收集 + 知识库链接入库。"""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        db_path = os.path.join(_data_dir(), "assistant.db")
        self.store = UnansweredStore(db_path)
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

    def _is_group_allowed(self, group_id: str) -> bool:
        """群白名单：未配置时所有群生效，配置后仅名单内的群生效。"""
        if not self.allowed_groups:
            return True
        return group_id in self.allowed_groups

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
    async def collect_unanswered(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """LLM 请求钩子：知识库未命中且消息像提问时，记录为未解答问题。

        注意：这里只做旁路记录，绝不修改请求，也不阻止事件流转。
        """
        try:
            group_id = str(event.get_group_id() or "").strip()
            if not group_id or not self._is_group_allowed(group_id):
                return

            text = (event.message_str or "").strip()
            if len(text) < MIN_QUESTION_LEN or text.startswith("/"):
                return
            if not any(k in text for k in self.question_keywords):
                return
            if self._kb_hit(req):
                return

            record_id = await self.store.add(
                group_id=group_id,
                sender_id=str(event.get_sender_id() or ""),
                sender_name=event.get_sender_name() or "",
                question=text,
                asked_at=_now_cst(),
            )
            if record_id:
                logger.info(
                    f"[团学助手] 记录未答问题 #{record_id}"
                    f"（群 {group_id}）: {text[:50]}"
                )
        except Exception as e:  # noqa: BLE001
            # 收集失败不能影响机器人的正常回答流程
            logger.debug(f"[团学助手] 未答收集异常: {e}")

    # ---------- 管理员指令 ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    @filter.command("未答列表")
    async def list_unanswered_cmd(self, event: AstrMessageEvent):
        """查看知识库无法解答的问题：/未答列表 [全部]"""
        event.stop_event()
        group_id = str(event.get_group_id() or "").strip()
        if not self._is_group_allowed(group_id):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        scope_all = "全部" in (event.message_str or "")
        rows = await self.store.list_pending(
            None if scope_all else group_id, self.list_limit
        )
        if not rows:
            yield event.plain_result("暂无待处理的未答问题，知识库覆盖良好。")
            return

        scope = "全部群" if scope_all else "本群"
        lines = [f"未解答问题（{scope}，待处理 {len(rows)} 条）："]
        for r in rows:
            asked = (r["asked_at"] or "")[5:16]  # MM-DD HH:MM
            lines.append(
                f"#{r['id']} [{asked}] "
                f"{r['sender_name'] or r['sender_id']}：{r['question']}"
            )
        lines.append("处理：/未答解决 <编号>；手动登记：/加入未答 <问题>")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    @filter.command("未答解决")
    async def resolve_unanswered_cmd(self, event: AstrMessageEvent):
        """标记一条未答问题为已处理：/未答解决 <编号>"""
        event.stop_event()
        group_id = str(event.get_group_id() or "").strip()
        if not self._is_group_allowed(group_id):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        parts = (event.message_str or "").split()
        if len(parts) != 2 or not parts[1].isdigit():
            yield event.plain_result(
                "用法：/未答解决 <编号>，编号可用 /未答列表 查看。"
            )
            return
        record_id = int(parts[1])
        if await self.store.resolve(record_id, group_id):
            yield event.plain_result(f"已标记 #{record_id} 为已处理。")
        else:
            yield event.plain_result(f"本群未找到待处理的 #{record_id}。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    @filter.command("加入未答")
    async def add_unanswered_cmd(self, event: AstrMessageEvent):
        """手动登记一条待补资料的问题：/加入未答 <问题文字>"""
        event.stop_event()
        group_id = str(event.get_group_id() or "").strip()
        if not self._is_group_allowed(group_id):
            yield event.plain_result("本群未启用团学助手功能。")
            return

        parts = (event.message_str or "").strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            yield event.plain_result("用法：/加入未答 <问题文字>")
            return
        record_id = await self.store.add(
            group_id=group_id,
            sender_id=str(event.get_sender_id() or ""),
            sender_name=event.get_sender_name() or "",
            question=parts[1].strip(),
            asked_at=_now_cst(),
        )
        if record_id is None:
            yield event.plain_result("该问题已在待处理列表中，无需重复登记。")
        else:
            yield event.plain_result(f"已登记未答问题 #{record_id}。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
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
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
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

    async def terminate(self):
        """插件卸载/停用时释放资源。"""
        self.store.close()
