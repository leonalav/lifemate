"""The prompt builder.

Full per-sample installed schemas are affordable: installed tool schemas are ~3.1k tokens
mean / 6.0k max and the full prompt ~3.9k mean / 7.4k max, against a 16,384-token cap
(docs/VERIFIED_FACTS.md §12.7). Hard tool retrieval is therefore not structurally necessary,
and a competitor measured retrieval variants *lowering* their public score — so every
installed schema is shown, in full (docs/DECISIONS.md, Decision 14).

The system policy encodes the labelling rules the metric actually rewards:

* out-of-scope request -> emit nothing and finish (7.2% of samples; any call scores 0)
* only fill an optional argument the user actually stated (27.3% of gold calls have none)
* read before write to obtain an id (55.1% of samples need a grounded id)
* copy fuzzy free text from the user's own wording
* stop after sys.ask_missing_info (14.0% of gold, always the last call)
"""
from __future__ import annotations

import json
from datetime import datetime

from ..data import Split, installed_tools
from ..text import relative_day_table

SYSTEM_POLICY = """\
Bạn là trợ lý điều khiển thiết bị và ứng dụng của người dùng.
Mỗi lượt bạn chỉ được thực hiện MỘT hành động, rồi hệ thống sẽ báo lại kết quả.

Quy tắc bắt buộc:
1. Nếu yêu cầu không liên quan tới bất kỳ app nào được cài đặt, hãy trả lời {"name": "finish"} và không gọi tool nào.
2. Chỉ truyền đúng các tham số mà người dùng đã nói. Không tự thêm tham số tuỳ chọn, không bịa giá trị.
3. Nếu cần một id (mã thiết bị, id sự kiện, id liên hệ...) mà chưa biết, hãy gọi tool đọc trước rồi mới dùng id đó.
4. Với tham số dạng văn bản tự do (tên, nội dung, ghi chú, địa chỉ...), hãy dùng đúng lời của người dùng, không diễn giải lại.
5. Với tham số số (nhiệt độ, tiền, phần trăm, số lượng), hãy dùng định dạng số thuần: 26, 2000000, 30.
6. Với tham số thời gian, dùng định dạng ISO 8601 có múi giờ: 2026-11-29T09:00:00+07:00. Dùng bảng ngày tương đối bên dưới, đừng tự tính ngày. Nếu người dùng nói "hôm đó", "hôm trước", "ngày đó" thì hiểu theo ngữ cảnh cuộc trò chuyện đã nói trước đó.
7. Nếu thiếu thông tin không thể suy ra, gọi sys.ask_missing_info với đúng hai tham số tool_name và params, rồi kết thúc lượt.
8. Trả lời đúng một đối tượng JSON: {"name": "<tên tool>", "arguments": {...}} hoặc {"name": "finish"}.

Lưu ý về tham số bắt buộc: nếu thiếu tham số bắt buộc thì lời gọi sẽ bị từ chối. Hãy đọc trước để lấy đủ thông tin."""

FINISH = "finish"


def _tool_block(split: Split, tool_names: list[str]) -> str:
    """Render the installed tools as a compact JSON schema list."""
    by_name = {
        tool["name"]: tool
        for app in split.catalog().get("apps", []) or []
        for tool in app.get("tools", []) or []
    }
    lines = []
    for name in tool_names:
        tool = by_name.get(name)
        if tool is None:
            continue
        spec = {
            "name": tool["name"],
            "kind": tool.get("kind"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters") or {},
        }
        if tool.get("returns"):
            spec["returns"] = tool["returns"]
        lines.append(json.dumps(spec, ensure_ascii=False))
    return "\n".join(lines)


def _history_block(sample: dict, transcript_for_history=None) -> str:
    """Compact the conversation history: text, the calls made, and trimmed results."""
    history = sample.get("history") or []
    if not history:
        return "(không có lịch sử)"
    lines = []
    for turn in history:
        role = turn.get("role")
        if role == "user":
            text = turn.get("audio_text")
            if text is None and transcript_for_history is not None:
                text = transcript_for_history(turn)
            lines.append(f"Người dùng: {text or '(có âm thanh)'}")
        elif role == "assistant":
            lines.append(f"Trợ lý: {turn.get('text') or ''}")
            for call in turn.get("tool_calls") or []:
                args = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
                lines.append(f"  -> {call.get('name')}({args})")
            for result in (turn.get("tool_results") or [])[:2]:
                text = json.dumps(result, ensure_ascii=False)
                if len(text) > 400:
                    text = text[:400] + "..."
                lines.append(f"  <- {text}")
    return "\n".join(lines)


def build_prompt(split: Split, sample: dict, transcript: str,
                 transcript_for_history=None, now: datetime | None = None) -> str:
    """The full prompt for one model step."""
    context = sample.get("context") or {}
    tools = installed_tools(sample)
    if now is None:
        current = context.get("current_time")
        now = datetime.fromisoformat(current) if current else datetime.now().astimezone()

    parts = [
        SYSTEM_POLICY,
        "",
        "## Ngữ cảnh",
        f"- Thời gian hiện tại: {context.get('current_time')}",
        f"- Múi giờ: {context.get('timezone')}",
        f"- Người dùng: {json.dumps(context.get('user_profile') or {}, ensure_ascii=False)}",
        f"- Vị trí: {json.dumps(context.get('location') or {}, ensure_ascii=False)}",
        "",
        "## Bảng ngày tương đối",
        relative_day_table(now),
        "",
        "## Lịch sử",
        _history_block(sample, transcript_for_history),
        "",
        "## Yêu cầu hiện tại của người dùng",
        transcript,
        "",
        "## Các tool được cài đặt",
        _tool_block(split, tools),
        "",
        "## Hành động tiếp theo",
        '{"name": "<tool>", "arguments": {...}}',
    ]
    return "\n".join(parts)


def build_action_schema(split: Split, sample: dict) -> dict:
    """The JSON schema the model output is constrained to.

    ``anyOf`` over every installed tool plus ``finish``: one object per step, so every
    dispatched call is independently attributable and independently fixable
    (docs/DECISIONS.md, Decision 4).
    """
    tools = installed_tools(sample)
    by_name = {
        tool["name"]: tool
        for app in split.catalog().get("apps", []) or []
        for tool in app.get("tools", []) or []
    }
    variants = [{
        "type": "object",
        "properties": {
            "name": {"const": "finish"},
            "arguments": {"type": "object"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }]
    for name in tools:
        tool = by_name.get(name)
        if tool is None:
            continue
        variants.append({
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": tool.get("parameters") or {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return {"anyOf": variants}
