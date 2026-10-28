# Copyright © 2026 Apple Inc.

# The DeepSeek-V4.1 prompt format, ported from encoding/encoding.py of the
# checkpoint (MIT License, Copyright (c) 2023 DeepSeek). Text only: image blocks
# render as unsupported content.

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

bos_token = "<｜begin▁of▁sentence｜>"
eos_token = "<｜end▁of▁sentence｜>"
thinking_start_token = "<think>"
thinking_end_token = "</think>"
dsml_token = "｜DSML｜"

SYSTEM_SP_TOKEN = "<｜System｜>"
USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

# Special tokens of internal classification tasks.
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}

response_format_template = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to "
    "reply:\n{schema}"
)
tool_output_template = "<tool_result>{content}</tool_result>"

tool_calls_block_name = " calls"
tool_call_tag_name = " invoke"
tool_parameter_tag_name = " parameter"
tool_call_template = (
    '<{dsml_token}{tool_call_tag_name} name="{name}">\n{arguments}\n'
    "</{dsml_token}{tool_call_tag_name}>"
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
DEFAULT_REASONING_EFFORT = "high"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, UnicodeEncodeError):
        return json.dumps(value, ensure_ascii=True)


def _split_tool_name(
    name: str, namespace: Optional[str] = None
) -> Tuple[Optional[str], str]:
    prefix, separator, bare_name = name.partition("::")
    if separator:
        if namespace not in (None, prefix):
            raise ValueError(f"Conflicting tool namespaces: {namespace} != {prefix}")
        namespace, name = prefix, bare_name
    if "::" in name or (namespace is not None and "::" in namespace):
        raise ValueError(f"Tool name or namespace must not contain '::': {name}")
    return namespace, name


def _tool_name_for_encoding(tool: Dict[str, Any]) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    namespace, name = _split_tool_name(tool["name"], namespace)
    return name if namespace is None else f"{namespace}::{name}"


def tools_from_openai_format(tools):
    functions = []
    for tool in tools:
        function = dict(tool["function"])
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = (
                namespace["description"] + "\n" + (function.get("description") or "")
            )
        functions.append(function)
    return functions


def tool_calls_from_openai_format(tool_calls):
    calls = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        namespace, name = _split_tool_name(
            function["name"], tool_call.get("namespace") or function.get("namespace")
        )
        call = {"name": name, "arguments": function["arguments"]}
        if namespace is not None:
            call["namespace"] = namespace
        calls.append(call)
    return calls


def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put tool results in user messages: V4.1 has no tool role."""
    merged = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")
        if role == "tool":
            content = msg.get("content", "")
            if msg.get("content_blocks") and not isinstance(msg.get("content"), str):
                content = msg["content_blocks"]
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            blocks = msg.get("content_blocks")
            if blocks is None:
                content = msg.get("content", "")
                blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].extend(blocks)
            else:
                msg["content_blocks"] = blocks
                merged.append(msg)
        else:
            merged.append(msg)
    return merged


def sort_tool_results_by_call_order(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order the tool results of a user message as the calls before it."""
    order = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            order = {}
            for idx, call in enumerate(msg["tool_calls"]):
                call_id = call.get("id") or call.get("function", {}).get("id", "")
                if call_id:
                    order[call_id] = idx
        elif role == "user" and msg.get("content_blocks"):
            results = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(results) > 1 and order:
                results = iter(
                    sorted(
                        results, key=lambda b: order.get(b.get("tool_use_id", ""), 0)
                    )
                )
                msg["content_blocks"] = [
                    next(results) if b.get("type") == "tool_result" else b
                    for b in msg["content_blocks"]
                ]
    return messages


def render_reasoning_effort(
    index: int, thinking_mode: str, effort: Union[str, int, None]
) -> str:
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    valid_int = type(effort) is int and 1 <= effort <= 100
    if not valid_int and effort not in REASONING_EFFORT_MAPPINGS:
        raise ValueError(
            f"Invalid reasoning effort {effort!r}: use an int in [1, 100] or one of "
            f"{list(REASONING_EFFORT_MAPPINGS)}"
        )
    if type(effort) is str:
        effort = REASONING_EFFORT_MAPPINGS[effort]
    if index == 0 and thinking_mode == "thinking":
        return REASONING_EFFORT_TEMPLATE.format(budget=effort)
    return ""


def render_tools(tools: List[Dict[str, Any]]) -> str:
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(to_json(t) for t in tools),
        dsml_token=dsml_token,
        tc_block_name=tool_calls_block_name,
        tool_call_tag_name=tool_call_tag_name,
        tool_parameter_tag_name=tool_parameter_tag_name,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def encode_arguments_to_dsml(tool_call: Dict[str, Any]) -> str:
    template = (
        '<{dsml_token}{tag} name="{key}" string="{is_str}">{value}</{dsml_token}{tag}>'
    )
    arguments = tool_call["arguments"]
    if not isinstance(arguments, dict):
        # JSON strings, also double-encoded ones.
        for _ in range(2):
            if not isinstance(arguments, str):
                break
            try:
                arguments = json.loads(arguments)
            except ValueError:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call["arguments"]}
    return "\n".join(
        template.format(
            dsml_token=dsml_token,
            tag=tool_parameter_tag_name,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )
        for k, v in arguments.items()
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """The last user turn. Mid-conversation system messages count as one, and a
    message of only tool results does not."""
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "user" or (role == "system" and idx > 0):
            blocks = messages[idx].get("content_blocks")
            if blocks and not any(
                b.get("type") in ("text", "image", "image_url") for b in blocks
            ):
                continue
            return idx
    return -1


def _render_user_blocks(blocks: List[Dict[str, Any]]) -> str:
    parts = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                content = "\n\n".join(
                    (
                        b.get("text", "")
                        if b.get("type") == "text"
                        else f"[Unsupported {b.get('type')}]"
                    )
                    for b in content
                )
            parts.append(tool_output_template.format(content=content))
        else:
            parts.append(f"[Unsupported {kind}]")
    return "\n\n".join(parts)


def render_message(
    index: int,
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode {thinking_mode!r}")
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)
    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    tool_calls = msg.get("tool_calls")
    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    effort = render_reasoning_effort(index, thinking_mode, reasoning_effort)
    # The system token leads when there is an effort prefix or a first system message.
    prompt = SYSTEM_SP_TOKEN if index == 0 and (effort or role == "system") else ""
    prompt += effort

    if role == "system":
        if index > 0:
            prompt += SYSTEM_SP_TOKEN
        prompt += content or ""
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if msg.get("response_format"):
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(msg["response_format"])
            )
    elif role == "user":
        prompt += USER_SP_TOKEN
        blocks = msg.get("content_blocks")
        prompt += _render_user_blocks(blocks) if blocks else (content or "")
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + content
    elif role == "assistant":
        calls = ""
        if tool_calls:
            rendered = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    tool_call_tag_name=tool_call_tag_name,
                    name=_tool_name_for_encoding(tc),
                    arguments=encode_arguments_to_dsml(tc),
                )
                for tc in tool_calls
            ]
            calls = "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(rendered),
                tc_block_name=tool_calls_block_name,
            )
        thinking = ""
        # The output of a task has no thinking.
        prev_has_task = index > 0 and messages[index - 1].get("task") is not None
        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking = (msg.get("reasoning_content") or "") + thinking_end_token
        prompt += thinking + (content or "") + calls
        if not msg.get("wo_eos", False):
            prompt += eos_token
    else:
        raise ValueError(f"Unknown role {role!r}")

    following = messages[index + 1].get("role") if index + 1 < len(messages) else None
    if following is not None and following not in ("assistant", "latest_reminder"):
        return prompt

    task = msg.get("task")
    if task is not None:
        if task not in DS_TASK_SP_TOKENS:
            raise ValueError(f"Invalid task {task!r}")
        if task != "action":
            prompt += DS_TASK_SP_TOKENS[task]
        else:
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                thinking_start_token
                if thinking_mode == "thinking"
                else thinking_end_token
            )
            prompt += DS_TASK_SP_TOKENS[task]
    elif role == "user" or (role == "system" and index > 0):
        prompt += ASSISTANT_SP_TOKEN
        if thinking_mode == "thinking" and (
            not drop_thinking or index >= last_user_idx
        ):
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token
    return prompt


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the reasoning of assistant turns before the last user turn."""
    last_user_idx = find_last_user_index(messages)
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
    result = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)
    return result


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str = "thinking",
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    context = merge_tool_messages(context or [])
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    full_messages = context + messages
    prompt = bos_token if add_default_bos_token and not context else ""

    # Turns with tools keep their thinking.
    if any(m.get("tools") for m in full_messages):
        drop_thinking = False
    if thinking_mode == "thinking" and drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        count = len(full_messages) - len(_drop_thinking_messages(context))
        start = len(full_messages) - count
    else:
        count, start = len(messages), len(context)

    for idx in range(count):
        prompt += render_message(
            idx + start,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            reasoning_effort=reasoning_effort,
        )
    return prompt


def apply_chat_template(
    messages,
    tools=None,
    continue_final_message=False,
    add_generation_prompt=False,
    **kwargs,
):
    if continue_final_message and add_generation_prompt:
        raise ValueError(
            "Only one of continue_final_message or add_generation_prompt can be True"
        )
    thinking = kwargs.pop("enable_thinking", True)
    thinking_mode = kwargs.pop("thinking_mode", "thinking" if thinking else "chat")
    messages = [dict(m) for m in messages]
    if tools:
        # Tools are declared in the first message, which renders them as system.
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = tools
    if continue_final_message and messages[-1]["role"] == "assistant":
        messages[-1]["wo_eos"] = True
    out = encode_messages(
        messages,
        thinking_mode=thinking_mode,
        reasoning_effort=kwargs.get("reasoning_effort"),
    )
    if not add_generation_prompt and messages[-1]["role"] == "user":
        for header in (thinking_start_token, thinking_end_token):
            out = out.removesuffix(ASSISTANT_SP_TOKEN + header)
    return out
