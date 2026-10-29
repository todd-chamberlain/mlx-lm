# Copyright © 2026 Apple Inc.

"""
Tool calls in the DSML format of DeepSeek-V4.1 (see chat_templates/deepseek_v41):

<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>

A string="true" value is the text as written; any other value is JSON.
"""

import json
from typing import Any

import regex as re

tool_call_start = "<｜DSML｜ calls>"
tool_call_end = "</｜DSML｜ calls>"

_invoke_regex = re.compile(
    r'<｜DSML｜ invoke name="([^"]*)">(.*?)</｜DSML｜ invoke>', re.DOTALL
)
_parameter_regex = re.compile(
    r'<｜DSML｜ parameter name="([^"]*)" string="(true|false)">(.*?)'
    r"</｜DSML｜ parameter>",
    re.DOTALL,
)


def _value(text: str, is_string: str) -> Any:
    if is_string == "true":
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_tool_call(text: str, tools: Any | None = None):
    calls = []
    for name, body in _invoke_regex.findall(text):
        arguments = {
            key: _value(value, is_string)
            for key, is_string, value in _parameter_regex.findall(body)
        }
        calls.append(dict(name=name, arguments=arguments))
    if not calls:
        raise ValueError("No DSML tool call found.")
    return calls
