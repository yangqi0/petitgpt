from __future__ import annotations

import argparse
import re
from pathlib import Path

NEW_FUNCTION = '''
def openai_compatible_chat(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 120,
    api_key: Optional[str] = None,
    timeout: int = 120,
    disable_thinking: bool = True,
) -> str:
    import urllib.request
    import urllib.error

    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
    url = api_base.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Qwen3 / Qwen3.5 thinking models may emit visible reasoning unless thinking is disabled.
    # vLLM accepts this non-OpenAI-standard field in the OpenAI-compatible endpoint.
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))

    msg = out["choices"][0]["message"]
    content = msg.get("content") or ""

    # Safety cleanup in case the server still returns visible <think> blocks.
    content = re.sub(r"(?is)<think>.*?</think>\\s*", "", content).strip()

    if content.lower().startswith("thinking process:"):
        raise RuntimeError(
            "Teacher returned visible thinking content. "
            "Check that chat_template_kwargs={'enable_thinking': False} is supported by your vLLM server, "
            "or restart vLLM with --default-chat-template-kwargs '{\\"enable_thinking\\": false}'."
        )

    return content
'''

def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "disable_thinking: bool = True" in text and '"chat_template_kwargs"' in text:
        print(f"[skip] already patched: {path}")
        return

    if "import re" not in text.splitlines()[:40]:
        # Insert near other imports.
        if "import random\n" in text:
            text = text.replace("import random\n", "import random\nimport re\n")
        elif "import os\n" in text:
            text = text.replace("import os\n", "import os\nimport re\n")
        else:
            text = "import re\n" + text

    # Replace the whole openai_compatible_chat function by locating start and next top-level def.
    start = text.find("def openai_compatible_chat(")
    if start < 0:
        raise RuntimeError("Could not find openai_compatible_chat in the target file.")

    next_def = text.find("\\ndef ", start + 1)
    if next_def < 0:
        end = len(text)
    else:
        end = next_def + 1

    text = text[:start] + NEW_FUNCTION.strip() + "\\n\\n" + text[end:]
    path.write_text(text, encoding="utf-8")
    print(f"[patched] {path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general_utils", default="distill/general_utils.py")
    args = ap.parse_args()
    patch_file(Path(args.general_utils))

if __name__ == "__main__":
    main()
