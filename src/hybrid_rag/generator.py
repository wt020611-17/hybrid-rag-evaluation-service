import json
import os
import urllib.error
import urllib.request
from typing import Iterable, Optional

from .models import SearchHit


class OpenAICompatibleGenerator:
    """Minimal optional OpenAI-compatible chat client using the standard library."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout_seconds = timeout_seconds or float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def generate(self, query: str, hits: Iterable[SearchHit]) -> str:
        if not self.configured:
            raise RuntimeError("LLM is not configured")
        context = "\n\n".join(
            f"[{index}] source={hit.source}\n{hit.text}"
            for index, hit in enumerate(hits, start=1)
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "仅依据提供的上下文回答；证据不足时明确说不知道，并使用 [n] 标注引用。",
                },
                {"role": "user", "content": f"问题：{query}\n\n上下文：\n{context}"},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return body["choices"][0]["message"]["content"].strip()

