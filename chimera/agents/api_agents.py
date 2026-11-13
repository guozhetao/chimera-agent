from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ..data.dataset import Case
from ..features.neural import summarize_neural
from ..utils.jsonio import read_json, write_json


@dataclass
class AgentPrediction:
    case_id: str
    decision: Any
    reasoning: dict[str, Any] | None
    raw: str
    provider: str
    model: str


def compact_case_payload(case: Case, max_text_chars: int = 9000) -> dict[str, Any]:
    """Build a compact API payload. Full 960/1024-d vectors are summarized, not sent raw."""
    prompt = dict(case.prompt)
    prompt.pop("note_sections", None)
    clinical = dict(case.clinical)
    # The report text is usually small and highly informative; keep it but bound the size.
    for key in list(clinical):
        value = clinical[key]
        if isinstance(value, str) and len(value) > max_text_chars:
            clinical[key] = value[:max_text_chars] + "\n[TRUNCATED]"
    neural_summary = summarize_neural(case.neural).__dict__
    return {
        "case_id": case.case_id,
        "task": case.task,
        "structured_prompt": prompt,
        "clinical_data": clinical,
        "neural_representation_summary": neural_summary,
        "neural_vector_policy": "Original MRI/WSI image files are unavailable; provided neural representations are summarized as modality presence, count, mean L2 norm, dispersion, and top activated dimensions.",
    }


def top_dimensions_summary(case: Case, top_k: int = 12) -> dict[str, list[int]]:
    from ..features.neural import modality_arrays
    arrays = modality_arrays(case.neural)
    out = {}
    for key, arr in arrays.items():
        if arr is None or len(arr) == 0:
            out[key] = []
        else:
            mean = arr.mean(axis=0)
            out[key] = [int(i) for i in mean.argsort()[-top_k:][::-1].tolist()]
    return out


def build_task_instruction(case: Case) -> str:
    if case.task == 1:
        return (
            "Task 1: decide whether prostate biopsy is recommended using MRI and pre-biopsy clinical data. "
            'Return only JSON: {"decision":"yes"|"no","reasoning":{"confidence":"clear"|"borderline"|"uncertain",'
            '"variable_weights":{...},"reveal_sequence":["radiology_report"],"free_text":"..."}}.'
        )
    if case.task == 2:
        return (
            "Task 2: recommend EAU/NCCN-aligned management after MRI and biopsy. "
            'Return only JSON: {"decision":"active_surveillance"|"active_treatment"|"continued_surveillance"|"watchful_waiting",'
            '"reasoning":{"confidence":"clear"|"borderline","variable_weights":{...},"reveal_sequence":[],"free_text":"..."}}.'
        )
    return (
        "Task 3: estimate time-dependent biochemical-recurrence risk after radical prostatectomy. "
        'Return only JSON: {"months_to_recurrence":number,"event":0|1,"risk":number,'
        '"reasoning":{"confidence":"clear"|"borderline","free_text":"..."}}.'
    )


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = re.findall(r"\{.*\}", text, flags=re.S)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model did not return valid JSON: {text[:300]}")


class BaseApiAgent:
    provider = "base"

    def __init__(self, model: str, cache_dir: str | Path = "work/api_cache", temperature: float = 0.1):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.temperature = temperature

    def predict(self, case: Case) -> AgentPrediction:
        payload = compact_case_payload(case)
        payload["neural_top_dimensions"] = top_dimensions_summary(case)
        prompt = json.dumps(payload, ensure_ascii=False)
        raw = self._complete(prompt, build_task_instruction(case))
        parsed = parse_model_json(raw)
        if case.task == 3:
            decision = {
                "months_to_recurrence": float(parsed["months_to_recurrence"]),
                "event": int(parsed["event"]),
            }
        else:
            decision = parsed["decision"]
        return AgentPrediction(case.case_id, decision, parsed.get("reasoning"), raw, self.provider, self.model)

    def _cache_path(self, prompt: str, instruction: str) -> Path:
        key = hashlib.sha256(f"{self.provider}|{self.model}|{instruction}|{prompt}".encode("utf-8")).hexdigest()
        return self.cache_dir / self.provider / self.model / f"{key}.json"

    def _cached(self, path: Path) -> str | None:
        obj = read_json(path)
        return obj.get("raw") if isinstance(obj, dict) else None


class DeepSeekAgent(BaseApiAgent):
    provider = "deepseek"

    def __init__(self, model: str = "deepseek-flash", api_key: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Set DEEPSEEK_API_KEY; do not hard-code API keys in source code.")

    def _complete(self, prompt: str, instruction: str) -> str:
        path = self._cache_path(prompt, instruction)
        cached = self._cached(path)
        if cached is not None:
            return cached
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a rigorous clinical-decision AI. Return only valid JSON and do not invent unavailable data."},
                {"role": "user", "content": instruction + "\n\nCASE JSON:\n" + prompt},
            ],
            temperature=self.temperature,
            stream=False,
        )
        raw = response.choices[0].message.content
        write_json(path, {"raw": raw})
        return raw


class ZhipuAgent(BaseApiAgent):
    provider = "zhipu"

    def __init__(self, model: str = "glm-5.3", api_key: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("Set ZHIPU_API_KEY; do not hard-code API keys in source code.")

    def _complete(self, prompt: str, instruction: str) -> str:
        path = self._cache_path(prompt, instruction)
        cached = self._cached(path)
        if cached is not None:
            return cached
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise ImportError("Install the zai SDK or configure an OpenAI-compatible GLM endpoint.") from exc
        client = ZhipuAiClient(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a rigorous clinical-decision AI. Return only valid JSON and do not invent unavailable data."},
                {"role": "user", "content": instruction + "\n\nCASE JSON:\n" + prompt},
            ],
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content
        write_json(path, {"raw": raw})
        return raw


AGENT_REGISTRY = {
    "deepseek-flash": lambda **kw: DeepSeekAgent("deepseek-flash", **kw),
    "deepseek-v4-pro": lambda **kw: DeepSeekAgent("deepseek-v4-pro", **kw),
    "glm-flash": lambda **kw: ZhipuAgent("glm-flash", **kw),
    "glm-pro": lambda **kw: ZhipuAgent("glm-pro", **kw),
    "glm-image": lambda **kw: ZhipuAgent("glm-image", **kw),
}
