from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request

from ..data.dataset import Case
from ..features.neural import summarize_neural
from ..utils.jsonio import read_json, write_json


@dataclass
class DirectApiConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int = 1600
    cache_dir: Path = Path("work/api_cache")


class DirectApiAgent:
    """Dependency-free OpenAI-compatible client used for DeepSeek and GLM."""

    def __init__(self, config: DirectApiConfig):
        self.config = config
        if config.provider == "zhipu":
            # GLM-5.3 spends part of the completion budget on hidden reasoning;
            # a 8000-token cap truncated the JSON for a few cases.
            self.config.max_tokens = max(config.max_tokens, 16000)

    def _endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key.strip()}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        system: str,
        user: str,
        cache_key: str,
        required_keys: set[str] | None = None,
    ) -> str:
        cache_key = "v2_" + cache_key
        cache_path = self.config.cache_dir / self.config.provider / self.config.model / f"{cache_key}.json"
        cached = read_json(cache_path)
        if isinstance(cached, dict) and cached.get("content"):
            content = str(cached["content"])
            # A truncated or malformed response must not be replayed forever:
            # only reuse the cache when the payload satisfies its schema.
            if _payload_is_valid(content, required_keys):
                return content
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.provider == "deepseek":
            # Flash can otherwise spend the entire completion budget on hidden reasoning.
            payload["reasoning_effort"] = "low"
            payload["thinking"] = {"type": "disabled"}
        elif self.config.provider == "zhipu":
            # GLM-5.3 rejects temperature=0.0 and does not need the DeepSeek
            # "thinking" field.  Keep a near-deterministic but valid temperature.
            payload["temperature"] = max(float(self.config.temperature), 0.01)
        req = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        body = self._post(req)
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        finish_reason = choice.get("finish_reason")
        # If the model ran out of completion budget (usually hidden reasoning
        # consumed it), retry with a larger budget before parsing.
        for _ in range(2):
            if finish_reason != "length" or self.config.max_tokens >= 32000:
                break
            previous_budget = self.config.max_tokens
            self.config.max_tokens = min(32000, self.config.max_tokens * 2)
            payload["max_tokens"] = self.config.max_tokens
            req = urllib.request.Request(
                self._endpoint(),
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            try:
                body = self._post(req)
            except RuntimeError as exc:
                # The provider may cap output tokens below the requested value;
                # keep the first response rather than failing the case.
                if "HTTP 400" not in str(exc):
                    raise
                self.config.max_tokens = previous_budget
                break
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
        write_json(cache_path, {"content": content})
        return content

    def _post(self, req: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=240) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail[:800]}")
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(f"network error: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise last_error or RuntimeError("request failed")


class TwoStageClinicalAgent:
    """Stage 1 extracts evidence; Stage 2 makes the decision using that evidence."""

    def __init__(self, agent: DirectApiAgent):
        self.agent = agent

    def _neural_summary(self, case: Case) -> dict[str, Any]:
        summary = summarize_neural(case.neural).__dict__
        # Keep the prompt small and interpretable for text-only API models.
        summary.pop("mri_mean_norm", None)
        summary.pop("biopsy_mean_norm", None)
        summary.pop("prostatectomy_mean_norm", None)
        return summary

    def extract_evidence(self, case: Case) -> dict[str, Any]:
        system = (
            "You are a clinical information extraction engine. Return only strict JSON. "
            "Extract the most decision-relevant variables, explicitly preserving negative and missing findings."
        )
        user = json.dumps({
            "task": case.task,
            "structured_prompt": case.prompt,
            "clinical_data": case.clinical,
            "neural_representation_summary": self._neural_summary(case),
            "required_evidence_schema": {
                "patient": "age, sex if available, comorbidities, medications",
                "psa": "latest PSA, PSA density, PSA velocity, prior PSA if available",
                "mri": "PI-RADS, lesion size, lesion location, DWI/ADC/DCE, extracapsular extension, seminal vesicle invasion, lymph nodes",
                "biopsy": "biopsy status, ISUP grade group, Gleason primary/secondary, cribriform/intraductal, perineural invasion, stability or upgrade",
                "surgical_pathology": "Gleason, ISUP, pT stage, margins, extraprostatic extension, seminal vesicle invasion, lymph nodes, lymphovascular invasion",
                "neural_representations": "modality presence and slide counts only",
                "missing_or_unknown": "list all important unavailable fields",
            },
        }, ensure_ascii=False)
        raw = self.agent.chat(
            system,
            user,
            f"extract_{case.task}_{case.case_id}",
            required_keys={"patient"},
        )
        return parse_json(raw, {"patient"})

    def predict(self, case: Case) -> dict[str, Any]:
        try:
            evidence = self.extract_evidence(case)
        except ValueError:
            # Stage 1 is an evidence aid, not the decision: if its JSON is
            # unusable, continue with the structured prompt alone.
            evidence = {"extraction_error": "stage-1 response was invalid; use structured_prompt directly"}
        system = (
            "You are a cautious clinical decision agent. Use only supplied evidence. "
            "Return strict JSON. Do not invent variables. If evidence is missing, mark it missing."
        )
        if case.task == 1:
            instruction = {
                "decision": "yes|no",
                "reasoning": {
                    "confidence": "clear|borderline|uncertain",
                    "variable_weights": "dict of variable -> not_used|noted|important|decisive",
                    "reveal_sequence": ["radiology_report"],
                    "free_text": "3-6 sentence evidence-grounded explanation",
                },
            }
        elif case.task == 2:
            instruction = {
                "decision": "active_surveillance|active_treatment|continued_surveillance|watchful_waiting",
                "reasoning": {
                    "confidence": "clear|borderline",
                    "variable_weights": "dict",
                    "reveal_sequence": [],
                    "free_text": "3-6 sentence EAU/NCCN-grounded explanation",
                },
            }
        else:
            instruction = {
                "months_to_recurrence": "number",
                "event": "0|1",
                "risk": "number from 0 to 1",
                "reasoning": {
                    "confidence": "clear|borderline",
                    "free_text": "3-6 sentence pathology-grounded explanation",
                },
            }
        user = json.dumps({
            "case_id": case.case_id,
            "task": case.task,
            "structured_prompt": case.prompt,
            "extracted_evidence": evidence,
            "required_output_schema": instruction,
        }, ensure_ascii=False)
        required = {"decision"} if case.task in (1, 2) else None
        raw = self.agent.chat(
            system,
            user,
            f"predict_{case.task}_{case.case_id}",
            required_keys=required,
        )
        try:
            result = parse_json(raw, required)
        except ValueError:
            result = {}
        decision = normalize_decision(case.task, result.get("decision")) if case.task in (1, 2) else None
        if case.task in (1, 2) and decision is None:
            result = self._repair_decision(case, evidence)
            decision = normalize_decision(case.task, result.get("decision"))
        if decision is not None:
            result["decision"] = decision
        result["extracted_evidence"] = evidence
        return result

    def _repair_decision(self, case: Case, evidence: dict[str, Any]) -> dict[str, Any]:
        """Ask once more for a minimal, schema-only decision.

        Used when the first response was truncated or omitted the label; it is
        still the model making the decision, just with a shorter output format.
        """
        allowed = (
            "yes|no"
            if case.task == 1
            else "active_surveillance|active_treatment|continued_surveillance|watchful_waiting"
        )
        system = "You are a clinical decision agent. Return only strict JSON with no prose."
        user = json.dumps({
            "case_id": case.case_id,
            "task": case.task,
            "structured_prompt": case.prompt,
            "extracted_evidence": evidence,
            "required_output_schema": {
                "decision": allowed,
                "reasoning": {
                    "confidence": "clear|borderline|uncertain",
                    "free_text": "2-4 sentence evidence-grounded explanation",
                },
            },
        }, ensure_ascii=False)
        raw = self.agent.chat(
            system, user, f"repair_{case.task}_{case.case_id}", required_keys={"decision"}
        )
        return parse_json(raw, {"decision"})


def parse_json(text: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    """Parse strict JSON, tolerating Markdown fences and trailing prose.

    When ``required_keys`` is given, a parsed object is only accepted if it
    contains every key.  This prevents a truncated response from being
    mistaken for one of its nested dictionaries.
    """

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text).strip()
    candidates: list[dict[str, Any]] = []
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            candidates.append(value)
    except json.JSONDecodeError:
        pass
    if not candidates:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
    for value in candidates:
        if not required_keys or required_keys <= set(value):
            return value
    if candidates and not required_keys:
        return candidates[0]
    raise ValueError(
        f"JSON response missing required keys {sorted(required_keys or [])}: {text[:300]}"
    )


def _payload_is_valid(text: str, required_keys: set[str] | None) -> bool:
    if not required_keys:
        return bool(text.strip())
    try:
        parse_json(text, required_keys)
    except ValueError:
        return False
    return True


DECISION_ALIASES = {
    "yes": "yes",
    "no": "no",
    "biopsy": "yes",
    "perform_biopsy": "yes",
    "repeat_biopsy": "yes",
    "no_biopsy": "no",
    "do_not_biopsy": "no",
    "negative": "no",
    "positive": "yes",
    "as": "active_surveillance",
    "active_surveillance": "active_surveillance",
    "at": "active_treatment",
    "active_treatment": "active_treatment",
    "cs": "continued_surveillance",
    "continued_surveillance": "continued_surveillance",
    "continue_surveillance": "continued_surveillance",
    "ww": "watchful_waiting",
    "watchful_waiting": "watchful_waiting",
    "watch_and_wait": "watchful_waiting",
    "watchful-waiting": "watchful_waiting",
}


def normalize_decision(task: int, value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    mapped = DECISION_ALIASES.get(key)
    if mapped is None:
        return None
    if task == 1 and mapped not in {"yes", "no"}:
        return None
    if task == 2 and mapped in {"yes", "no"}:
        return None
    return mapped


PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "models": {
            "deepseek-flash": "deepseek-flash",
            "deepseek-v4-pro": "deepseek-v4-pro",
        },
        "key_env": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": {
            # 2026-11 model IDs returned by the provider's /models endpoint.
            "glm-5.3": "glm-5.3",
            "glm-5.3-flash": "glm-5.3-flash",
            "glm-flash": "glm-5.3-flash",
            "glm-flashx": "glm-5.3-flashx",
            "glm-pro": "glm-5.3",
            "glm-legacy-air": "glm-4.5-air",
            "glm-legacy": "glm-4.5",
            "glm-legacy-flash": "glm-4-flash",
            "glm-image": "glm-4v-flash",
        },
        "key_env": "ZHIPU_API_KEY",
    },
}


def make_agent(provider: str, model_alias: str, api_key: str | None = None, temperature: float = 0.0) -> TwoStageClinicalAgent:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider}")
    spec = PROVIDERS[provider]
    if model_alias not in spec["models"]:
        raise ValueError(f"Unknown model alias {model_alias}")
    key = api_key or os.getenv(spec["key_env"])
    if not key:
        raise ValueError(f"Set {spec['key_env']} before calling {provider}")
    config = DirectApiConfig(
        provider=provider,
        model=spec["models"][model_alias],
        api_key=key,
        base_url=spec["base_url"],
        temperature=temperature,
    )
    return TwoStageClinicalAgent(DirectApiAgent(config))
