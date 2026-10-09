"""Local-first request orchestration shared by the API routes.

Every request runs the local diagnostic engine first; its result is complete on
its own. The optional AI layer is consulted afterwards, only when the policy
allows it, and any AI failure is reported in ``report.ai`` instead of failing
the request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import Settings
from app.errors import AppError
from app.runtime_config import RuntimeConfig
from app.schemas.common import AnalysisMode, FindingStatus, Personality, ScanTrigger
from app.schemas.diagnostics import AIRun, DiagnosticReport, ScanAnalysisResponse
from app.schemas.scene import SceneModel, SceneSignals, SceneStats
from app.services.ai_policy import USER_TRIGGERS, AIBudget, AIPolicy
from app.services.ai_providers import AIError
from app.services.ai_service import AIService, DiagnoseRequest, DiagnoseResult, error_info
from app.services.diagnostic_service import DiagnosticService
from app.services.metrics import Metrics
from app.services.scan_store import ScanSession, canonical_bug_id, utcnow
from app.services.vision_service import PreparedImage, frame_signals


@dataclass(slots=True)
class Reasoning:
    result: DiagnoseResult | None
    run: AIRun


class Pipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        config: RuntimeConfig,
        diagnostics: DiagnosticService,
        ai: AIService,
        policy: AIPolicy,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.config = config
        self.diagnostics = diagnostics
        self.ai = ai
        self.policy = policy
        self.metrics = metrics

    # -- the optional AI layer -------------------------------------------------------------

    def _run(self, status: str, *, trigger: ScanTrigger | None, reason: str | None = None, **extra: object) -> AIRun:
        ai = self.ai.status
        return AIRun(
            status=status,  # type: ignore[arg-type]
            provider=ai.provider if ai.configured else None,
            model=ai.model,
            trigger=trigger.value if trigger else None,
            reason=reason,
            **extra,  # type: ignore[arg-type]
        )

    def off_run(self, trigger: ScanTrigger | None = None) -> AIRun:
        return self._run("off", trigger=trigger, reason=self.ai.status.detail)

    async def reason(self, budget: AIBudget, request: DiagnoseRequest, *, ahash: int | None) -> Reasoning:
        trigger = request.trigger or ScanTrigger.MANUAL
        if not self.ai.enabled:
            return Reasoning(None, self.off_run(trigger))
        decision = self.policy.decide(budget, trigger=trigger, mode=request.mode, ahash=ahash, focus=request.focus)
        if decision.cached is not None:
            self.metrics.ai_cached += 1
            cached: DiagnoseResult = decision.cached
            return Reasoning(
                cached,
                AIRun(
                    status="cached",
                    provider=cached.provider,
                    model=cached.model,
                    latency_ms=0,
                    trigger=trigger.value,
                    reason=decision.reason,
                ),
            )
        if not decision.call:
            self.metrics.record_ai_skip(decision.code or "skipped")
            return Reasoning(None, self._run("skipped", trigger=trigger, reason=decision.reason))
        self.policy.record_call(budget)
        try:
            result = await self.ai.diagnose(request)
        except AIError as exc:
            self.policy.on_failure(exc.code)
            return Reasoning(
                None,
                self._run(
                    "unavailable",
                    trigger=trigger,
                    reason="AI reasoning failed; the local results are complete on their own.",
                    error=error_info(exc),
                ),
            )
        self.policy.on_success()
        self.policy.store(budget, mode=request.mode, focus=request.focus, ahash=ahash, result=result)
        return Reasoning(
            result,
            AIRun(
                status="ok",
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                trigger=trigger.value,
                reason="served by the fallback provider" if result.fallback_used else None,
            ),
        )

    # -- live scans -----------------------------------------------------------------------------

    async def observe(self, session: ScanSession, scene: SceneModel) -> ScanAnalysisResponse:
        """A local observation (scene model only, no image, never an AI call)."""
        started = time.perf_counter()
        async with session.lock:
            obs = self.diagnostics.observe(session, scene, mode=AnalysisMode.LIVE)
            suggestion = None
            if self.ai.enabled:
                pending = [
                    (tf.id, tf.severity, len(tf.object_ids) >= 2)
                    for tf in session.findings.values()
                    if tf.source == "local"
                    and tf.status in (FindingStatus.CONFIRMED, FindingStatus.TRACKING)
                    and not tf.out_of_view
                ]
                suggestion = self.policy.suggest(session.ai, scene=scene, pending=pending)
            run = session.last_ai_run or (
                self.off_run(ScanTrigger.OBSERVE)
                if not self.ai.enabled
                else self._run("skipped", trigger=ScanTrigger.OBSERVE, reason="No AI call needed for this observation.")
            )
            tracked = list(session.findings.values())
            report = self.diagnostics.report(
                mode=AnalysisMode.LIVE,
                personality=session.personality,
                scene=scene,
                findings=obs.findings,
                ai_run=run,
                latency_ms=int((time.perf_counter() - started) * 1000),
                local=obs.local,
                score_findings=tracked,
                scene_info=session.scene,
                final=session.final_diagnosis,
                trigger=ScanTrigger.OBSERVE,
            )
            return ScanAnalysisResponse(
                scan=self.diagnostics.scan_state(session), report=report, events=obs.events, ai_suggestion=suggestion
            )

    async def live_frame(
        self,
        session: ScanSession,
        *,
        scene: SceneModel | None,
        image: PreparedImage | None,
        trigger: ScanTrigger,
        mode: AnalysisMode,
        focus: str | None = None,
    ) -> ScanAnalysisResponse:
        """A frame (and its scene model) sent for AI reasoning during a live scan."""
        started = time.perf_counter()
        focus_id = canonical_bug_id(focus) if focus else None
        if focus and focus_id is None:
            raise AppError("Unknown finding id.", code="INVALID_FOCUS", status_code=422, hint="Use an id like BUG-003.")
        async with session.lock:
            if scene is not None:
                obs = self.diagnostics.observe(session, scene, mode=mode)
                current_ids = [f.id for f in obs.findings]
                events = list(obs.events)
                local = obs.local
            else:
                current_ids, events, local = [], [], []
            local_briefs, ai_briefs = self.diagnostics.briefs(session)
            request = DiagnoseRequest(
                mode=mode,
                personality=session.personality,
                images=[image] if image is not None else [],
                scene=scene,
                relations=self.diagnostics.relation_briefs(scene),
                trigger=trigger,
                local_findings=local_briefs,
                ai_findings=ai_briefs,
                focus=focus_id,
            )
            if image is None and trigger not in USER_TRIGGERS:
                reasoning = Reasoning(
                    None,
                    self.off_run(trigger)
                    if not self.ai.enabled
                    else self._run("skipped", trigger=trigger, reason="No frame attached."),
                )
            else:
                reasoning = await self.reason(session.ai, request, ahash=image.ahash if image is not None else None)
            ai_models = []
            if reasoning.run.status == "ok" and reasoning.result is not None:
                ai_models, ai_events, explained = self.diagnostics.apply_ai(session, reasoning.result, mode=mode, scene=scene)
                events.extend(ai_events)
                self.policy.after_call(session.ai, trigger=trigger, scene=scene, explained=explained)
            if reasoning.run.status != "off":
                session.last_ai_run = reasoning.run
            if focus_id and focus_id not in current_ids and focus_id in session.findings:
                current_ids.append(focus_id)
            local_models = [session.findings[i].to_model() for i in current_ids if i in session.findings]
            report = self.diagnostics.report(
                mode=mode,
                personality=session.personality,
                scene=scene,
                findings=local_models + ai_models,
                ai_run=reasoning.run,
                latency_ms=int((time.perf_counter() - started) * 1000),
                local=local,
                ai=reasoning.result,
                score_findings=list(session.findings.values()),
                scene_info=session.scene,
                final=session.final_diagnosis,
                trigger=trigger,
                image=image,
            )
            return ScanAnalysisResponse(scan=self.diagnostics.scan_state(session), report=report, events=events)

    # -- one-shot analyses -------------------------------------------------------------------------

    def signal_scene(self, image: PreparedImage) -> SceneModel:
        """A scene model with pixel signals only, for images sent without one."""
        brightness, sharpness = frame_signals(image, self.config)
        return SceneModel(
            width=image.width,
            height=image.height,
            signals=SceneSignals(brightness=brightness, sharpness=sharpness),
            stats=SceneStats(detectors=[]),
        )

    async def analyze_image(
        self,
        *,
        personality: Personality,
        scene: SceneModel | None,
        image: PreparedImage | None,
        reasoning: bool = True,
    ) -> DiagnosticReport:
        started = time.perf_counter()
        warnings: list[str] = []
        if scene is None:
            if image is None:
                raise AppError("Send a scene model or an image.", code="MISSING_INPUT", status_code=422)
            scene = self.signal_scene(image)
            warnings.append("No on-device detections were supplied; only image signals were measured locally.")
        now = utcnow()
        local, _ = self.diagnostics.evaluate(scene, mode=AnalysisMode.IMAGE, personality=personality)
        local_models = self.diagnostics.stateless_local(local, scene, now=now)
        object_ids = {m.id: list(lf.object_ids) for m, lf in zip(local_models, local, strict=True)}
        if reasoning and image is not None:
            request = DiagnoseRequest(
                mode=AnalysisMode.IMAGE,
                personality=personality,
                images=[image],
                scene=scene,
                relations=self.diagnostics.relation_briefs(scene),
                trigger=ScanTrigger.MANUAL,
                local_findings=self.diagnostics.stateless_briefs(local_models, object_ids),
            )
            result = await self.reason(self.policy.stateless, request, ahash=image.ahash)
        else:
            result = Reasoning(None, self.off_run(ScanTrigger.MANUAL) if not self.ai.enabled else self._run(
                "skipped", trigger=ScanTrigger.MANUAL, reason="No image attached; local analysis only."
            ))
        ai_models = []
        if result.result is not None:
            ai_models, _ = self.diagnostics.stateless_ai(result.result.diagnosis, local_models, now=now)
        return self.diagnostics.report(
            mode=AnalysisMode.IMAGE,
            personality=personality,
            scene=scene,
            findings=local_models + ai_models,
            ai_run=result.run,
            latency_ms=int((time.perf_counter() - started) * 1000),
            local=local,
            ai=result.result,
            image=image,
            warnings=warnings,
        )
