"""FastAPI adapter for RailPulse.

Run locally after installing project dependencies:
  uvicorn app.api:app --reload

The application is built by a factory so tests can construct an isolated
instance with their own store, gateway and settings instead of inheriting
whatever the import-time module globals happened to be.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai_copilot import (
    NonCompliantDraftProvider,
    OutreachCopilot,
    build_draft_provider,
)
from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway, PaymentGateway, RazorpayPaymentLinkGateway
from app.models import (
    EventType,
    FailureClass,
    InvalidStateTransition,
    PaymentEvent,
    PaymentRail,
    RecoveryCase,
    utc_now,
)
from app.service import RecoveryService
from app.store import ConcurrentCaseUpdate, RecoveryStore

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

#: Rs 100 crore, well above any real subscription renewal. Exists so a hostile
#: or malformed amount cannot reach the SQLite integer bind and 500 the route.
MAX_AMOUNT_PAISE = 100_00_00_000_00

METHOD_TO_RAIL = {
    "card": PaymentRail.CARD,
    "upi": PaymentRail.UPI_AUTOPAY,
    "emandate": PaymentRail.EMANDATE,
    "nach": PaymentRail.EMANDATE,
}


# --------------------------------------------------------------------- schema


class OutreachPreviewRequest(BaseModel):
    language: Literal["en", "hi", "hinglish"] = "hinglish"


class ReopenRequest(BaseModel):
    """A human taking a case back off the manual-review pile.

    ``note`` is required and has no default. A reopen is a person overriding
    the engine's decision to stop deciding, and an audit trail that records
    that it happened but not why is not much of an audit trail.
    """

    note: str = Field(min_length=3, max_length=500)
    #: Optionally supply what the operator worked out that the classifier
    #: could not. Without it the case is re-decided on what it already knows,
    #: which for an unclassified failure means it goes straight back.
    failure_class: Literal["transient", "customer_action", "risk", "unknown"] | None = None


# ---------------------------------------------------------------- normalising


def _first_entity(payload: dict[str, Any], entity_name: str) -> dict[str, Any]:
    entity = payload.get("payload", {})
    if not isinstance(entity, dict):
        return {}
    wrapper = entity.get(entity_name, {})
    if not isinstance(wrapper, dict):
        return {}
    inner = wrapper.get("entity", {})
    return inner if isinstance(inner, dict) else {}


def _issuer_of(payment: dict[str, Any]) -> str:
    """Which bank declined this.

    ``bank`` is populated for netbanking and eMandate. For **card** payments it
    is null and the issuer lives under ``card.issuer`` -- so reading only
    ``bank`` sent every card decline, on every issuer, into a single "unknown"
    bucket. BankHealthMonitor keys on (issuer, rail), so one bad issuer marked
    the entire card rail degraded and parked healthy traffic in cooldown; the
    release quota keys on the same tuple, so all of it shared one 25/tick
    budget. Per-issuer outage detection -- the mechanism the whole cooldown
    design exists for -- was inert on the dominant rail.
    """
    card = payment.get("card")
    card_issuer = card.get("issuer") if isinstance(card, dict) else None
    return str(
        payment.get("bank") or card_issuer or payment.get("wallet") or "unknown"
    )


def _occurred_at(payload: dict[str, Any]) -> datetime:
    """Webhook timestamp, falling back to now rather than to the epoch.

    A missing ``created_at`` previously produced 1970-01-01, which placed the
    event outside every health window and scheduled cooldowns in the past.
    """
    created_at = payload.get("created_at")
    if isinstance(created_at, (int, float)) and created_at > 0:
        try:
            return datetime.fromtimestamp(created_at, tz=UTC)
        except (OverflowError, OSError, ValueError):
            logger.warning("webhook created_at %r is out of range; using now", created_at)
    return utc_now()


def normalize_webhook(payload: dict[str, Any], event_id: str) -> PaymentEvent:
    try:
        event_type = EventType(payload["event"])
    except KeyError as exc:
        raise ValueError("webhook is missing the 'event' field") from exc
    except ValueError as exc:
        raise ValueError(f"unsupported webhook event: {payload.get('event')!r}") from exc

    payment = _first_entity(payload, "payment")
    subscription = _first_entity(payload, "subscription")
    payment_link = _first_entity(payload, "payment_link")
    order = _first_entity(payload, "order")
    invoice = _first_entity(payload, "invoice")
    # Disputes and refunds reference the payment they reverse, and their
    # payloads carry a different entity shape from the events above.
    dispute = _first_entity(payload, "dispute")
    refund = _first_entity(payload, "refund")
    logical_key = (
        invoice.get("id")
        or subscription.get("id")
        or order.get("id")
        or payment.get("order_id")
        or payment_link.get("reference_id")
        or payment_link.get("id")
        or dispute.get("order_id")
        or refund.get("order_id")
    )
    if not logical_key:
        raise ValueError("webhook does not contain a recovery correlation key")

    method = str(payment.get("method") or "unknown").lower()
    amount = payment.get("amount")
    if amount is None:
        amount = payment_link.get("amount")
    if amount is None:
        amount = dispute.get("amount") or refund.get("amount")
    try:
        # OverflowError matters here: json.loads("1e400") yields inf, and
        # int(inf) raises it. A 400-digit integer parses fine and then dies in
        # the SQLite bind instead. Either way an attacker-chosen amount turned
        # a webhook into a 500, which Razorpay reads as a server fault and
        # retries on its full backoff schedule forever.
        amount_paise = max(0, int(amount or 0))
    except (TypeError, ValueError, OverflowError):
        logger.warning("webhook carried an unusable amount %r; treating as 0", amount)
        amount_paise = 0
    if amount_paise > MAX_AMOUNT_PAISE:
        logger.warning("webhook amount %s exceeds the sane ceiling; clamping", amount_paise)
        amount_paise = MAX_AMOUNT_PAISE

    return PaymentEvent(
        event_id=event_id,
        event_type=event_type,
        logical_key=str(logical_key),
        occurred_at=_occurred_at(payload),
        amount_paise=amount_paise,
        payment_id=payment.get("id"),
        order_id=order.get("id") or payment.get("order_id"),
        subscription_id=subscription.get("id"),
        invoice_id=invoice.get("id"),
        payment_link_id=payment_link.get("id"),
        customer_id=(
            payment.get("customer_id")
            or subscription.get("customer_id")
            or invoice.get("customer_id")
            or payment_link.get("customer_id")
        ),
        reversal_id=dispute.get("id") or refund.get("id"),
        issuer=_issuer_of(payment),
        rail=METHOD_TO_RAIL.get(method, PaymentRail.UNKNOWN),
        failure_code=payment.get("error_reason") or payment.get("error_code"),
        captured=bool(payment.get("captured", False)),
        raw=payload,
    )


def _serialize_case(case: RecoveryCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "logical_key": case.logical_key,
        "state": case.state,
        "failure_class": case.failure_class,
        "failure_code": case.failure_code,
        "issuer": case.issuer,
        "rail": case.rail,
        "amount_paise": case.amount_paise,
        "next_action_at": case.next_action_at,
        "contacts_used": case.attention.contact_count_7d,
        "max_contacts": case.attention.max_contacts_7d,
        "attempt_count": case.attempt_count,
        "payment_link_id": case.payment_link_id,
        "payment_link_url": case.payment_link_url,
        "payment_link_status": case.payment_link_status,
        "outreach_preview": case.outreach_preview,
        "stop_reason": case.stop_reason,
        "requires_manual_reconciliation": case.requires_manual_reconciliation,
        "reversal_reason": case.reversal_reason,
        "reversed_at": case.reversed_at,
        "updated_at": case.updated_at,
    }


# ------------------------------------------------------------------- factory


def build_gateway(settings: Settings) -> PaymentGateway:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if key_id and key_secret:
        logger.info("using the live Razorpay payment-link gateway")
        return RazorpayPaymentLinkGateway(
            key_id,
            key_secret,
            timeout_seconds=settings.gateway_timeout_seconds,
            max_attempts=settings.gateway_max_attempts,
            backoff_seconds=settings.gateway_backoff_seconds,
        )
    logger.info("using the in-memory fake gateway; set RAZORPAY_KEY_ID/SECRET for live mode")
    return FakeRazorpayGateway()


def create_app(
    *,
    settings: Settings | None = None,
    store: RecoveryStore | None = None,
    gateway: PaymentGateway | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    owns_store = store is None
    resolved_store = store or RecoveryStore(resolved_settings.database_path)
    resolved_gateway = gateway or build_gateway(resolved_settings)
    health = BankHealthMonitor(
        window=resolved_settings.health_window,
        min_samples=resolved_settings.health_min_samples,
        degraded_success_rate=resolved_settings.health_degraded_success_rate,
        max_tracked_keys=resolved_settings.health_max_tracked_keys,
        sink=resolved_store,
    )
    # Same keys that drive the classifier drive the drafting seam. Without
    # one, OutreachCopilot falls back to its three templates and says so in
    # the preview's provider field, so a demo can never pass a lookup off as
    # a model here either.
    copilot = OutreachCopilot(
        build_draft_provider(
            os.getenv("ANTHROPIC_API_KEY"), os.getenv("OPENAI_API_KEY")
        )
    )
    service = RecoveryService(
        resolved_store,
        health,
        resolved_gateway,
        copilot=copilot,
        settings=resolved_settings,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # This warning used to say signatures were "NOT verified", which was
        # written before the endpoint started failing closed and never updated.
        # It told an operator the opposite of what would happen: the next
        # webhook was refused with a 503 they had no reason to expect. Each
        # branch now names the status the endpoint will actually return.
        if not os.getenv("RAZORPAY_WEBHOOK_SECRET"):
            if _unsigned_webhooks_allowed():
                logger.warning(
                    "RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS=1 and RAZORPAY_WEBHOOK_SECRET is not "
                    "set: webhook signatures are NOT verified and any caller can post events. "
                    "This is acceptable for local fixtures only."
                )
            else:
                logger.warning(
                    "RAZORPAY_WEBHOOK_SECRET is not set: POST /webhooks/razorpay will refuse "
                    "every request with 503. Set it to verify signatures, or set "
                    "RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS=1 to accept unsigned local fixtures."
                )
        now = utc_now()
        with resolved_store.transaction():
            pruned_health = resolved_store.prune_observations(
                now - resolved_settings.health_window
            )
            pruned_events = resolved_store.prune_processed_events(
                now - resolved_settings.processed_event_retention
            )
        logger.info(
            "pruned %s health observations and %s expired idempotency claims",
            pruned_health,
            pruned_events,
        )
        restored = health.restore(now)
        logger.info("restored %s issuer-health observations from storage", restored)
        yield
        if owns_store:
            resolved_store.close()

    application = FastAPI(title="RailPulse", version="0.2.0", lifespan=lifespan)
    application.add_middleware(GZipMiddleware, minimum_size=512)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    application.state.settings = resolved_settings
    application.state.store = resolved_store
    application.state.service = service
    application.state.health = health

    _register_routes(application)
    return application


def get_service(request: Request) -> RecoveryService:
    return request.app.state.service


def get_store(request: Request) -> RecoveryStore:
    return request.app.state.store


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _unsigned_webhooks_allowed() -> bool:
    """Only ever true when someone has explicitly asked for it."""
    return os.getenv("RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS") == "1"


def _verify_signature(raw: bytes, signature: str | None) -> None:
    """Reject anything unsigned unless a developer opted in by name.

    This used to return silently when RAZORPAY_WEBHOOK_SECRET was unset, with
    a startup log line as the only warning. That fails *open*: a deploy that
    misspells the variable, or loses it from a secret manager, accepts any POST
    from anyone with no signal at request time -- and a forged payment.failed
    plus a dispatch mints a real payment link against a merchant invoice the
    caller does not own. Absent configuration now means no traffic, and the
    dev escape hatch has to be spelled out.
    """
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")
    if not secret:
        if _unsigned_webhooks_allowed():
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "webhook signature verification is not configured; set "
                "RAZORPAY_WEBHOOK_SECRET, or RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS=1 "
                "for local fixtures"
            ),
        )
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    # Compare as bytes: Starlette decodes headers as latin-1, and
    # hmac.compare_digest raises TypeError on non-ASCII str, which would turn a
    # junk signature header into a 500 with a traceback instead of a clean 401.
    supplied = (signature or "").encode("latin-1", errors="ignore")
    if not hmac.compare_digest(expected.encode(), supplied):
        raise HTTPException(status_code=401, detail="invalid webhook signature")


def require_operator(
    authorization: str | None = Header(default=None),
) -> None:
    """Shared-secret guard for endpoints that mutate state or read case data.

    /actions/dispatch triggers outbound side effects and spends each customer's
    contact budget; /cases returns invoice identifiers, amounts and live
    payment-link URLs, which are bearer credentials. Both were open. When
    RAILPULSE_OPERATOR_TOKEN is unset the guard stays off so local demos and
    the test suite are unaffected -- but then it is a localhost tool, and the
    README says so rather than leaving it implied.
    """
    token = os.getenv("RAILPULSE_OPERATOR_TOKEN")
    if not token:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, supplied):
        raise HTTPException(status_code=401, detail="operator token required")


# -------------------------------------------------------------------- routes


def _register_routes(application: FastAPI) -> None:
    @application.get("/health")
    def health_check() -> dict[str, str]:
        """Liveness, plus whether the operator guard is on.

        The dashboard asks this before its first data call. Without it, turning
        RAILPULSE_OPERATOR_TOKEN on made the UI answer every panel with the
        same unexplained failure and no way to supply a token -- the guard was
        added and the only client of the endpoints it guards was never taught
        about it. Reporting that auth is required leaks nothing a 401 does not
        already say, and it is the difference between a prompt and a dead page.
        """
        return {
            "status": "ok",
            "operator_auth": "required" if os.getenv("RAILPULSE_OPERATOR_TOKEN") else "disabled",
        }

    @application.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @application.get("/cases", dependencies=[Depends(require_operator)])
    def cases(
        store: RecoveryStore = Depends(get_store),
        settings: Settings = Depends(get_settings),
        limit: int | None = Query(default=None, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        page_size = min(limit or settings.default_case_page_size, settings.max_case_page_size)
        return [_serialize_case(case) for case in store.list_cases(limit=page_size, offset=offset)]

    @application.get("/metrics")
    def metrics(store: RecoveryStore = Depends(get_store)) -> dict[str, int]:
        return store.metrics()

    @application.get("/overview", dependencies=[Depends(require_operator)])
    def overview(
        request: Request,
        store: RecoveryStore = Depends(get_store),
        settings: Settings = Depends(get_settings),
        limit: int | None = Query(default=None, ge=1, le=500),
    ) -> dict[str, Any]:
        """Everything the dashboard needs in a single round trip."""
        page_size = min(limit or settings.default_case_page_size, settings.max_case_page_size)
        service: RecoveryService = request.app.state.service
        now = utc_now()
        return {
            "generated_at": now,
            "simulation_enabled": isinstance(service.gateway, FakeRazorpayGateway),
            "metrics": store.metrics(),
            "total_cases": store.count_cases(),
            "cases": [_serialize_case(case) for case in store.list_cases(limit=page_size)],
            "degraded_rails": [
                snapshot.as_dict() for snapshot in request.app.state.health.degraded_snapshots(now)
            ],
        }

    @application.post("/actions/dispatch", dependencies=[Depends(require_operator)])
    def dispatch_actions(service: RecoveryService = Depends(get_service)) -> dict[str, Any]:
        """Demo/scheduler hook that dispatches currently due safe actions."""
        updated = service.dispatch_due_actions()
        return {
            "dispatched_case_ids": [case.id for case in updated],
            "dispatched": len(updated),
        }

    @application.get("/cases/{case_id}", dependencies=[Depends(require_operator)])
    def case_detail(
        case_id: str, store: RecoveryStore = Depends(get_store)
    ) -> dict[str, Any]:
        case = store.get_case_by_id(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="recovery case not found")
        payload = _serialize_case(case)
        payload["actions"] = [
            {
                "id": action.id,
                "action_type": action.action_type,
                "action_key": action.action_key,
                "status": action.status,
                "metadata": action.metadata,
                "created_at": action.created_at,
            }
            for action in store.list_actions(case_id)
        ]
        return payload

    @application.get("/cases/{case_id}/actions", dependencies=[Depends(require_operator)])
    def case_actions(
        case_id: str, store: RecoveryStore = Depends(get_store)
    ) -> list[dict[str, Any]]:
        if store.get_case_by_id(case_id) is None:
            raise HTTPException(status_code=404, detail="recovery case not found")
        return [
            {
                "id": action.id,
                "action_type": action.action_type,
                "action_key": action.action_key,
                "status": action.status,
                "metadata": action.metadata,
                "created_at": action.created_at,
            }
            for action in store.list_actions(case_id)
        ]

    @application.post("/cases/{case_id}/outreach-preview", dependencies=[Depends(require_operator)])
    def create_outreach_preview(
        case_id: str,
        payload: OutreachPreviewRequest,
        service: RecoveryService = Depends(get_service),
    ) -> dict[str, Any]:
        try:
            return service.create_outreach_preview(case_id, language=payload.language).model_dump()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConcurrentCaseUpdate as exc:
            raise HTTPException(status_code=409, detail="recovery case was modified concurrently") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post(
        "/cases/{case_id}/reopen", dependencies=[Depends(require_operator)]
    )
    def reopen_case(
        case_id: str,
        payload: ReopenRequest,
        service: RecoveryService = Depends(get_service),
    ) -> dict[str, Any]:
        """Return a manual-review case to the engine. Never un-stops a case.

        409 rather than 403 on a STOPPED case: the request is well-formed and
        the caller is authorised, it is the case's state that forbids it.
        """
        try:
            case = service.reopen(
                case_id,
                utc_now(),
                note=payload.note,
                failure_class=FailureClass(payload.failure_class)
                if payload.failure_class
                else None,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConcurrentCaseUpdate as exc:
            raise HTTPException(
                status_code=409, detail="recovery case was modified concurrently"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "case_id": case.id,
            "state": case.state.value,
            "stop_reason": case.stop_reason,
            "attempt_count": case.attempt_count,
            "next_action_at": case.next_action_at.isoformat() if case.next_action_at else None,
        }

    @application.get("/demo/config")
    def demo_config(service: RecoveryService = Depends(get_service)) -> dict[str, bool]:
        return {"simulation_enabled": isinstance(service.gateway, FakeRazorpayGateway)}

    def _require_simulation(service: RecoveryService) -> None:
        if not isinstance(service.gateway, FakeRazorpayGateway):
            raise HTTPException(
                status_code=403, detail="simulation is disabled with real Razorpay credentials"
            )

    @application.post("/demo/race-condition")
    def demo_race_condition(service: RecoveryService = Depends(get_service)) -> dict[str, Any]:
        """Run a fake-gateway-only late-authorisation demo for the dashboard."""
        _require_simulation(service)
        now = utc_now()
        logical_key = f"demo_invoice_{uuid4().hex[:8]}"
        payment_id = f"pay_{uuid4().hex[:12]}"

        def demo_event(event_type: EventType, **overrides: Any) -> PaymentEvent:
            base: dict[str, Any] = {
                "event_id": f"evt_{uuid4().hex}",
                "event_type": event_type,
                "logical_key": logical_key,
                "occurred_at": now,
                "amount_paise": 49900,
                "payment_id": payment_id,
                "invoice_id": logical_key,
                "issuer": "hdfc",
                "rail": PaymentRail.CARD,
            }
            base.update(overrides)
            return PaymentEvent(**base)

        case, _ = service.ingest(demo_event(EventType.PAYMENT_FAILED, failure_code="CARD_EXPIRED"))
        service.dispatch_due_actions(now)
        pending, _ = service.ingest(demo_event(EventType.PAYMENT_AUTHORIZED, captured=False))
        recovered, _ = service.ingest(demo_event(EventType.PAYMENT_CAPTURED, captured=True))
        return {
            "case_id": case.id if case else None,
            "timeline": [
                "payment.failed → consent_required",
                "payment_link.created → link_sent",
                f"payment.authorized → {pending.state if pending else 'unmatched'} (eligible link cancelled)",
                f"payment.captured → {recovered.state if recovered else 'unmatched'}",
            ],
        }

    @application.post("/demo/outreach-preview")
    def demo_outreach_preview(
        compliant: bool = Query(default=True),
        service: RecoveryService = Depends(get_service),
    ) -> dict[str, Any]:
        """Create a fake recovery link and a *preview only* message.

        ``compliant=false`` swaps in a drafting provider that returns coercive
        copy, an invented discount and its own URL. The validator is unchanged
        -- only the proposal differs -- so the rejection path can be watched
        rather than taken on trust. Without this the guardrail was unreachable:
        the template provider emits three hardcoded compliant strings, so the
        validator had nothing to refuse and the dashboard's blocked branch was
        dead code.
        """
        _require_simulation(service)
        now = utc_now()
        logical_key = f"demo_outreach_{uuid4().hex[:8]}"
        case, _ = service.ingest(
            PaymentEvent(
                event_id=f"evt_{uuid4().hex}",
                event_type=EventType.PAYMENT_FAILED,
                logical_key=logical_key,
                occurred_at=now,
                amount_paise=49900,
                payment_id=f"pay_{uuid4().hex[:12]}",
                invoice_id=logical_key,
                issuer="axis",
                rail=PaymentRail.UPI_AUTOPAY,
                failure_code="MANDATE_CANCELLED",
            )
        )
        service.dispatch_due_actions(now)
        if case is None:
            raise HTTPException(status_code=500, detail="demo case was not created")
        drafting = None if compliant else OutreachCopilot(NonCompliantDraftProvider())
        preview = service.create_outreach_preview(
            case.id, language="hinglish", now=now, copilot=drafting
        )
        return {
            "case_id": case.id,
            "compliant_draft_requested": compliant,
            "preview": preview.model_dump(),
        }

    @application.post("/webhooks/razorpay")
    async def razorpay_webhook(
        request: Request,
        x_razorpay_event_id: str | None = Header(default=None),
        x_razorpay_signature: str | None = Header(default=None),
        service: RecoveryService = Depends(get_service),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        raw = await request.body()
        if len(raw) > settings.max_webhook_bytes:
            raise HTTPException(status_code=413, detail="webhook payload too large")
        _verify_signature(raw, x_razorpay_signature)
        if not x_razorpay_event_id:
            raise HTTPException(status_code=400, detail="missing X-Razorpay-Event-Id")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="webhook body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="webhook body must be a JSON object")
        try:
            event = normalize_webhook(payload, x_razorpay_event_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            # ingest() is synchronous and, on a success event, performs a live
            # gateway call outside the transaction. Calling it directly from an
            # async handler parks the event loop for the whole of that call --
            # up to gateway_timeout x gateway_max_attempts, roughly 30s by
            # default -- during which every other webhook, health probe and
            # dashboard request behind it stalls and Razorpay starts timing out
            # and redelivering. run_in_threadpool puts it on a worker thread,
            # which is where FastAPI would have run it had this route been a
            # plain def like its siblings.
            case, processed = await run_in_threadpool(service.ingest, event)
        except ConcurrentCaseUpdate as exc:
            # Worth redelivering: another writer won the race, and the same
            # body will apply cleanly on a second attempt.
            logger.warning("webhook %s lost a write race: %s", x_razorpay_event_id, exc)
            raise HTTPException(
                status_code=409, detail="recovery case was modified concurrently"
            ) from exc
        except InvalidStateTransition as exc:
            # Deterministic for a given body and case state, so redelivery
            # reproduces it forever and the event is never applied -- the
            # idempotency claim was rolled back with the transaction, so it
            # cannot even converge. Acknowledge it, park the case for a human,
            # and stop the provider retrying into a wall.
            logger.error(
                "webhook %s describes a transition the state machine forbids: %s",
                x_razorpay_event_id,
                exc,
            )
            return {
                "processed": False,
                "case_id": None,
                "state": None,
                "detail": "event acknowledged but not applied; flagged for manual review",
            }
        return {
            "processed": processed,
            "case_id": case.id if case else None,
            "state": case.state if case else None,
        }


app = create_app()
