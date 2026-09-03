from __future__ import annotations

import hashlib
import hmac
import json
import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import MAX_AMOUNT_PAISE, create_app, normalize_webhook
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentRail
from app.store import RecoveryStore


def webhook_body(
    event: str = "payment.failed",
    *,
    created_at: int | None = 1787500000,
    order_id: str = "order_api_1",
    error_reason: str = "CARD_EXPIRED",
) -> dict:
    payload: dict = {
        "event": event,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_api_1",
                    "order_id": order_id,
                    "amount": 49900,
                    "method": "card",
                    "bank": "HDFC",
                    "error_reason": error_reason,
                    "captured": event == "payment.captured",
                }
            }
        },
    }
    if created_at is not None:
        payload["created_at"] = created_at
    return payload


class NormalizeWebhookTests(unittest.TestCase):
    def test_missing_created_at_falls_back_to_now_not_the_epoch(self) -> None:
        event = normalize_webhook(webhook_body(created_at=None), "evt_1")
        self.assertGreater(event.occurred_at.year, 2000, "epoch fallback regressed")

    def test_out_of_range_created_at_is_rejected_gracefully(self) -> None:
        event = normalize_webhook(webhook_body(created_at=10**18), "evt_2")
        self.assertGreater(event.occurred_at.year, 2000)

    def test_valid_timestamp_is_preserved(self) -> None:
        event = normalize_webhook(webhook_body(created_at=1787500000), "evt_3")
        self.assertEqual(event.occurred_at, datetime.fromtimestamp(1787500000, tz=UTC))

    def test_unsupported_event_is_a_value_error(self) -> None:
        # Deliberately not payment.dispute.created any more: that is now a
        # supported event, and using it here would have kept passing for the
        # wrong reason (no correlation key rather than an unknown type).
        with self.assertRaises(ValueError) as caught:
            normalize_webhook({"event": "payment.totally.made.up"}, "evt_4")
        self.assertIn("unsupported webhook event", str(caught.exception))

    def test_a_dispute_is_a_supported_event(self) -> None:
        payload = {
            "event": "payment.dispute.created",
            "created_at": 1787500000,
            "payload": {
                "dispute": {"entity": {"id": "disp_1", "order_id": "order_x", "amount": 49900}}
            },
        }
        event = normalize_webhook(payload, "evt_disp")
        self.assertEqual(event.event_type, EventType.PAYMENT_DISPUTE_CREATED)
        self.assertEqual(event.logical_key, "order_x")
        self.assertEqual(event.amount_paise, 49900)
        self.assertEqual(event.reversal_id, "disp_1")

    def test_missing_correlation_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_webhook({"event": "payment.failed", "payload": {}}, "evt_5")

    def test_malformed_nested_payload_does_not_crash(self) -> None:
        payload = {"event": "payment.failed", "payload": {"payment": "not-an-object"}}
        with self.assertRaises(ValueError):
            normalize_webhook(payload, "evt_6")

    def test_rail_and_amount_are_normalised(self) -> None:
        payload = webhook_body()
        payload["payload"]["payment"]["entity"]["method"] = "UPI"
        payload["payload"]["payment"]["entity"]["amount"] = "notanumber"
        event = normalize_webhook(payload, "evt_7")
        self.assertEqual(event.rail, PaymentRail.UPI_AUTOPAY)
        self.assertEqual(event.amount_paise, 0)
        self.assertEqual(event.event_type, EventType.PAYMENT_FAILED)


class IssuerExtractionTests(unittest.TestCase):
    """Which bank declined this, on a rail where the answer is not obvious.

    `bank` is populated for netbanking and eMandate. For **card** payments it
    is null and the issuer sits under `card.issuer`, so reading only `bank`
    put every card decline on every issuer into one "unknown" bucket.

    That is not a cosmetic loss. BankHealthMonitor keys on (issuer, rail), so
    one bad issuer marked the entire card rail degraded and parked healthy
    traffic in cooldown, and the release quota keys on the same tuple, so all
    of it shared a single per-tick budget. Per-issuer outage detection -- the
    mechanism the whole cooldown design exists for -- was inert on the
    dominant rail, and nothing failed loudly when it was. Hence these tests.
    """

    def _issuer(self, payment: dict) -> str:
        payload = webhook_body()
        payload["payload"]["payment"]["entity"] = payment
        return normalize_webhook(payload, "evt_issuer").issuer

    def test_a_card_declines_under_its_own_issuer(self) -> None:
        """The regression. This returned "unknown" for every card in the book."""
        issuer = self._issuer(
            {"id": "pay_1", "order_id": "order_1", "amount": 49900,
             "method": "card", "card": {"issuer": "HDFC", "network": "Visa"}}
        )
        self.assertEqual(issuer, "HDFC")

    def test_two_card_issuers_stay_distinct(self) -> None:
        """The property that matters: health is keyed per issuer, so two
        issuers collapsing to one label is what disabled outage detection."""
        base = {"id": "p", "order_id": "o", "amount": 49900, "method": "card"}
        first = self._issuer({**base, "card": {"issuer": "HDFC"}})
        second = self._issuer({**base, "card": {"issuer": "ICICI"}})
        self.assertNotEqual(first, second)

    def test_bank_still_wins_where_it_is_populated(self) -> None:
        """Netbanking and eMandate carry it at the top level; the card lookup
        must be a fallback, not a replacement."""
        issuer = self._issuer(
            {"id": "p", "order_id": "o", "amount": 49900,
             "method": "netbanking", "bank": "SBIN", "card": {"issuer": "HDFC"}}
        )
        self.assertEqual(issuer, "SBIN")

    def test_a_wallet_is_named_too(self) -> None:
        issuer = self._issuer(
            {"id": "p", "order_id": "o", "amount": 49900,
             "method": "wallet", "wallet": "paytm"}
        )
        self.assertEqual(issuer, "paytm")

    def test_nothing_identifiable_is_unknown_not_a_crash(self) -> None:
        issuer = self._issuer(
            {"id": "p", "order_id": "o", "amount": 49900, "method": "card"}
        )
        self.assertEqual(issuer, "unknown")

    def test_a_malformed_card_object_does_not_raise(self) -> None:
        """Provider payloads are attacker-adjacent input. A string where an
        object was expected must not become a 500 and a redelivery loop."""
        for card in ("not-an-object", 42, [], None):
            with self.subTest(card=card):
                issuer = self._issuer(
                    {"id": "p", "order_id": "o", "amount": 49900,
                     "method": "card", "card": card}
                )
                self.assertEqual(issuer, "unknown")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # Signature verification fails closed, so these fixtures opt in by name
        # rather than relying on a permissive default.
        patcher = mock.patch.dict(
            os.environ, {"RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"}, clear=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.app = create_app(
            settings=Settings(database_path=":memory:"),
            store=self.store,
            gateway=FakeRazorpayGateway(),
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def post_webhook(self, body: dict, event_id: str) -> dict:
        response = self.client.post(
            "/webhooks/razorpay", content=json.dumps(body), headers={"X-Razorpay-Event-Id": event_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_health_and_overview(self) -> None:
        self.assertEqual(
            self.client.get("/health").json(),
            {"status": "ok", "operator_auth": "disabled"},
        )
        overview = self.client.get("/overview").json()
        self.assertTrue(overview["simulation_enabled"])
        self.assertEqual(overview["cases"], [])
        self.assertEqual(overview["metrics"]["total_cases"], 0)
        self.assertIn("degraded_rails", overview)

    def test_webhook_then_dispatch_then_preview(self) -> None:
        first = self.post_webhook(webhook_body(), "evt_api_1")
        self.assertTrue(first["processed"])
        self.assertEqual(first["state"], "consent_required")

        # Redelivery of the same event id is a no-op.
        repeat = self.post_webhook(webhook_body(), "evt_api_1")
        self.assertFalse(repeat["processed"])

        dispatched = self.client.post("/actions/dispatch").json()
        self.assertEqual(dispatched["dispatched"], 1)
        case_id = dispatched["dispatched_case_ids"][0]

        detail = self.client.get(f"/cases/{case_id}").json()
        self.assertEqual(detail["state"], "link_sent")
        self.assertEqual(len(detail["actions"]), 1)

        preview = self.client.post(
            f"/cases/{case_id}/outreach-preview", json={"language": "hinglish"}
        ).json()
        self.assertTrue(preview["approved"])
        self.assertTrue(preview["preview_only"])
        self.assertIn("rzp.io", preview["final_message"])

        overview = self.client.get("/overview").json()
        self.assertEqual(overview["metrics"]["links_sent"], 1)
        self.assertEqual(overview["total_cases"], 1)

    def test_missing_event_id_is_rejected(self) -> None:
        response = self.client.post("/webhooks/razorpay", content=json.dumps(webhook_body()))
        self.assertEqual(response.status_code, 400)

    def test_invalid_json_body_is_rejected(self) -> None:
        response = self.client.post(
            "/webhooks/razorpay", content="{not json", headers={"X-Razorpay-Event-Id": "evt_bad"}
        )
        self.assertEqual(response.status_code, 400)

    def test_oversized_body_is_rejected(self) -> None:
        app = create_app(
            settings=Settings(database_path=":memory:", max_webhook_bytes=10),
            store=RecoveryStore(),
            gateway=FakeRazorpayGateway(),
        )
        with TestClient(app) as client:
            response = client.post(
                "/webhooks/razorpay",
                content=json.dumps(webhook_body()),
                headers={"X-Razorpay-Event-Id": "evt_big"},
            )
        self.assertEqual(response.status_code, 413)

    def test_signature_is_enforced_when_a_secret_is_configured(self) -> None:
        secret = "whsec_test"
        body = json.dumps(webhook_body(order_id="order_sig"))
        digest = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        with mock.patch.dict(os.environ, {"RAZORPAY_WEBHOOK_SECRET": secret}):
            bad = self.client.post(
                "/webhooks/razorpay",
                content=body,
                headers={"X-Razorpay-Event-Id": "evt_sig_1", "X-Razorpay-Signature": "wrong"},
            )
            self.assertEqual(bad.status_code, 401)

            good = self.client.post(
                "/webhooks/razorpay",
                content=body,
                headers={"X-Razorpay-Event-Id": "evt_sig_2", "X-Razorpay-Signature": digest},
            )
            self.assertEqual(good.status_code, 200, good.text)

    def test_unknown_case_returns_404(self) -> None:
        self.assertEqual(self.client.get("/cases/case_missing").status_code, 404)
        self.assertEqual(self.client.get("/cases/case_missing/actions").status_code, 404)
        self.assertEqual(
            self.client.post("/cases/case_missing/outreach-preview", json={}).status_code, 404
        )

    def test_preview_requires_an_active_link(self) -> None:
        self.post_webhook(webhook_body(order_id="order_nolink"), "evt_api_nolink")
        case_id = self.client.get("/cases").json()[0]["id"]
        response = self.client.post(f"/cases/{case_id}/outreach-preview", json={})
        self.assertEqual(response.status_code, 409)

    def test_demo_endpoints_run_end_to_end(self) -> None:
        race = self.client.post("/demo/race-condition").json()
        self.assertEqual(len(race["timeline"]), 4)
        self.assertIn("recovered_natural", race["timeline"][3])

        outreach = self.client.post("/demo/outreach-preview").json()
        self.assertTrue(outreach["preview"]["approved"])

    def test_cases_endpoint_paginates(self) -> None:
        for index in range(3):
            self.post_webhook(webhook_body(order_id=f"order_page_{index}"), f"evt_page_{index}")
        self.assertEqual(len(self.client.get("/cases?limit=2").json()), 2)
        self.assertEqual(len(self.client.get("/cases").json()), 3)


if __name__ == "__main__":
    unittest.main()


class SecurityTests(unittest.TestCase):
    """The two failure modes that fail open are worth pinning explicitly."""

    def _client(self) -> TestClient:
        store = RecoveryStore()
        self.addCleanup(store.close)
        app = create_app(
            settings=Settings(database_path=":memory:"),
            store=store,
            gateway=FakeRazorpayGateway(),
        )
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_unconfigured_signature_verification_refuses_traffic(self) -> None:
        """No secret and no explicit opt-in must mean no webhooks accepted.

        This previously returned silently, so a deploy that lost the secret
        from its environment accepted forged payment.failed events from anyone
        with only a startup log line as warning.
        """
        with mock.patch.dict(os.environ, {}, clear=True):
            response = self._client().post(
                "/webhooks/razorpay",
                content=json.dumps(webhook_body()),
                headers={"X-Razorpay-Event-Id": "evt_unsigned"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("RAZORPAY_WEBHOOK_SECRET", response.json()["detail"])

    def test_startup_warning_matches_what_the_endpoint_then_does(self) -> None:
        """A warning that contradicts the endpoint is worse than none.

        This said "signatures are NOT verified ... acceptable for local
        fixtures" in exactly the configuration where every webhook is refused
        with a 503. It was written before the endpoint failed closed and never
        updated, so an operator who read it and posted an event got a status
        the log had told them not to expect.
        """
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertLogs("app.api", level="WARNING") as captured:
                client = self._client()
            warning = "\n".join(captured.output)
            self.assertIn("503", warning)
            self.assertNotIn("NOT verified", warning)
            response = client.post(
                "/webhooks/razorpay",
                content=json.dumps(webhook_body()),
                headers={"X-Razorpay-Event-Id": "evt_warned"},
            )
        self.assertEqual(response.status_code, 503)

    def test_opt_in_startup_warning_admits_signatures_are_unverified(self) -> None:
        """The other branch has the opposite duty: say plainly that it is open."""
        with mock.patch.dict(os.environ, {"RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"}, clear=True):
            with self.assertLogs("app.api", level="WARNING") as captured:
                client = self._client()
            warning = "\n".join(captured.output)
            self.assertIn("NOT verified", warning)
            response = client.post(
                "/webhooks/razorpay",
                content=json.dumps(webhook_body()),
                headers={"X-Razorpay-Event-Id": "evt_opt_in"},
            )
        self.assertEqual(response.status_code, 200)

    def test_a_wrong_signature_is_rejected(self) -> None:
        body = json.dumps(webhook_body())
        with mock.patch.dict(os.environ, {"RAZORPAY_WEBHOOK_SECRET": "shh"}, clear=True):
            response = self._client().post(
                "/webhooks/razorpay",
                content=body,
                headers={
                    "X-Razorpay-Event-Id": "evt_bad_sig",
                    "X-Razorpay-Signature": "0" * 64,
                },
            )
        self.assertEqual(response.status_code, 401)

    def test_a_correct_signature_is_accepted(self) -> None:
        body = json.dumps(webhook_body())
        secret = "shh"
        digest = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        with mock.patch.dict(os.environ, {"RAZORPAY_WEBHOOK_SECRET": secret}, clear=True):
            response = self._client().post(
                "/webhooks/razorpay",
                content=body,
                headers={
                    "X-Razorpay-Event-Id": "evt_good_sig",
                    "X-Razorpay-Signature": digest,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)

    def test_a_non_ascii_signature_is_rejected_not_a_crash(self) -> None:
        """Tested at the function, because httpx refuses to transmit the header.

        Starlette decodes headers as latin-1, and hmac.compare_digest raises
        TypeError when handed a non-ASCII str -- which turned a junk signature
        into a 500 with a traceback rather than a clean rejection. Comparing
        bytes fixes it.
        """
        from app.api import _verify_signature

        with mock.patch.dict(
            os.environ, {"RAZORPAY_WEBHOOK_SECRET": "shh"}, clear=True
        ), self.assertRaises(HTTPException) as caught:
            _verify_signature(b"{}", "\xff\xff\xff")
        self.assertEqual(caught.exception.status_code, 401)

    def test_operator_token_guards_case_data_and_dispatch(self) -> None:
        """/cases leaks invoice ids, amounts and live payment-link URLs, and
        /actions/dispatch spends customers' contact budgets."""
        env = {"RAILPULSE_OPERATOR_TOKEN": "sekrit", "RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            client = self._client()
            self.assertEqual(client.get("/cases").status_code, 401)
            self.assertEqual(client.post("/actions/dispatch").status_code, 401)
            self.assertEqual(client.get("/cases", headers={"Authorization": "Bearer wrong"}).status_code, 401)
            ok = client.get("/cases", headers={"Authorization": "Bearer sekrit"})
            self.assertEqual(ok.status_code, 200)
            # /health must stay open so a load balancer can probe it.
            self.assertEqual(client.get("/health").status_code, 200)

    def test_no_operator_token_configured_leaves_the_guard_off(self) -> None:
        with mock.patch.dict(os.environ, {"RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"}, clear=True):
            self.assertEqual(self._client().get("/cases").status_code, 200)

    def test_health_says_whether_the_guard_is_on(self) -> None:
        """The dashboard is the only client of the guarded endpoints, and it
        was never taught the guard existed: turning RAILPULSE_OPERATOR_TOKEN on
        made every panel fail with no way to supply a token. It now asks here
        first, which is why this field has to be there and has to be honest."""
        with mock.patch.dict(
            os.environ, {"RAILPULSE_OPERATOR_TOKEN": "sekrit"}, clear=True
        ):
            body = self._client().get("/health").json()
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["operator_auth"], "required")

        with mock.patch.dict(os.environ, {}, clear=True):
            body = self._client().get("/health").json()
            self.assertEqual(body["operator_auth"], "disabled")

    def test_health_never_reveals_the_token(self) -> None:
        with mock.patch.dict(
            os.environ, {"RAILPULSE_OPERATOR_TOKEN": "sekrit"}, clear=True
        ):
            self.assertNotIn("sekrit", self._client().get("/health").text)

    def test_the_dashboard_sends_the_token_it_is_given(self) -> None:
        """A static check on the shipped page, because the bug was not that the
        auth was wrong -- it was that the only caller never sent a header at
        all, and no server-side test could have noticed."""
        page = (
            Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Authorization", page)
        self.assertIn("Bearer ${state.operatorToken}", page)
        self.assertIn("operator_auth", page)
        self.assertIn("response.status === 401", page)

    def test_the_dashboard_does_not_persist_the_token(self) -> None:
        """It is a bearer secret for endpoints that hand out other bearer
        secrets. It lives in the tab and dies with it."""
        page = (
            Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
        ).read_text(encoding="utf-8")
        for call in (
            "localStorage.setItem",
            "localStorage.getItem",
            "sessionStorage.setItem",
            "sessionStorage.getItem",
            "document.cookie",
        ):
            self.assertNotIn(call, page, f"the dashboard persists the token via {call}")


class ReopenEndpointTests(unittest.TestCase):
    """The escape hatch from manual review, over HTTP.

    Mostly a test of what the endpoint refuses. It is authenticated and it
    looks like ordinary operations, which is exactly why it must not be able
    to resume contacting someone who opted out.
    """

    def setUp(self) -> None:
        patcher = mock.patch.dict(
            os.environ, {"RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"}, clear=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.app = create_app(
            settings=Settings(database_path=":memory:"),
            store=self.store,
            gateway=FakeRazorpayGateway(),
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def _park(self, error_reason: str = "") -> str:
        """An absent failure reason classifies as UNKNOWN, which parks it."""
        body = webhook_body(error_reason=error_reason)
        response = self.client.post(
            "/webhooks/razorpay",
            content=json.dumps(body),
            headers={"X-Razorpay-Event-Id": "evt_park"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "manual_review")
        return self.client.get("/cases").json()[0]["id"]

    def test_reopening_with_a_classification_returns_it_to_the_engine(self) -> None:
        case_id = self._park()
        response = self.client.post(
            f"/cases/{case_id}/reopen",
            json={"note": "issuer confirmed a replaced card", "failure_class": "customer_action"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "consent_required")
        self.assertEqual(response.json()["attempt_count"], 0)

    def test_a_note_is_required(self) -> None:
        """An audit entry saying a human overrode the engine, but not why, is
        not worth writing down."""
        case_id = self._park()
        self.assertEqual(
            self.client.post(f"/cases/{case_id}/reopen", json={}).status_code, 422
        )
        self.assertEqual(
            self.client.post(f"/cases/{case_id}/reopen", json={"note": "x"}).status_code, 422
        )

    def test_reopening_a_case_that_is_not_parked_is_refused(self) -> None:
        response = self.client.post(
            "/webhooks/razorpay",
            content=json.dumps(webhook_body()),
            headers={"X-Razorpay-Event-Id": "evt_live"},
        )
        self.assertEqual(response.json()["state"], "consent_required")
        case_id = self.client.get("/cases").json()[0]["id"]
        refused = self.client.post(f"/cases/{case_id}/reopen", json={"note": "why not"})
        self.assertEqual(refused.status_code, 409)
        self.assertIn("manual_review", refused.json()["detail"])

    def test_an_unknown_case_is_a_404(self) -> None:
        response = self.client.post("/cases/case_nope/reopen", json={"note": "ghost"})
        self.assertEqual(response.status_code, 404)

    def test_the_endpoint_is_behind_the_operator_guard(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RAILPULSE_OPERATOR_TOKEN": "sekrit", "RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"},
        ):
            response = self.client.post("/cases/whatever/reopen", json={"note": "open up"})
            self.assertEqual(response.status_code, 401)


class RobustnessTests(unittest.TestCase):
    """Inputs an attacker chooses must not look like server faults.

    Razorpay reads a 5xx as "try again later" and retries on its full backoff
    schedule, so a 500 on a malformed body is not just untidy -- it turns one
    bad payload into an indefinite redelivery loop.
    """

    def test_absurd_amounts_do_not_crash_normalisation(self) -> None:
        def body(amount: object) -> dict:
            payload = webhook_body()
            payload["payload"]["payment"]["entity"]["amount"] = amount
            return payload

        # json.loads("1e400") is inf, and int(inf) raises OverflowError.
        infinity = json.loads('{"a": 1e400}')["a"]
        self.assertEqual(normalize_webhook(body(infinity), "e1").amount_paise, 0)
        # A 400-digit integer survives int() and then dies in the SQLite bind.
        huge = normalize_webhook(body(int("9" * 400)), "e2").amount_paise
        self.assertLessEqual(huge, MAX_AMOUNT_PAISE)
        self.assertEqual(normalize_webhook(body(-5000), "e3").amount_paise, 0)
        self.assertEqual(normalize_webhook(body(49900), "e4").amount_paise, 49900)

    def test_a_clamped_amount_still_stores(self) -> None:
        """The point of the ceiling: the value has to survive the DB bind."""
        store = RecoveryStore()
        self.addCleanup(store.close)
        app = create_app(
            settings=Settings(database_path=":memory:"),
            store=store,
            gateway=FakeRazorpayGateway(),
        )
        payload = webhook_body(order_id="order_huge")
        payload["payload"]["payment"]["entity"]["amount"] = int("9" * 400)
        with mock.patch.dict(
            os.environ, {"RAILPULSE_ALLOW_UNSIGNED_WEBHOOKS": "1"}, clear=True
        ), TestClient(app) as client:
            response = client.post(
                "/webhooks/razorpay",
                content=json.dumps(payload),
                headers={"X-Razorpay-Event-Id": "evt_huge"},
            )
        self.assertEqual(response.status_code, 200, response.text)


class RetentionTests(unittest.TestCase):
    def test_idempotency_claims_are_pruned(self) -> None:
        """Nothing aged this table out, so it grew one row per webhook forever."""
        store = RecoveryStore()
        self.addCleanup(store.close)
        old = datetime(2026, 1, 1, tzinfo=UTC)
        recent = datetime(2026, 8, 21, tzinfo=UTC)
        with store.transaction():
            store.claim_event("evt_old", old)
            store.claim_event("evt_recent", recent)

        removed = store.prune_processed_events(datetime(2026, 6, 1, tzinfo=UTC))
        self.assertEqual(removed, 1)
        # The pruned claim is now free again; the retained one is not.
        with store.transaction():
            self.assertTrue(store.claim_event("evt_old", recent))
            self.assertFalse(store.claim_event("evt_recent", recent))

    def test_claiming_is_not_dependent_on_an_exception_class(self) -> None:
        store = RecoveryStore()
        self.addCleanup(store.close)
        now = datetime(2026, 8, 21, tzinfo=UTC)
        with store.transaction():
            self.assertTrue(store.claim_event("evt_once", now))
            self.assertFalse(store.claim_event("evt_once", now))
