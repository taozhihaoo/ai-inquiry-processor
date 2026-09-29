"""HTTP-level tests for the desk REST API (auth, validation, status codes, flows)."""

import pytest


@pytest.fixture
def api(desk_api):
    return desk_api


@pytest.fixture
def customer_id(api):
    response = api.client.post(
        "/customers",
        json={"name": "Alice Example", "email": "alice@example.com"},
        headers=api.admin_headers,
    )
    assert response.status_code == 201
    return response.json()["id"]


@pytest.fixture
def ticket_id(api, customer_id):
    response = api.client.post(
        "/tickets",
        json={"customer_id": customer_id, "subject": "Cannot log in",
              "message": "I cannot log into my account.", "priority": "high"},
        headers=api.admin_headers,
    )
    assert response.status_code == 201
    return response.json()["id"]


class TestAuthentication:
    def test_missing_header_is_401(self, api):
        response = api.client.get("/tickets")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_unknown_agent_is_401(self, api):
        response = api.client.get("/tickets", headers={"X-Agent-Id": "f" * 32})
        assert response.status_code == 401

    def test_deactivated_agent_is_401(self, api):
        inactive, _ = api.desk_store.insert_agent(
            name="Gone", email="gone@x.com", role="agent", active=False
        )
        response = api.client.get("/tickets", headers={"X-Agent-Id": inactive.id})
        assert response.status_code == 401
        assert "deactivated" in response.json()["error"]["message"]

    def test_agent_role_forbidden_on_admin_endpoints(self, api):
        assert api.client.get("/audit", headers=api.agent_headers).status_code == 403
        assert api.client.post("/tags", json={"name": "x"}, headers=api.agent_headers).status_code == 403
        assert api.client.post(
            "/agents", json={"name": "x", "email": "x@x.com"}, headers=api.agent_headers
        ).status_code == 403

    def test_admin_endpoints_work_for_admin(self, api):
        assert api.client.get("/audit", headers=api.admin_headers).status_code == 200
        assert api.client.get("/tickets", headers=api.admin_headers).status_code == 200


class TestCustomers:
    def test_create_get_search(self, api, customer_id):
        fetched = api.client.get(f"/customers/{customer_id}", headers=api.admin_headers)
        assert fetched.status_code == 200
        assert fetched.json()["email"] == "alice@example.com"

        listed = api.client.get(
            "/customers?search=alice", headers=api.admin_headers
        ).json()
        assert listed["total"] == 1

    def test_duplicate_email_returns_existing_with_200(self, api):
        body = {"name": "Alice", "email": "dup@example.com"}
        first = api.client.post("/customers", json=body, headers=api.admin_headers)
        second = api.client.post("/customers", json=body, headers=api.admin_headers)
        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]

    @pytest.mark.parametrize("body", [
        {"name": "", "email": "a@b.com"},
        {"name": "X", "email": "bad-email"},
        {"email": "a@b.com"},
    ])
    def test_validation_422(self, api, body):
        response = api.client.post("/customers", json=body, headers=api.admin_headers)
        assert response.status_code == 422

    def test_unknown_customer_404(self, api):
        assert api.client.get("/customers/ghost", headers=api.admin_headers).status_code == 404

    def test_customer_tickets(self, api, customer_id, ticket_id):
        response = api.client.get(
            f"/customers/{customer_id}/tickets", headers=api.admin_headers
        )
        assert response.status_code == 200
        assert response.json()["items"][0]["id"] == ticket_id


class TestTickets:
    def test_create_and_detail(self, api, customer_id, ticket_id):
        detail = api.client.get(f"/tickets/{ticket_id}", headers=api.admin_headers)
        assert detail.status_code == 200
        body = detail.json()
        assert body["ticket"]["status"] == "open"
        assert body["customer"]["id"] == customer_id
        assert [message["body"] for message in body["messages"]] == [
            "I cannot log into my account."
        ]

    def test_unknown_ticket_404(self, api):
        assert api.client.get("/tickets/ghost", headers=api.admin_headers).status_code == 404

    def test_create_with_bad_priority_422(self, api, customer_id):
        response = api.client.post(
            "/tickets",
            json={"customer_id": customer_id, "subject": "S", "message": "M",
                  "priority": "asap"},
            headers=api.admin_headers,
        )
        assert response.status_code == 422

    def test_patch_status_and_invalid_transition_409(self, api, ticket_id):
        ok = api.client.patch(
            f"/tickets/{ticket_id}", json={"status": "pending"},
            headers=api.admin_headers,
        )
        assert ok.status_code == 200
        assert ok.json()["status"] == "pending"

        bad = api.client.patch(
            f"/tickets/{ticket_id}", json={"status": "closed"},
            headers=api.admin_headers,
        )
        assert bad.status_code == 409
        assert bad.json()["error"]["code"] == "conflict"

    def test_patch_empty_body_422(self, api, ticket_id):
        response = api.client.patch(f"/tickets/{ticket_id}", json={}, headers=api.admin_headers)
        assert response.status_code == 422

    def test_patch_invalid_status_422(self, api, ticket_id):
        response = api.client.patch(
            f"/tickets/{ticket_id}", json={"status": "done"}, headers=api.admin_headers
        )
        assert response.status_code == 422

    def test_duplicate_content_409(self, api, customer_id, ticket_id):
        response = api.client.post(
            "/tickets",
            json={"customer_id": customer_id, "subject": "Whatever",
                  "message": "i cannot log into my account."},
            headers=api.admin_headers,
        )
        assert response.status_code == 409
        assert response.json()["error"]["existing_ticket_id"] == ticket_id

    def test_inbox_filters(self, api, customer_id, ticket_id):
        api.client.patch(
            f"/tickets/{ticket_id}", json={"priority": "urgent"}, headers=api.admin_headers
        )
        open_list = api.client.get("/tickets?status=open", headers=api.admin_headers).json()
        assert open_list["total"] == 1
        urgent = api.client.get("/tickets?priority=urgent", headers=api.admin_headers).json()
        assert [item["id"] for item in urgent["items"]] == [ticket_id]
        unassigned = api.client.get("/tickets?assignee=none", headers=api.admin_headers).json()
        assert unassigned["total"] == 1

    def test_search(self, api, ticket_id):
        found = api.client.get("/tickets?search=log", headers=api.admin_headers).json()
        assert [item["id"] for item in found["items"]] == [ticket_id]

    def test_assignment_flow(self, api, ticket_id):
        agents = api.client.get("/agents", headers=api.admin_headers).json()
        agent_id = agents["items"][0]["id"]
        assigned = api.client.post(
            f"/tickets/{ticket_id}/assign", json={"agent_id": agent_id},
            headers=api.admin_headers,
        )
        assert assigned.json()["assignee_id"] == agent_id
        mine = api.client.get(f"/agents/{agent_id}/tickets", headers=api.admin_headers).json()
        assert mine["total"] == 1
        unassigned = api.client.post(
            f"/tickets/{ticket_id}/unassign", headers=api.admin_headers
        )
        assert unassigned.json()["assignee_id"] is None

    def test_history(self, api, customer_id, ticket_id):
        api.client.patch(
            f"/tickets/{ticket_id}", json={"status": "pending"}, headers=api.admin_headers
        )
        history = api.client.get(f"/tickets/{ticket_id}/history", headers=api.admin_headers)
        event_types = [event["event_type"] for event in history.json()["items"]]
        assert event_types[0] == "ticket_created"
        assert "status_changed" in event_types


class TestNotesMessagesTags:
    def test_note_flow(self, api, ticket_id):
        created = api.client.post(
            f"/tickets/{ticket_id}/notes", json={"body": "Checked logs."},
            headers=api.agent_headers,
        )
        assert created.status_code == 201
        assert created.json()["author_name"] == "Sam Agent"
        listed = api.client.get(f"/tickets/{ticket_id}/notes", headers=api.agent_headers)
        assert listed.json()["total"] == 1

    def test_message_flow(self, api, ticket_id):
        created = api.client.post(
            f"/tickets/{ticket_id}/messages", json={"body": "Still broken."},
            headers=api.agent_headers,
        )
        assert created.status_code == 201
        detail = api.client.get(f"/tickets/{ticket_id}", headers=api.admin_headers).json()
        assert len(detail["messages"]) == 2

    def test_tag_flow(self, api, ticket_id):
        tag = api.client.post(
            "/tags", json={"name": "billing"}, headers=api.admin_headers
        ).json()
        linked = api.client.post(
            f"/tickets/{ticket_id}/tags", json={"tag_id": tag["id"]},
            headers=api.agent_headers,
        )
        assert linked.status_code == 200
        tags = api.client.get(f"/tickets/{ticket_id}/tags", headers=api.agent_headers).json()
        assert [item["name"] for item in tags["items"]] == ["billing"]

        removed = api.client.delete(
            f"/tickets/{ticket_id}/tags/{tag['id']}", headers=api.admin_headers
        )
        assert removed.status_code == 204
        assert api.client.get(f"/tickets/{ticket_id}/tags", headers=api.agent_headers).json()["total"] == 0

    def test_unknown_tag_link_404(self, api, ticket_id):
        response = api.client.post(
            f"/tickets/{ticket_id}/tags", json={"tag_id": "ghost"},
            headers=api.admin_headers,
        )
        assert response.status_code == 404


class TestDeskAI:
    def test_analyze_endpoint(self, api, ticket_id):
        response = api.client.post(
            f"/tickets/{ticket_id}/ai/analyze", headers=api.agent_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert "may require human review" in body["note"]
        assert body["analysis"]["category"] in (
            "Sales", "Technical Support", "Billing", "General Question",
        )
        assert body["analysis"]["sentiment"] in ("positive", "neutral", "negative")

    def test_suggest_reply_endpoint_is_draft(self, api, ticket_id):
        response = api.client.post(
            f"/tickets/{ticket_id}/ai/suggest-reply", headers=api.agent_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["suggested_reply"]
        assert "never auto-sent" in body["note"]

        detail = api.client.get(f"/tickets/{ticket_id}", headers=api.admin_headers).json()
        assert len(detail["messages"]) == 1  # draft is not a conversation message

    def test_ai_on_unknown_ticket_404(self, api):
        assert api.client.post("/tickets/ghost/ai/analyze", headers=api.agent_headers).status_code == 404


class TestLegacyEndpointsPreserved:
    def test_health_and_inquiries_still_work(self, api):
        assert api.client.get("/health").json()["status"] == "ok"
        response = api.client.post(
            "/inquiries",
            json={"customer_name": "Legacy", "message": "printer on fire"},
        )
        assert response.status_code == 201
        assert response.json()["status"] == "success"
