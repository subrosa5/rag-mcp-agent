import json

from src.grounding import evidence_item, validate_draft

ITEM = evidence_item("deployment.txt", "Rollback takes 3 minutes, after approval.")


def draft(**overrides):
    value = {
        "status": "answered",
        "claims": [
            {
                "text": "Rollback takes 3 minutes after approval.",
                "citations": [
                    {
                        "evidence_id": ITEM["id"],
                        "quote": "Rollback takes 3 minutes, after approval.",
                    }
                ],
            }
        ],
    }
    value.update(overrides)
    return value


def test_valid_citation_and_snapshot():
    result = validate_draft(json.dumps(draft()), [ITEM])
    assert result["status"] == "answered"
    assert result["answer"].endswith("[1]")
    assert result["sources"][0]["content"] == ITEM["content"]


def test_fabricated_quote_rejected():
    data = draft()
    data["claims"][0]["citations"][0]["quote"] = "Rollback takes 1 minute"
    assert validate_draft(json.dumps(data), [ITEM])["reason"] == "invalid_citation"


def test_fabricated_source_rejected():
    data = draft()
    data["claims"][0]["citations"][0]["evidence_id"] = "invented"
    assert validate_draft(json.dumps(data), [ITEM])["status"] == "insufficient_evidence"


def test_uncited_claim_rejected():
    data = draft()
    data["claims"][0]["citations"] = []
    assert validate_draft(json.dumps(data), [ITEM])["status"] == "insufficient_evidence"


def test_malformed_or_extra_fields_fail_closed():
    for raw in [
        "no JSON",
        "{}",
        json.dumps(draft(secret="injected")),
        json.dumps(draft(claims=[])),
    ]:
        assert validate_draft(raw, [ITEM])["status"] == "insufficient_evidence"


def test_abstention_discards_model_claims():
    result = validate_draft(json.dumps(draft(status="insufficient_evidence")), [ITEM])
    assert result["sources"] == [] and result["claims"] == []


def test_evidence_id_changes_with_content_or_source():
    assert evidence_item("a", "old")["id"] != evidence_item("a", "new")["id"]
    assert evidence_item("a", "same")["id"] != evidence_item("b", "same")["id"]
