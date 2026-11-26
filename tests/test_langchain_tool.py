"""Check the model-facing tool contract against a real tenant-filtered store."""
from knowledge_retrieval.langchain_tool import make_search_tool
from knowledge_retrieval.retrieval import Store


def test_search_tool_keeps_tenant_outside_model_arguments(tmp_path):
    store = Store(str(tmp_path / "knowledge.db"))
    store.ingest("demo", "runbook", "incident handoff demo procedure")
    store.ingest("other", "private", "incident handoff private procedure")
    tool = make_search_tool(store, "demo")
    schema = tool.get_input_schema().model_json_schema()
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]

    # Even an extra tenant argument cannot redirect the closure's tenant scope.
    hits = tool.invoke({"query": "incident handoff", "tenant": "other"})
    assert hits == store.search("demo", "incident handoff")
    assert hits
    assert {hit["tenant"] for hit in hits} == {"demo"}
    assert {hit["source"] for hit in hits} == {"runbook"}
