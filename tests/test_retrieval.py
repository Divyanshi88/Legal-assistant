from types import SimpleNamespace

import query_database


class FakeCompletions:
    def __init__(self, content):
        self.content = content

    def create(self, **_kwargs):
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def test_retriever_finds_domestic_violence_source():
    retriever = query_database.LexicalRetriever(query_database.load_source_documents())

    results = retriever.search("What is domestic violence?")

    assert results
    assert all(result.metadata["page"] >= 1 for result in results)


def test_pipeline_accepts_valid_source_citation():
    pipeline = query_database.EnhancedRAGPipeline(
        client=FakeClient("Domestic violence includes specified conduct. [S1]")
    )

    result = pipeline.query_with_sources("What is domestic violence?")

    assert result["sources"]
    assert "[S1]" in result["answer"]


def test_pipeline_rejects_uncited_answer():
    pipeline = query_database.EnhancedRAGPipeline(
        client=FakeClient("This answer has no source citation.")
    )

    result = pipeline.query_with_sources("What is domestic violence?")

    assert result["sources"] == []
    assert "source-backed" in result["answer"]


def test_irrelevant_question_is_refused_without_model_call():
    pipeline = query_database.EnhancedRAGPipeline(
        client=FakeClient("Should never be used.")
    )

    result = pipeline.query_with_sources("How do I repair a bicycle derailleur?")

    assert result["sources"] == []
    assert "enough relevant information" in result["answer"]


def test_pipeline_works_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pipeline = query_database.EnhancedRAGPipeline()

    result = pipeline.query_with_sources("What is domestic violence?")

    assert result["sources"]
    assert "[S1]" in result["answer"]
    assert "relevant passages" in result["answer"]
