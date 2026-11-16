import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from rag.generate import (
    OllamaError,
    answer_coverage,
    build_plain_messages,
    build_rag_messages,
    format_context,
    is_refusal,
    ollama_chat,
)


def make_hits():
    return [
        SimpleNamespace(title="Egri vár", section="Története", text="Az egri vár 1552-ben állt ellen."),
        SimpleNamespace(title="Diósgyőri vár", section="", text="Nagy Lajos idején élte fénykorát."),
    ]


def test_format_context_numbers_sources_and_shows_section():
    context = format_context(make_hits())
    assert context.startswith("[1] Egri vár › Története\nAz egri vár")
    assert "[2] Diósgyőri vár\nNagy Lajos" in context
    assert "›" not in context.split("[2]")[1]


def test_rag_messages_contain_question_and_sources():
    messages = build_rag_messages("Mikor ostromolták?", make_hits())
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    assert "Mikor ostromolták?" in messages[1]["content"]
    assert "1552-ben" in messages[1]["content"]


def test_plain_messages_have_no_sources():
    messages = build_plain_messages("Mikor ostromolták?")
    assert "Források" not in messages[1]["content"]


@pytest.mark.parametrize("answer", [
    "A megadott forrásokból nem derül ki.", "Nem tudom.", "", "  ",
    "Sajnos erről nincs információ a szövegben.",
])
def test_is_refusal_true(answer):
    assert is_refusal(answer)


def test_is_refusal_false_for_real_answer():
    assert not is_refusal("A kastély 101 helyiségből áll.")


def test_answer_coverage_handles_inflection():
    assert answer_coverage("Riolittufából.", "A falak riolittufa anyagból épültek.") == 1.0
    assert answer_coverage("Victor Rumpelmayer.", "Victor Rumpelmayer tervezte.") == 1.0
    assert answer_coverage("Victor Rumpelmayer.", "Ybl Miklós tervezte.") == 0.0
    assert answer_coverage("", "bármi") == 0.0


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] == "hianyzo":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"model not found"}')
            return
        reply = {"message": {"content": " 101 helyiség. "}, "total_duration": 2_500_000_000, "eval_count": 7}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_ollama_chat_parses_response(server):
    result = ollama_chat(build_plain_messages("?"), "valami", server)
    assert result.text == "101 helyiség."
    assert result.seconds == pytest.approx(2.5)
    assert result.output_tokens == 7


def test_ollama_chat_unknown_model_suggests_pull(server):
    with pytest.raises(OllamaError, match="ollama pull hianyzo"):
        ollama_chat(build_plain_messages("?"), "hianyzo", server)


def test_ollama_chat_unreachable():
    with pytest.raises(OllamaError, match="Nem érem el az Ollamát"):
        ollama_chat(build_plain_messages("?"), "x", "http://127.0.0.1:9", timeout=2)


def test_rag_prompt_variants_differ_and_unknown_fails():
    strict = build_rag_messages("?", make_hits(), "strict")[0]["content"]
    balanced = build_rag_messages("?", make_hits(), "balanced")[0]["content"]
    assert strict != balanced
    assert "A megadott forrásokból nem derül ki." in balanced
    with pytest.raises(ValueError):
        build_rag_messages("?", make_hits(), "ismeretlen")