"""Mobile Sigma streams must preserve glossary separators and source formatting."""
import pytest
import test_nanogpt_quiet_stream as transport

@pytest.fixture
def server(monkeypatch):
    fixture = transport.server.__wrapped__()
    instance = next(fixture)
    monkeypatch.setenv('GLOSSARION_LOCAL_LLM_URL', f'http://127.0.0.1:{instance.server_address[1]}/v1')
    yield instance
    next(fixture, None)


@pytest.mark.parametrize("parts", [
    ['```csv', '\n', 'original,translated,type', '\n', '기사,Knight,title', '\n', '왕,King,title', '\n', '```'],
    ['First paragraph.', '\n\n', 'Second', ' ', 'paragraph.', '\r\n', '\t', 'Indented.'],
])
def test_local_stream_matches_nonstream_whitespace(monkeypatch, tmp_path, server, parts):
    monkeypatch.setattr(transport, 'WORDS', parts)
    streamed = transport._send(monkeypatch, tmp_path, server, 'local', streaming=True)
    buffered = transport._send(monkeypatch, tmp_path, server, 'local', streaming=False)
    assert streamed.content == buffered.content == ''.join(parts)
    assert server.bodies[0]['stream'] is True
    assert not server.bodies[1].get('stream')


def test_empty_thought_tags_still_removed_without_removing_newlines(monkeypatch, tmp_path, server):
    monkeypatch.setattr(transport, 'WORDS', ['First', '</thought>', '\n', 'Second'])
    response = transport._send(monkeypatch, tmp_path, server, 'local', streaming=True)
    assert response.content == 'First\nSecond'
