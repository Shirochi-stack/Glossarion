from unified_api_client import UnifiedResponse


def test_unified_response_truncation_flags():
    for reason in ['length', 'max_tokens', 'stop_sequence_limit', 'truncated', 'incomplete']:
        r = UnifiedResponse(content='x', finish_reason=reason)
        assert r.is_truncated is True
        assert r.is_complete is False


def test_unified_response_completion_flags():
    for reason in [None, 'stop', 'complete', 'end_turn', 'finished']:
        r = UnifiedResponse(content='x', finish_reason=reason)
        assert r.is_complete is True
        assert r.is_truncated is False


def test_unified_response_error_detection():
    r1 = UnifiedResponse(content='', finish_reason='error')
    assert r1.is_error is True

    r2 = UnifiedResponse(content='', finish_reason='stop', error_details={'message': 'fail'})
    assert r2.is_error is True

    r3 = UnifiedResponse(content='ok', finish_reason='stop')
    assert r3.is_error is False
