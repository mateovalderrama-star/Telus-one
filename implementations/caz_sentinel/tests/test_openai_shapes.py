from caz_sentinel.openai_shapes import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Usage,
    build_suppressed_response, build_pass_response,
)

def test_request_parses_minimal():
    req = ChatCompletionRequest.model_validate({
        "model": "pythia-6.9b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.messages[0].content == "hi"
    assert req.stream is False

def test_suppressed_response_shape():
    r = build_suppressed_response(request_id="req-1", model="pythia-6.9b",
                                  refusal="nope", prompt_tokens=4)
    assert r.choices[0].finish_reason == "content_filter"
    assert r.choices[0].message.content == "nope"
    assert r.usage.completion_tokens == 0
    assert r.id.startswith("chatcmpl-")

def test_pass_response_shape():
    r = build_pass_response(request_id="req-2", model="pythia-6.9b",
                            completion="hello there", prompt_tokens=3, completion_tokens=5)
    assert r.choices[0].finish_reason == "stop"
    assert r.choices[0].message.content == "hello there"
    assert r.usage.completion_tokens == 5
