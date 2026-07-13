from backends.model_provider import get_model
m = get_model(max_completion_tokens=2048, max_tokens=2048)
resp = m.invoke("Say the word OK and nothing else.")
print("CONTENT:", repr(resp.content))
print("METADATA:", resp.response_metadata)
print("ADDITIONAL:", resp.additional_kwargs)
