import requests
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

# Initialize Traceloop
Traceloop.init(app_name="vllm-client-traceloop")

@task(name="prepare_headers")
def prepare_headers():
    return {}

@task(name="create_payload")
def create_payload(prompt):
    return {
        "model": "ibm-granite/granite-3.0-2b-instruct",
        "prompt": prompt,
        "max_tokens": 10,
        "n": 1,
        "best_of": 1,
        "use_beam_search": "false",
        "temperature": 0.0,
    }

@task(name="make_vllm_request")
def make_vllm_request(url, payload):
    return requests.post(url, json=payload)

@task(name="process_response")
def process_response(response):
    return response.json()

@workflow(name="vllm_client_workflow")
def run_chat():
    vllm_url = "http://9.30.109.130:8000/v1/completions"
    prompt = "San Francisco is a"
    
    #headers = prepare_headers()
    payload = create_payload(prompt)
    response = make_vllm_request(vllm_url, payload)
    result = process_response(response)
    
    return result

if __name__ == "__main__":
    result = run_chat()
    print("Chat Response:", result)
