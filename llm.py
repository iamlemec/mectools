# generic llm fetch routines

import os
import json
import requests
import subprocess

# presets for known llm providers
LLM_PROVIDERS = {
    'anthropic': {
        'url': 'https://api.anthropic.com/v1/messages',
        'model': 'claude-3-5-sonnet-20240620',
        'auth_header': 'x-api-key',
        'api_key_name': 'ANTHROPIC_API_KEY',
        'has_system_role': False,
        'extra_headers': {
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'prompt-caching-2024-07-31',
        },
    },
    'openai': {
        'url': 'https://api.openai.com/v1/chat/completions',
        'model': 'gpt-4o',
        'auth_header': 'Authorization',
        'auth_prefix': 'Bearer ',
        'api_key_name': 'OPENAI_API_KEY', 
    },
    'fireworks': {
        'url': 'https://api.fireworks.ai/inference/v1/chat/completions',
        'model': 'accounts/fireworks/models/llama-v3-70b-instruct',
        'auth_header': 'Authorization',
        'auth_prefix': 'Bearer ',
        'api_key_name': 'FIREWORKS_API_KEY',
    },
}

# default local system prompt
SYSTEM = 'You are a helpful and knowledgable AI assistant. Answer the queries provided to the best of your ability.'

def get_llm_response(
    prompt, provider='local', system=SYSTEM, url=None, port=8000,
    model=None, max_tokens=1024, **kwargs
):
    # base payload
    payload = {'max_tokens': max_tokens, **kwargs}
    headers = {'Content-Type': 'application/json'}

    # handle specific providers
    if provider == 'local':
        # default for local provision (llama-cpp-python)
        if url is None:
            url = f'http://localhost:{port}/v1/chat/completions'

        # add in chat messages
        payload['messages'] = [
            {'role': 'system', 'content': system},
            {'role': 'user'  , 'content': prompt},
        ]
    else:
        # external provider
        prov = LLM_PROVIDERS[provider]
    
        # get url to request
        if url is None:
            url = prov['url']

        # get model to use
        if model is None:
            model = prov['model']

        # find api key
        if (api_key := os.environ.get(key_env := prov['api_key_name'])) == None:
            raise Exception('Cannot find API key in {key_env}')

        # get auth params
        auth_header = prov['auth_header']
        auth_prefix = prov.get('auth_prefix', '')
        auth_value = f'{auth_prefix}{api_key}'

        # augment payload and headers
        headers[auth_header] = auth_value
        payload['model'] = model

        # handle system prompt
        if prov.get('has_system_role', True):
            payload['messages'] = [
                {'role': 'system', 'content': system},
                {'role': 'user'  , 'content': prompt},
            ]
        else:
            payload['system'] = system
            payload['messages'] = [
                {'role': 'user'  , 'content': prompt},
            ]

        # add any extra headers
        if 'extra_headers' in prov:
            headers.update(prov['extra_headers'])

    # request response and return
    data = json.dumps(payload)
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    reply = response.json()

    # get text reply from data
    text = None
    if 'choices' in reply and len(reply['choices']) > 0:
        choice = reply['choices'][0]
        if 'text' in choice:
            text = choice['text']
        elif 'message' in choice:
            text = choice['message']['content']
    elif 'content' in reply and len(reply['content']) > 0:
        content = reply['content'][0]
        if 'text' in content:
            text = content['text']

    # check if we have reply
    if text is None:
        raise Exception('No valid text response generated')

    # otherwise result text
    return text.strip()

def run_llama_server(model, n_gpu_layers=-1, *args):
    opts = ['--model', model, '--n_gpu_layers', n_gpu_layers, *args]
    cmds = ['python', '-m', 'llama_cpp.server', *opts]
    subprocess.run([str(x) for x in cmds])

if __name__ == '__main__':
    import fire
    fire.Fire({
        'serve': run_llama_server,
        'chat': get_llm_response,
    })
