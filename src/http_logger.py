import json
import os
from datetime import datetime
from pathlib import Path

# Global state - don't import requests yet!
_log_folder = None
_patched = False

def _save_http_log(method, url, headers, body, response_status=None, response_headers=None, response_body=None):
    """Save HTTP request/response to a JSON file"""
    try:
        if not _log_folder:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = _log_folder / f"http_{timestamp}.json"
        
        # Redact sensitive headers
        safe_headers = {}
        for k, v in (headers or {}).items():
            if any(sensitive in k.lower() for sensitive in ['authorization', 'api-key', 'key']):
                safe_headers[k] = "***REDACTED***"
            else:
                safe_headers[k] = v
        
        safe_response_headers = {}
        for k, v in (response_headers or {}).items():
            safe_response_headers[k] = v
        
        # Try to parse body as JSON if it's a dict or string
        request_body = body
        if isinstance(body, dict):
            request_body = body
        elif isinstance(body, str):
            try:
                request_body = json.loads(body)
            except:
                request_body = body
        elif isinstance(body, bytes):
            try:
                request_body = json.loads(body.decode('utf-8'))
            except:
                request_body = "<binary data>"
        elif body:
            request_body = str(body)
        
        # Try to parse response body
        parsed_response_body = response_body
        if isinstance(response_body, str):
            try:
                parsed_response_body = json.loads(response_body)
            except:
                parsed_response_body = response_body
        elif isinstance(response_body, dict):
            parsed_response_body = response_body
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request": {
                "method": method,
                "url": url,
                "headers": safe_headers,
                "body": request_body
            }
        }
        
        if response_status is not None:
            log_data["response"] = {
                "status_code": response_status,
                "headers": safe_response_headers,
                "body": parsed_response_body
            }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass  # Silently fail - we don't want to break the app

def enable_detailed_http_logging(log_folder="http_requests"):
    """
    Enable HTTP request/response logging by monkey-patching requests.
    This captures EVERYTHING that goes over the wire.
    """
    global _log_folder, _patched
    
    # Allow disabling via env (used by render workers)
    try:
        if os.environ.get('DISABLE_HTTP_LOGGER', '') == '1' or os.environ.get('GLOSSARION_RENDER_WORKER', '') == '1':
            return None
    except Exception:
        pass
    
    if _patched:
        return _log_folder
    
    # Create log folder
    _log_folder = Path(log_folder)
    _log_folder.mkdir(exist_ok=True)
    
    # NOW import requests
    import requests
    
    # Store original methods
    original_request = requests.request
    original_session_request = requests.Session.request
    
    # Patch requests.request
    def _patched_request(method, url, **kwargs):
        headers = kwargs.get('headers', {})
        json_body = kwargs.get('json')
        data_body = kwargs.get('data')
        body = json_body if json_body is not None else data_body
        
        response = original_request(method, url, **kwargs)
        
        try:
            response_body = response.text
        except:
            response_body = None
        
        _save_http_log(
            method=method.upper(),
            url=url,
            headers=headers,
            body=body,
            response_status=response.status_code,
            response_headers=dict(response.headers),
            response_body=response_body
        )
        
        return response
    
    # Patch Session.request
    def _patched_session_request(self, method, url, **kwargs):
        headers = kwargs.get('headers', {})
        json_body = kwargs.get('json')
        data_body = kwargs.get('data')
        body = json_body if json_body is not None else data_body
        
        response = original_session_request(self, method, url, **kwargs)
        
        try:
            response_body = response.text
        except:
            response_body = None
        
        _save_http_log(
            method=method.upper(),
            url=url,
            headers=headers,
            body=body,
            response_status=response.status_code,
            response_headers=dict(response.headers),
            response_body=response_body
        )
        
        return response
    
    requests.request = _patched_request
    requests.Session.request = _patched_session_request
    
    # Patch httpx if available
    try:
        import httpx
        original_httpx_send = httpx.Client.send
        
        def _patched_httpx_send(self, request, **kwargs):
            # Extract request info
            method = request.method
            url = str(request.url)
            headers = dict(request.headers)
            
            # Get body
            try:
                if hasattr(request, 'content'):
                    body = request.content
                    if body:
                        try:
                            body = json.loads(body)
                        except:
                            body = body.decode('utf-8') if isinstance(body, bytes) else str(body)
                else:
                    body = None
            except:
                body = None
            
            # Send request
            response = original_httpx_send(self, request, **kwargs)
            
            # Get response
            try:
                response_body = response.text
            except:
                response_body = None
            
            _save_http_log(
                method=method,
                url=url,
                headers=headers,
                body=body,
                response_status=response.status_code,
                response_headers=dict(response.headers),
                response_body=response_body
            )
            
            return response
        
        httpx.Client.send = _patched_httpx_send
    except:
        pass
    
    _patched = True
    print(f"[HTTP Logger] Enabled - logs in: {_log_folder.absolute()}")
    return _log_folder
