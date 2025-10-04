"""
Datadog Logging & LLM Observability
- Send logs to Datadog US1 via HTTP API
- Track OpenAI LLM calls with LLM Observability (agentless)
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional


class DatadogLogger:
    """Simple Datadog HTTP API logger"""
    
    def __init__(self):
        self.api_key: Optional[str] = None
        self.enabled = False
        self.service = "nba-ai-commentary"
        self.env = "production"
        self.url = "https://http-intake.logs.datadoghq.com/api/v2/logs"
        self.session: Optional[requests.Session] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization - called on first log"""
        if self._initialized:
            return
        
        self._initialized = True
        self.api_key = os.getenv("DD_API_KEY", "").strip()
        self.enabled = bool(self.api_key)
        self.env = os.getenv("ENV", "production")
        
        if self.enabled:
            self.session = requests.Session()
            self.session.headers.update({
                "DD-API-KEY": self.api_key,
                "Content-Type": "application/json"
            })
            print("✓ Datadog logging enabled (US1)")
        else:
            print("ℹ️  Datadog logging disabled (DD_API_KEY not set)")
    
    def log(self, level: str, message: str, **context):
        """Send log to Datadog"""
        self._ensure_initialized()
        
        if not self.enabled:
            return
        
        try:
            log_entry = {
                "ddsource": "python",
                "ddtags": f"env:{self.env},service:{self.service}",
                "hostname": os.uname().nodename,
                "message": message,
                "level": level.upper(),
                "timestamp": int(datetime.now().timestamp() * 1000),
                **context  # Add all context fields
            }
            
            # Send async (don't wait for response)
            if self.session:
                self.session.post(self.url, data=json.dumps(log_entry), timeout=2)
        except Exception as e:
            # Print debug info if send fails
            print(f"[DEBUG] Datadog send failed: {e}")
    
    def info(self, message: str, **context):
        """Log INFO level"""
        self.log("INFO", message, **context)
        print(f"[INFO] {message}")
    
    def warning(self, message: str, **context):
        """Log WARNING level"""
        self.log("WARNING", message, **context)
        print(f"[WARN] {message}")
    
    def error(self, message: str, **context):
        """Log ERROR level"""
        self.log("ERROR", message, **context)
        print(f"[ERROR] {message}")


# Global logger instance (lazy initialization)
logger = DatadogLogger()


# ==================== LLM Observability ====================

def setup_llmobs():
    """
    Initialize Datadog LLM Observability (agentless mode)
    Automatically traces OpenAI API calls
    """
    dd_api_key = os.getenv("DD_API_KEY", "").strip()
    
    if not dd_api_key:
        print("ℹ️  LLM Observability disabled (DD_API_KEY not set)")
        return
    
    try:
        from ddtrace.llmobs import LLMObs
        
        # US1 region configuration
        # For US1, site should be "datadoghq.com" (default)
        # Other regions: us3.datadoghq.com, us5.datadoghq.com, datadoghq.eu, etc.
        dd_site = os.getenv("DD_SITE", "datadoghq.com")
        
        # Enable LLMObs in agentless mode
        LLMObs.enable(
            ml_app="nba-ai-commentary",
            api_key=dd_api_key,
            site=dd_site,
            agentless_enabled=True,
            integrations_enabled=True,  # Auto-instrument OpenAI
            env=os.getenv("ENV", "production")
        )
        
        print(f"✓ Datadog LLM Observability enabled (site: {dd_site})")
        
    except ImportError:
        print("⚠️  ddtrace not installed - run: pip install ddtrace>=2.9.0")
    except Exception as e:
        print(f"⚠️  LLM Observability init failed: {e}")
