import time
from datetime import datetime , timezone


# In-memory log for the demo (would be a DB in production)
_NOTIFICATION_LOG = []

def notify_user(severity :str , message: str , channels : list = None) -> dict:
    """
    ReAct tool: dispatch a notification.

    Args:
        severity: "info" | "warning" | "alert" | "emergency"
        message: Human-readable alert text.
        channels: List of delivery channels (default: ["console"]).

    Returns:
        {"delivered": bool, "severity": str, "channels": list, "timestamp": str}
    """

    if channels is None:
        channels = ["console"]
    
    record = {
        "severity": severity,
        "message": message,
        "channels": channels,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _NOTIFICATION_LOG.append(record)

    #Delivery simulation
    print(f"\n🔔 [NOTIFY/{severity.upper()}] {message}")
    print(f"   channels={channels} at {record['timestamp']}\n")
    time.sleep(0.05)  # simulate network latency

    return {
        "delivered": True,
        "severity": severity,
        "channels": channels,
        "timestamp": record["timestamp"],
    }

def get_notification_log() -> list:
    """Helper for UI/tests — not exposed as a tool."""
    return list(_NOTIFICATION_LOG)