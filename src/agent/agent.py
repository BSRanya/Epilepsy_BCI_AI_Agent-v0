import os 
import json 
import time
from groq import Groq

#tool importing 
from src.tools.detect_seizure import detect_seizure
from src.tools.notify_user import notify_user
from src.prompts.system_prompts import REACT_SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()


client =Groq(api_key= os.getenv("GROQ_API_KEY"))


#define the tool

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_seizure",
            "description": "Classify a 5-second, 6-channel EEG window. Returns probability, label, and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "eeg_window": {
                        "type": "array",
                        "description": "2D array [6][1280] — 6 channels, 1280 samples.",
                        "items": {"type": "array", "items": {"type": "number"}},
                    }
                },
                "required": ["eeg_window"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_user",
            "description": "Send an alert to the patient or caregiver. Use only when detection indicates seizure or preictal state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "alert", "emergency"]},
                    "message": {"type": "string", "description": "Human-readable alert text."},
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional delivery channels, e.g. ['push', 'sms'].",
                    },
                },
                "required": ["severity", "message"],
                "additionalProperties": False,
            },
        },
    },
]
















"""
TOOLS=[
    {
        "type": "function",
        "function": {
            "name": "detect_seizure",
            "description": (
                "Detect seizure activity from a 5-second, 6-channel EEG window. "
                "Returns seizure probability, label, and confidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eeg_window": {
                        "type": "array",
                        "description": "2D array of shape (6, 1280): 6 channels, 1280 samples (5s at 256Hz).",
                        "items": {"type": "array", "items": {"type": "number"}},
                    }
                },
                "required": ["eeg_window"],
                "additionalProperties": False  # SECURITY: Prevents extra, unexpected parameters
            },
        },
    }
]

"""

#A safe allwlisted function mapping , to ensure the model can only call functions we trust 

TOOL_FUNCTIONS = {
    "detect_seisure" :detect_seizure,
    "notify_user" : notify_user ,
}

#to prevent infinite loop 
MAX_ITERATIONS=5







"""
"""
SYSTEM_PROMPT ="""You are a clinical seizure monitoring agent, You are skilled helpful and life saver you only receive EEG detection results and must decide wisely and carefully whether to alert the user .
if seizure_probability >=0.75 , generate a consise , urgent notification.
if 0.45<= seizure_probability < 0.75 , generate a serious warning that a seizure may be imminent.
Otherwise , return a brief status update. Always include the probability and confidence in your resopnse """
"""
"""



#Input validation for the tool arguments 

def validate_tool_arguments(tool_name: str , arguments : dict )->bool:
    """Validate th arguments passed by the model before the execution .
    returns True if valid , False otherwise ."""
    if tool_name =="detect_seizure" or "notify_user" :
        eeg_window = arguments.get("eeg_window")
        if not isinstance(eeg_window , list) or len(eeg_window) !=6:
            return False
        for channel in eeg_window:
            if not isinstance(channel , list) or len(channel)!=1280:
                return False
            if not all(isinstance(x , (int,float)) for x in channel):
                return False
    return True


def run_react_step(eeg_window , trace: list = None)->dict:
    """
    Run one full ReAct episode for a single EEG window
    Args:
        eeg_window: np.ndarray (6, 1280).
        trace: Optional list to append the reasoning trace to (for UI display).

    Returns:
        {
            "final_answer": str,
            "trace": list of {thought, action, observation},
            "notified": bool,
            "iterations": int,
            "elapsed_ms": float,
        }
    """
    
    if trace is None :
        trace = []

    t0=time.time()
    window_list = eeg_window.tolist() if hasattr( eeg_window , "tolist") else eeg_window

    messages = [
        {"role" : "system" , "content" : REACT_SYSTEM_PROMPT} ,
        {
            "role" :"user",
            "content" : ("New EEG window receive .Analyze it and take appropriate action. \n"
                         f"window shape :6 channels *1280 samples. \n"
                         f"Data: {json.dumps(window_list)} "
            
            
            ),
        },
    ]

    notified = False
    final_answer = None

    for iteration in range (MAX_ITERATIONS):
        response = client.chat.completions.create(
            model ="qwen/qwen3.8-27b",
            messages=messages,
            tools=TOOLS , 
            tool_choice="auto" , 
            #parallel_tool_calls= True ,
            temperature=0.1 , 
        )
        msg = response.choices[0].message

        #No tool call ---> final answer 
        if not msg.tool_calls:
            final_answer = msg.content or"(no content)"
            trace.append({"thought" : msg.content or "" , "action": None , "observation" : None})
            break

        #tool calls then Act 
        # Record the model's reasoning (if any) as "Thought"
    
        thought = msg.content or "(reasoning implied by tool call )"
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls":[tc.model_dump() for tc in msg.tool_calls],

        })

        for tc in msg.tool_calls :
            name =tc.function.name 
            try:
                args = json.lads(tc.function.arguments)
            except json.JSONDecodeError :
                args = {}

                # Security: allowlist check
            if name not in TOOL_FUNCTIONS:
                obs = {"error": f"unauthorized tool: {name}"}
            else:
                try:
                    obs = TOOL_FUNCTIONS[name](**args)
                    if name == "notify_user":
                        notified = True
                except Exception as e:
                    obs = {"error": f"{type(e).__name__}: {e}"}

            # Append observation back to the conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(obs),
            })

            trace.append({
                "thought": thought,
                "action": {"tool": name, "args": _summarize_args(name, args)},
                "observation": obs,
            })

    # Fallback if loop exhausted
    if final_answer is None:
        final_answer = "[ReAct loop exhausted without final answer]"

    return {
        "final_answer": final_answer,
        "trace": trace,
        "notified": notified,
        "iterations": iteration + 1,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
    }


def _summarize_args(tool_name: str, args: dict) -> dict:
    """Don't dump a 6x1280 array into the trace — summarize it."""
    if tool_name == "detect_seizure":
        w = args.get("eeg_window", [])
        return {"eeg_window": f"<{len(w)}×{len(w[0]) if w else 0} array>"}
    return args











