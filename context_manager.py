from redis import Redis

# Initialize Redis for persistent memory
redis_client = Redis(host='localhost', port=6379, decode_responses=True)

def add_to_memory(session_id: str, message: str, role: str = "user"):
    """
    Store the conversation in Redis.
    Maintains both short-term and long-term memory.
    """
    # Save in Redis for persistence
    redis_client.lpush(session_id, f"{role}: {message}")
    
    # Trim the list to maintain short-term memory (e.g., last 20 messages)
    redis_client.ltrim(session_id, 0, 19)

def get_memory(session_id: str) -> list:
    """
    Retrieve the chat history from Redis.
    Returns a list of dictionaries with 'role' and 'content' keys.
    """
    chat_history = redis_client.lrange(session_id, 0, -1)
    
    # Format the chat history correctly for GPT-4
    formatted_history = []
    for msg in reversed(chat_history):
        role, content = msg.split(": ", 1)
        # Map role to 'user' or 'assistant' correctly
        if role not in ["user", "assistant"]:
            role = "user"  # Default to user if role is unknown
        formatted_history.append({"role": role, "content": content})
    
    return formatted_history

