from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """
    You are a NASA mission intelligence expert.

    Use ONLY the provided NASA mission context to answer questions.

    Requirements:
    - Cite mission names and document sources when possible.
    - Do not make up information.
    - If the context does not contain the answer, clearly say so.
    - Focus on Apollo 11, Apollo 13, and Challenger mission data.
    - Keep responses factual, concise, and well-structured.
    """

    # Create messages list
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add retrieved context
    messages.append(
        {
            "role": "system",
            "content": f"NASA Reference Context:\n\n{context}"
        }
    )

    # Add conversation history
    for msg in conversation_history:
        messages.append(
            {
                "role": msg["role"],
                "content": msg["content"]
            }
        )

    # Add current user message
    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # Create OpenAI client
    client = OpenAI(
        base_url="https://openai.vocareum.com/v1",
        api_key=openai_key
    )

    # Send request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    # Return response
    return response.choices[0].message.content
    pass