from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from app.core.utils.llm_factory import get_llm
from app.config.settings import get_settings

settings = get_settings()

class EmpatheticConversationChain:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        memory_window: int = 5
    ):
        """
        Initialize the Empathetic Conversation Chain.
        
        Args:
            provider: LLM provider to use
            model_name: Specific model name
            temperature: Controls randomness in output
            max_tokens: Maximum tokens to generate
            memory_window: Number of conversation turns to remember
        """
        self.llm = get_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an empathetic and supportive AI companion focused on providing emotional support and guidance.
            Your responses should be:
            1. Warm and understanding
            2. Non-judgmental
            3. Focused on active listening
            4. Encouraging self-reflection
            5. Respectful of boundaries
            
            Remember to:
            - Acknowledge the user's feelings
            - Ask open-ended questions when appropriate
            - Provide gentle guidance rather than direct advice
            - Maintain appropriate emotional distance
            - Recognize when to suggest professional help"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            The AI's response
        """
        try:
            response = self.chain.predict(input=user_input)
            return response
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your message right now. Please try again later. Error: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the current chat history.
        
        Returns:
            List of chat messages
        """
        return self.memory.chat_memory.messages 