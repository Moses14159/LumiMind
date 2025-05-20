"""
Empathetic Conversation Chain for mental health support.
"""
from typing import Dict, Any, List, Optional, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory

from core.prompts.mental_health_prompts import (
    EMPATHETIC_CONVERSATION_CHAT_TEMPLATE
)


class EmpatheticConversationChain:
    """
    Chain for empathetic conversation focused on mental health support.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Optional[BaseRetriever] = None,
        memory: Optional[BaseMemory] = None,
        memory_key: str = "chat_history",
        verbose: bool = False
    ):
        """
        Initialize the empathetic conversation chain.
        
        Args:
            llm: The language model to use.
            retriever: The retriever to use for RAG.
            memory: The memory to use for storing conversation history.
            memory_key: The key to use for the memory in the prompt.
            verbose: Whether to print verbose output.
        """
        self.llm = llm
        self.retriever = retriever
        self.memory = memory or ConversationBufferWindowMemory(
            memory_key=memory_key,
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
        self.memory_key = memory_key
        self.verbose = verbose
        
        # Build the chain
        self.chain = self._build_chain()
    
    def _build_chain(self) -> Runnable:
        """
        Build the conversation chain.
        
        Returns:
            The conversation chain.
        """
        # Define the components
        _prompt = EMPATHETIC_CONVERSATION_CHAT_TEMPLATE
        _retrieval_chain = self._build_retrieval_chain() if self.retriever else None
        
        # Build the main chain
        def _get_memory(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Get memory from the input dictionary."""
            memory_dict = {self.memory_key: self.memory.load_memory_variables(input_dict)[self.memory_key]}
            if "input" in input_dict:
                memory_dict["input"] = input_dict["input"]
            if self.verbose:
                print(f"Memory: {memory_dict}")
            return memory_dict
        
        def _combine_context(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Combine the retrieved context with the input dictionary."""
            if _retrieval_chain is not None and "input" in input_dict:
                if self.verbose:
                    print(f"Retrieving context for: {input_dict['input']}")
                
                context = _retrieval_chain.invoke({"input": input_dict["input"]})
                input_dict["context"] = context
                
                if self.verbose:
                    print(f"Retrieved context: {context}")
            else:
                input_dict["context"] = ""
            
            return input_dict
        
        # Define the chain
        chain = (
            RunnablePassthrough()
            | RunnableLambda(_get_memory)
            | RunnableLambda(_combine_context)
            | _prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _build_retrieval_chain(self) -> Runnable:
        """
        Build the retrieval chain.
        
        Returns:
            The retrieval chain.
        """
        # Basic retrieval prompt
        retrieval_prompt = PromptTemplate.from_template(
            "Given the following conversation and a follow up question, "
            "retrieve relevant information to help answer the question.\n"
            "Chat History: {chat_history}\n"
            "Follow Up Input: {input}\n"
        )
        
        # Build the retrieval chain
        retrieval_chain = (
            retrieval_prompt
            | self.llm
            | StrOutputParser()
            | self.retriever
            | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))
        )
        
        return retrieval_chain
    
    def invoke(self, input_dict: Dict[str, Any]) -> str:
        """
        Invoke the chain.
        
        Args:
            input_dict: The input dictionary containing the user message.
            
        Returns:
            The response from the LLM.
        """
        if "input" not in input_dict:
            raise ValueError("Input dictionary must contain 'input' key.")
        
        # Get the user input
        user_input = input_dict["input"]
        
        # Invoke the chain
        response = self.chain.invoke(input_dict)
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return response
    
    async def ainvoke(self, input_dict: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the chain.
        
        Args:
            input_dict: The input dictionary containing the user message.
            
        Returns:
            The response from the LLM.
        """
        if "input" not in input_dict:
            raise ValueError("Input dictionary must contain 'input' key.")
        
        # Get the user input
        user_input = input_dict["input"]
        
        # Invoke the chain
        response = await self.chain.ainvoke(input_dict)
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return response
    
    def get_memory(self) -> List[Union[HumanMessage, AIMessage]]:
        """
        Get the conversation memory.
        
        Returns:
            The conversation memory as a list of messages.
        """
        return self.memory.load_memory_variables({})[self.memory_key]
    
    def clear_memory(self) -> None:
        """
        Clear the conversation memory.
        """
        self.memory.clear() 