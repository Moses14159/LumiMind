"""
Response Coach Chain for communication coaching.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory

from core.prompts.communication_prompts import RESPONSE_COACH_CHAT_TEMPLATE


class ResponseOption(BaseModel):
    """
    Model for a response option.
    """
    text: str = Field(description="The text of the response option.")
    explanation: str = Field(description="An explanation of the response option, including its tone, potential impact, and when it's most appropriate.")


class CommunicationAdvice(BaseModel):
    """
    Model for communication advice.
    """
    situation_analysis: str = Field(description="Analysis of the communication situation.")
    response_options: List[ResponseOption] = Field(description="List of response options.")
    metacognitive_questions: List[str] = Field(description="Questions to help the user reflect on their communication goals.")


class ResponseCoachChain:
    """
    Chain for coaching users on how to respond in various communication situations.
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
        Initialize the response coach chain.
        
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
        
        # Define structured output parser
        self.parser = PydanticOutputParser(pydantic_object=CommunicationAdvice)
    
    def _build_chain(self) -> Runnable:
        """
        Build the response coach chain.
        
        Returns:
            The response coach chain.
        """
        # Define the components
        _prompt = RESPONSE_COACH_CHAT_TEMPLATE
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
            "Given the following conversation and a follow up question about communication, "
            "retrieve relevant information about communication skills, etiquette, or strategies.\n"
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
    
    def parse_structured_output(self, text: str) -> CommunicationAdvice:
        """
        Parse the LLM output into a structured CommunicationAdvice object.
        
        This is a best-effort attempt to parse the output, which may not always succeed
        if the LLM doesn't follow the expected format. In that case, the parsing errors
        are caught and a default structured output is constructed.
        
        Args:
            text: The text to parse.
            
        Returns:
            A CommunicationAdvice object.
        """
        try:
            # Attempt to parse the output using the parser
            return self.parser.parse(text)
        except Exception as e:
            # If parsing fails, construct a basic structure from the text
            if self.verbose:
                print(f"Error parsing structured output: {e}")
            
            # Extract sections heuristically
            lines = text.split("\n")
            situation_analysis = ""
            response_options = []
            metacognitive_questions = []
            
            current_section = None
            current_option = None
            current_option_text = ""
            current_option_explanation = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if "situation" in line.lower() or "analysis" in line.lower():
                    current_section = "situation"
                    continue
                elif "option" in line.lower() or "response" in line.lower() and "#" in line:
                    # If we were already parsing an option, save it
                    if current_option is not None and current_option_text:
                        response_options.append(ResponseOption(
                            text=current_option_text,
                            explanation=current_option_explanation
                        ))
                    
                    current_section = "option"
                    current_option = line
                    current_option_text = ""
                    current_option_explanation = ""
                    continue
                elif "explanation" in line.lower() or "impact" in line.lower() or "tone" in line.lower():
                    current_section = "explanation"
                    continue
                elif "question" in line.lower() or "reflect" in line.lower() or "metacognition" in line.lower():
                    current_section = "questions"
                    continue
                
                # Add content to the appropriate section
                if current_section == "situation":
                    situation_analysis += line + " "
                elif current_section == "option" and not current_option_text:
                    current_option_text = line
                elif current_section == "explanation":
                    current_option_explanation += line + " "
                elif current_section == "questions":
                    if line.strip().endswith("?"):
                        metacognitive_questions.append(line)
            
            # Add the last option if there is one
            if current_option is not None and current_option_text:
                response_options.append(ResponseOption(
                    text=current_option_text,
                    explanation=current_option_explanation
                ))
            
            # Ensure we have at least some minimal structure
            if not situation_analysis:
                situation_analysis = "Analysis of the communication situation."
            
            if not response_options:
                response_options = [ResponseOption(
                    text="Consider asking for more details to better understand the situation.",
                    explanation="When unclear about the context, gathering more information is often the best first step."
                )]
            
            if not metacognitive_questions:
                metacognitive_questions = [
                    "What is your main goal in this communication?",
                    "How do you want the other person to feel after your response?"
                ]
            
            return CommunicationAdvice(
                situation_analysis=situation_analysis,
                response_options=response_options,
                metacognitive_questions=metacognitive_questions
            )
    
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