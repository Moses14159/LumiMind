"""
CBT Exercise Chain for guiding users through CBT exercises.
"""
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

from core.prompts.mental_health_prompts import CBT_EXERCISE_CHAT_TEMPLATE


class CBTStage(str, Enum):
    """
    Enum for the stages of the CBT exercise.
    """
    INTRODUCTION = "introduction"
    SITUATION = "situation"
    THOUGHTS = "thoughts"
    FEELINGS = "feelings"
    DISTORTIONS = "distortions"
    EVIDENCE = "evidence"
    ALTERNATIVE_THOUGHTS = "alternative_thoughts"
    REFLECTION = "reflection"
    SUMMARY = "summary"


class ThoughtRecord(BaseModel):
    """
    Model for a CBT thought record.
    """
    situation: str = Field(default="", description="A description of the situation that triggered the negative thoughts")
    thoughts: List[str] = Field(default_factory=list, description="The automatic thoughts that came to mind")
    feelings: Dict[str, int] = Field(default_factory=dict, description="The feelings experienced and their intensity (0-100%)")
    distortions: List[str] = Field(default_factory=list, description="The cognitive distortions identified in the thoughts")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence that supports the negative thoughts")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Evidence that contradicts the negative thoughts")
    alternative_thoughts: List[str] = Field(default_factory=list, description="More balanced alternative thoughts")
    initial_distress: int = Field(default=0, description="Initial distress level (0-100%)")
    final_distress: int = Field(default=0, description="Final distress level after the exercise (0-100%)")


class CBTExerciseChain:
    """
    Chain for guiding users through CBT exercises.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[BaseMemory] = None,
        memory_key: str = "chat_history",
        verbose: bool = False
    ):
        """
        Initialize the CBT exercise chain.
        
        Args:
            llm: The language model to use.
            memory: The memory to use for storing conversation history.
            memory_key: The key to use for the memory in the prompt.
            verbose: Whether to print verbose output.
        """
        self.llm = llm
        self.memory = memory or ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True
        )
        self.memory_key = memory_key
        self.verbose = verbose
        self.current_stage = CBTStage.INTRODUCTION
        self.thought_record = ThoughtRecord()
        
        # Build the chain
        self.chain = self._build_chain()
    
    def _build_chain(self) -> Runnable:
        """
        Build the CBT exercise chain.
        
        Returns:
            The CBT exercise chain.
        """
        # Define the prompt
        _prompt = CBT_EXERCISE_CHAT_TEMPLATE
        
        # Build the main chain
        def _get_memory(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Get memory from the input dictionary,并保留 input 字段"""
            memory_dict = {self.memory_key: self.memory.load_memory_variables(input_dict)[self.memory_key]}
            # 保留 input 字段
            if "input" in input_dict:
                memory_dict["input"] = input_dict["input"]
            if self.verbose:
                print(f"Memory: {memory_dict}")
            return memory_dict
        
        def _add_stage_info(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Add the current stage and context to the input dictionary."""
            input_dict["stage"] = self.current_stage.value
            input_dict["context"] = self._get_stage_context()
            return input_dict
        
        # Define the chain
        chain = (
            RunnablePassthrough()
            | RunnableLambda(_get_memory)
            | RunnableLambda(_add_stage_info)
            | _prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _get_stage_context(self) -> str:
        """
        Get the context for the current stage.
        
        Returns:
            The context string.
        """
        record = self.thought_record
        
        if self.current_stage == CBTStage.INTRODUCTION:
            return "Starting the CBT exercise. Introduce the exercise to the user."
        
        elif self.current_stage == CBTStage.SITUATION:
            return "Guide the user to describe a specific situation that triggered negative emotions."
        
        elif self.current_stage == CBTStage.THOUGHTS:
            context = f"Situation: {record.situation}\n\n"
            context += "Guide the user to identify automatic thoughts that came to mind in this situation."
            return context
        
        elif self.current_stage == CBTStage.FEELINGS:
            context = f"Situation: {record.situation}\n\n"
            context += f"Automatic thoughts: {', '.join(record.thoughts)}\n\n"
            context += "Guide the user to identify and rate the intensity of their feelings (0-100%)."
            return context
        
        elif self.current_stage == CBTStage.DISTORTIONS:
            context = f"Situation: {record.situation}\n\n"
            context += f"Automatic thoughts: {', '.join(record.thoughts)}\n\n"
            context += f"Feelings: {', '.join([f'{feeling} ({intensity}%)' for feeling, intensity in record.feelings.items()])}\n\n"
            context += "Help the user identify cognitive distortions in their automatic thoughts."
            return context
        
        elif self.current_stage == CBTStage.EVIDENCE:
            context = f"Situation: {record.situation}\n\n"
            context += f"Automatic thoughts: {', '.join(record.thoughts)}\n\n"
            context += f"Feelings: {', '.join([f'{feeling} ({intensity}%)' for feeling, intensity in record.feelings.items()])}\n\n"
            context += f"Cognitive distortions: {', '.join(record.distortions)}\n\n"
            context += "Help the user evaluate evidence that supports and contradicts their automatic thoughts."
            return context
        
        elif self.current_stage == CBTStage.ALTERNATIVE_THOUGHTS:
            context = f"Situation: {record.situation}\n\n"
            context += f"Automatic thoughts: {', '.join(record.thoughts)}\n\n"
            context += f"Supporting evidence: {', '.join(record.supporting_evidence)}\n\n"
            context += f"Contradicting evidence: {', '.join(record.contradicting_evidence)}\n\n"
            context += "Help the user develop more balanced alternative thoughts."
            return context
        
        elif self.current_stage == CBTStage.REFLECTION:
            context = f"Situation: {record.situation}\n\n"
            context += f"Original thoughts: {', '.join(record.thoughts)}\n\n"
            context += f"Alternative thoughts: {', '.join(record.alternative_thoughts)}\n\n"
            context += "Ask the user to reflect on how they feel now and rate their distress level again (0-100%)."
            return context
        
        elif self.current_stage == CBTStage.SUMMARY:
            context = "Summarize the exercise, highlighting the progress made and encouraging continued practice."
            return context
        
        return ""
    
    def _update_thought_record(self, stage: CBTStage, user_input: str, ai_response: str) -> None:
        """
        Update the thought record based on the user input and AI response.
        
        Args:
            stage: The current stage.
            user_input: The user's input.
            ai_response: The AI's response.
        """
        # Simple heuristic-based extraction - in a production system, you might use more sophisticated extraction
        if stage == CBTStage.SITUATION:
            self.thought_record.situation = user_input.strip()
        
        elif stage == CBTStage.THOUGHTS:
            # Extract thoughts from user input, splitting by newlines or commas
            thoughts = [t.strip() for t in user_input.replace("\n", ",").split(",") if t.strip()]
            self.thought_record.thoughts = thoughts
        
        elif stage == CBTStage.FEELINGS:
            # Try to extract feelings and their intensity
            feelings = {}
            lines = user_input.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for patterns like "Anxiety: 80%" or "Anxiety (80%)"
                if ":" in line:
                    parts = line.split(":")
                    feeling = parts[0].strip()
                    intensity_str = parts[1].strip().replace("%", "")
                    try:
                        intensity = int(intensity_str)
                        feelings[feeling] = intensity
                    except ValueError:
                        pass
                elif "(" in line and ")" in line:
                    parts = line.split("(")
                    feeling = parts[0].strip()
                    intensity_str = parts[1].replace(")", "").replace("%", "").strip()
                    try:
                        intensity = int(intensity_str)
                        feelings[feeling] = intensity
                    except ValueError:
                        pass
            
            if feelings:
                self.thought_record.feelings = feelings
                # Use the highest intensity as the initial distress level
                self.thought_record.initial_distress = max(feelings.values())
        
        elif stage == CBTStage.DISTORTIONS:
            # Extract distortions from user input
            distortions = [d.strip() for d in user_input.replace("\n", ",").split(",") if d.strip()]
            self.thought_record.distortions = distortions
        
        elif stage == CBTStage.EVIDENCE:
            # Try to extract supporting and contradicting evidence
            supporting = []
            contradicting = []
            
            # Look for sections labeled as supporting or contradicting
            if "supporting" in user_input.lower() or "for" in user_input.lower():
                parts = user_input.lower().split("supporting" if "supporting" in user_input.lower() else "for", 1)
                if len(parts) > 1:
                    supporting_text = parts[1]
                    if "contradicting" in supporting_text.lower() or "against" in supporting_text.lower():
                        supporting_text = supporting_text.split("contradicting" if "contradicting" in supporting_text.lower() else "against", 1)[0]
                    
                    supporting = [s.strip() for s in supporting_text.replace("\n", ",").split(",") if s.strip()]
            
            if "contradicting" in user_input.lower() or "against" in user_input.lower():
                parts = user_input.lower().split("contradicting" if "contradicting" in user_input.lower() else "against", 1)
                if len(parts) > 1:
                    contradicting_text = parts[1]
                    contradicting = [c.strip() for c in contradicting_text.replace("\n", ",").split(",") if c.strip()]
            
            if supporting:
                self.thought_record.supporting_evidence = supporting
            if contradicting:
                self.thought_record.contradicting_evidence = contradicting
        
        elif stage == CBTStage.ALTERNATIVE_THOUGHTS:
            # Extract alternative thoughts from user input
            alt_thoughts = [t.strip() for t in user_input.replace("\n", ",").split(",") if t.strip()]
            self.thought_record.alternative_thoughts = alt_thoughts
        
        elif stage == CBTStage.REFLECTION:
            # Try to extract final distress level
            try:
                # Look for a number followed by % in the user input
                import re
                match = re.search(r'(\d+)[%]', user_input)
                if match:
                    self.thought_record.final_distress = int(match.group(1))
            except ValueError:
                pass
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the chain.
        
        Args:
            input_dict: The input dictionary containing the user message.
            
        Returns:
            A dictionary containing:
                - `response`: The response from the LLM.
                - `stage`: The current stage.
                - `is_complete`: Whether the exercise is complete.
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
        
        # Update the thought record
        current_stage = self.current_stage
        self._update_thought_record(current_stage, user_input, response)
        
        # Advance to the next stage if this isn't the first interaction
        if not (current_stage == CBTStage.INTRODUCTION and "start" in user_input.lower()):
            self._advance_stage()
        
        # Check if the exercise is complete
        is_complete = self.current_stage == CBTStage.SUMMARY
        
        return {
            "response": response,
            "stage": self.current_stage.value,
            "is_complete": is_complete
        }
    
    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously invoke the chain.
        
        Args:
            input_dict: The input dictionary containing the user message.
            
        Returns:
            A dictionary containing:
                - `response`: The response from the LLM.
                - `stage`: The current stage.
                - `is_complete`: Whether the exercise is complete.
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
        
        # Update the thought record
        current_stage = self.current_stage
        self._update_thought_record(current_stage, user_input, response)
        
        # Advance to the next stage if this isn't the first interaction
        if not (current_stage == CBTStage.INTRODUCTION and "start" in user_input.lower()):
            self._advance_stage()
        
        # Check if the exercise is complete
        is_complete = self.current_stage == CBTStage.SUMMARY
        
        return {
            "response": response,
            "stage": self.current_stage.value,
            "is_complete": is_complete
        }
    
    def _advance_stage(self) -> None:
        """
        Advance to the next stage of the CBT exercise.
        """
        if self.current_stage == CBTStage.INTRODUCTION:
            self.current_stage = CBTStage.SITUATION
        elif self.current_stage == CBTStage.SITUATION:
            self.current_stage = CBTStage.THOUGHTS
        elif self.current_stage == CBTStage.THOUGHTS:
            self.current_stage = CBTStage.FEELINGS
        elif self.current_stage == CBTStage.FEELINGS:
            self.current_stage = CBTStage.DISTORTIONS
        elif self.current_stage == CBTStage.DISTORTIONS:
            self.current_stage = CBTStage.EVIDENCE
        elif self.current_stage == CBTStage.EVIDENCE:
            self.current_stage = CBTStage.ALTERNATIVE_THOUGHTS
        elif self.current_stage == CBTStage.ALTERNATIVE_THOUGHTS:
            self.current_stage = CBTStage.REFLECTION
        elif self.current_stage == CBTStage.REFLECTION:
            self.current_stage = CBTStage.SUMMARY
    
    def reset(self) -> None:
        """
        Reset the chain to start a new exercise.
        """
        self.current_stage = CBTStage.INTRODUCTION
        self.thought_record = ThoughtRecord()
        self.memory.clear()
    
    def get_thought_record(self) -> ThoughtRecord:
        """
        Get the current thought record.
        
        Returns:
            The current thought record.
        """
        return self.thought_record
    
    def get_memory(self) -> List[Union[HumanMessage, AIMessage]]:
        """
        Get the conversation memory.
        
        Returns:
            The conversation memory as a list of messages.
        """
        return self.memory.load_memory_variables({})[self.memory_key] 