"""
Role Play Chain for practicing difficult conversations.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

from core.prompts.communication_prompts import ROLE_PLAY_CHAT_TEMPLATE


class RolePlayScenario(BaseModel):
    """
    Model for a role play scenario.
    """
    name: str = Field(description="The name of the scenario.")
    character: str = Field(description="The character that the AI will play.")
    character_description: str = Field(description="A description of the character.")
    scenario_description: str = Field(description="A description of the scenario.")


class RolePlayChain:
    """
    Chain for role playing conversations to practice communication skills.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        scenario: Optional[RolePlayScenario] = None,
        memory: Optional[BaseMemory] = None,
        memory_key: str = "chat_history",
        verbose: bool = False
    ):
        """
        Initialize the role play chain.
        
        Args:
            llm: The language model to use.
            scenario: The role play scenario to use.
            memory: The memory to use for storing conversation history.
            memory_key: The key to use for the memory in the prompt.
            verbose: Whether to print verbose output.
        """
        self.llm = llm
        self.scenario = scenario
        self.memory = memory or ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True
        )
        self.memory_key = memory_key
        self.verbose = verbose
        self.in_feedback_mode = False
        
        # Build the chain if a scenario is provided
        if self.scenario:
            self.chain = self._build_chain()
    
    def _build_chain(self) -> Runnable:
        """
        Build the role play chain.
        
        Returns:
            The role play chain.
        """
        if not self.scenario:
            raise ValueError("A scenario must be provided to build the chain.")
        
        # Define the prompt
        _prompt = ROLE_PLAY_CHAT_TEMPLATE
        
        # Build the main chain
        def _get_memory(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Get memory from the input dictionary."""
            memory_dict = {self.memory_key: self.memory.load_memory_variables(input_dict)[self.memory_key]}
            if self.verbose:
                print(f"Memory: {memory_dict}")
            return memory_dict
        
        def _add_scenario_info(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Add the scenario information to the input dictionary."""
            input_dict["character"] = self.scenario.character
            input_dict["scenario"] = self.scenario.name
            input_dict["character_description"] = self.scenario.character_description
            input_dict["scenario_description"] = self.scenario.scenario_description
            return input_dict
        
        # Define the chain
        chain = (
            RunnablePassthrough()
            | RunnableLambda(_get_memory)
            | RunnableLambda(_add_scenario_info)
            | _prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def set_scenario(self, scenario: RolePlayScenario) -> None:
        """
        Set the role play scenario.
        
        Args:
            scenario: The role play scenario to use.
        """
        self.scenario = scenario
        self.chain = self._build_chain()
        self.clear_memory()
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the chain.
        
        Args:
            input_dict: The input dictionary containing the user message.
            
        Returns:
            A dictionary containing:
                - `response`: The response from the LLM.
                - `in_character`: Whether the response is in character.
        """
        if not self.scenario:
            raise ValueError("A scenario must be set before invoking the chain.")
        
        if "input" not in input_dict:
            raise ValueError("Input dictionary must contain 'input' key.")
        
        # Get the user input
        user_input = input_dict["input"]
        
        # Check if the user is asking for feedback or to end the role play
        self.in_feedback_mode = "feedback" in user_input.lower() or "how did i do" in user_input.lower()
        ending_role_play = "end role play" in user_input.lower() or "exit role play" in user_input.lower()
        
        # If in feedback mode or ending, generate appropriate response
        if self.in_feedback_mode:
            response = self._generate_feedback()
            in_character = False
        elif ending_role_play:
            response = self._end_role_play()
            in_character = False
        else:
            # Normal role play response
            response = self.chain.invoke(input_dict)
            in_character = True
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return {
            "response": response,
            "in_character": in_character
        }
    
    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously invoke the chain.
        
        Args:
            input_dict: The input dictionary containing the user message.
            
        Returns:
            A dictionary containing:
                - `response`: The response from the LLM.
                - `in_character`: Whether the response is in character.
        """
        if not self.scenario:
            raise ValueError("A scenario must be set before invoking the chain.")
        
        if "input" not in input_dict:
            raise ValueError("Input dictionary must contain 'input' key.")
        
        # Get the user input
        user_input = input_dict["input"]
        
        # Check if the user is asking for feedback or to end the role play
        self.in_feedback_mode = "feedback" in user_input.lower() or "how did i do" in user_input.lower()
        ending_role_play = "end role play" in user_input.lower() or "exit role play" in user_input.lower()
        
        # If in feedback mode or ending, generate appropriate response
        if self.in_feedback_mode:
            response = self._generate_feedback()
            in_character = False
        elif ending_role_play:
            response = self._end_role_play()
            in_character = False
        else:
            # Normal role play response
            response = await self.chain.ainvoke(input_dict)
            in_character = True
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return {
            "response": response,
            "in_character": in_character
        }
    
    def _generate_feedback(self) -> str:
        """
        Generate feedback on the user's communication in the role play.
        
        Returns:
            Feedback on the user's communication.
        """
        # Get the conversation history
        conversation = self.get_memory()
        
        # Create a feedback prompt
        feedback_prompt = f"""
You have been role-playing as {self.scenario.character} in a {self.scenario.name} scenario.

Here is the conversation so far:

{self._format_conversation(conversation)}

Now, stepping out of character, provide constructive feedback on the user's communication skills during this role play. Consider:
1. Effectiveness of their approach
2. Clarity of communication
3. Emotional intelligence and empathy
4. Areas of strength
5. Opportunities for improvement

Provide specific examples from the conversation to illustrate your points.
"""
        
        # Generate feedback
        feedback = self.llm.invoke(feedback_prompt)
        
        # Return the feedback as a string
        if hasattr(feedback, "content"):
            return feedback.content
        return str(feedback)
    
    def _end_role_play(self) -> str:
        """
        End the role play and provide a summary.
        
        Returns:
            A summary of the role play.
        """
        # Get the conversation history
        conversation = self.get_memory()
        
        # Create an ending prompt
        ending_prompt = f"""
You have been role-playing as {self.scenario.character} in a {self.scenario.name} scenario.

Here is the conversation that took place:

{self._format_conversation(conversation)}

Now that the role play is ending, provide:
1. A brief summary of how the conversation went
2. Key moments or turning points
3. Overall assessment of how effectively the situation was handled
4. 1-2 specific tips for similar situations in the future

Keep your response concise and focused on actionable insights.
"""
        
        # Generate summary
        summary = self.llm.invoke(ending_prompt)
        
        # Return the summary as a string
        if hasattr(summary, "content"):
            return summary.content
        return str(summary)
    
    def _format_conversation(self, conversation: List[Union[HumanMessage, AIMessage]]) -> str:
        """
        Format the conversation history for feedback or ending.
        
        Args:
            conversation: The conversation history.
            
        Returns:
            A formatted string representation of the conversation.
        """
        formatted = ""
        for message in conversation:
            if isinstance(message, HumanMessage):
                formatted += f"User: {message.content}\n\n"
            elif isinstance(message, AIMessage):
                formatted += f"{self.scenario.character}: {message.content}\n\n"
        
        return formatted
    
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


# Define some standard role play scenarios
STANDARD_SCENARIOS = {
    "salary_negotiation": RolePlayScenario(
        name="Salary Negotiation",
        character="Manager",
        character_description="""You are a mid-level manager at a technology company. You value the employee but have budget constraints. You're willing to negotiate but need to stay within company guidelines. You're busy and want to keep the conversation focused and professional.""",
        scenario_description="""The user is an employee who has been with the company for two years and has consistently performed well. They're asking for a 15% salary increase, which is above the company's standard 5% annual adjustment. The current economic climate is challenging, and the company is trying to control costs."""
    ),
    
    "difficult_feedback": RolePlayScenario(
        name="Giving Difficult Feedback",
        character="Colleague",
        character_description="""You are a colleague who feels unfairly criticized. You believe you've been pulling your weight on the project and that there are legitimate reasons for some of the delays. You're somewhat defensive but open to feedback if delivered respectfully.""",
        scenario_description="""The user needs to give feedback to you (their colleague) about missing deadlines on a joint project. The situation has created problems with the client, and the issue needs to be addressed. The user wants to preserve the working relationship while ensuring the project gets back on track."""
    ),
    
    "boundary_setting": RolePlayScenario(
        name="Setting Boundaries",
        character="Friend",
        character_description="""You are a close friend who has grown accustomed to asking for favors and emotional support at any time. You don't fully realize how your demands have been affecting the user. You tend to be somewhat sensitive to rejection but can understand boundaries if they're explained clearly.""",
        scenario_description="""The user needs to set boundaries with you (their friend) who frequently calls late at night with personal problems and asks for time-consuming favors. The user values the friendship but is feeling burnt out and needs to establish healthier boundaries while being compassionate."""
    ),
    
    "conflict_resolution": RolePlayScenario(
        name="Conflict Resolution",
        character="Team Member",
        character_description="""You are a team member with strong opinions about the project direction. You believe your approach is best for the company and have technical expertise to back it up. You can be stubborn but will respond to well-reasoned arguments. You value being respected for your expertise.""",
        scenario_description="""The user is working on a team project where there's significant disagreement about the approach. They need to address the conflict with you (another team member) who strongly disagrees with their proposed direction. The deadline is approaching, and the team needs to reach a resolution."""
    ),
    
    "customer_complaint": RolePlayScenario(
        name="Handling a Customer Complaint",
        character="Upset Customer",
        character_description="""You are a customer who paid premium price for a service/product that didn't meet expectations. You're frustrated after multiple attempts to resolve the issue through customer service channels. You're not abusive but are clearly upset and want a solution, not excuses.""",
        scenario_description="""The user is in a customer service role dealing with you, an upset customer who has experienced multiple issues with their purchase and has had a poor experience with previous customer service interactions. The user needs to de-escalate the situation and find an appropriate resolution."""
    )
} 