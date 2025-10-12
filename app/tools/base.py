import abc
from typing import Any, Dict

class ToolBase(abc.ABC):
	"""
	Abstract async base class for LangChain AI agent tools.
	Inherit this class to implement custom tools.
	"""
	name: str
	description: str

	def __init__(self, name: str, description: str):
		self.name = name
		self.description = description

	@abc.abstractmethod
	async def run(self, input: Any, context: Dict[str, Any] = None) -> Any:
		"""
		Asynchronously execute the tool logic.
		Args:
			input: Input data for the tool.
			context: Optional context dictionary.
		Returns:
			Any: The result of the tool execution.
		"""
		pass
