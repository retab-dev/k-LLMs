from typing import Any, Dict, Optional
from openai.types.chat import ChatCompletion

from pydantic import Field


class KLLMsChatCompletion(ChatCompletion):
    """
    Enhanced ChatCompletion that includes likelihoods for consensus results.
    Inherits from OpenAI's BaseModel to maintain compatibility.
    """

    likelihoods: Optional[Dict[str, Any]] = Field(
        default=None, description="Object defining the uncertainties of the fields extracted when using consensus. Follows the same structure as the extraction object."
    )
