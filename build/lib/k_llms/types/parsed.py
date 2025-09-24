from typing import Any, Dict, Optional
from openai.types.chat import ParsedChatCompletion

from pydantic import Field


class KLLMsParsedChatCompletion(ParsedChatCompletion):
    """
    Enhanced ChatCompletion that includes likelihoods for consensus results.
    Inherits from OpenAI's BaseModel to maintain compatibility.
    """

    likelihoods: Optional[Dict[str, Any]] = Field(
        default=None, description="Object defining the uncertainties of the fields extracted when using consensus. Follows the same structure as the extraction object."
    )
