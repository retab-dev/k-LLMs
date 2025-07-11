from typing import Any, Dict, List, Optional
from openai.types.responses import Response

from pydantic import Field


class KLLMsResponse(Response):
    """
    Enhanced Response that includes likelihoods for consensus results.
    Inherits from OpenAI's BaseModel to maintain compatibility.
    """

    likelihoods: Optional[Dict[str, Any]] = Field(
        default=None, description="Object defining the uncertainties of the fields extracted when using consensus. Follows the same structure as the extraction object."
    )
    
    individual_responses: Optional[List[Response]] = Field(
        default=None, description="List of individual responses used to create the consensus. Available when n_consensus > 1."
    )
