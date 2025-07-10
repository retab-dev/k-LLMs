from io import IOBase
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import PIL.Image
from openai.types.chat.chat_completion_reasoning_effort import ChatCompletionReasoningEffort
from pydantic import HttpUrl

from .._resource import AsyncAPIResource, SyncAPIResource
from ..utils.mime import prepare_mime_document
from ..types.evals import (
    CreateIterationRequest,
    DistancesResult,
    DocumentItem,
    Evaluation,
    EvaluationDocument,
    Iteration,
    UpdateEvaluationDocumentRequest,
    UpdateEvaluationRequest,
)
from ..types.inference_settings import InferenceSettings
from ..types.mime import MIMEData
from ..types.modalities import Modality
from ..types.browser_canvas import BrowserCanvas
from ..types.standards import PreparedRequest


class DeleteResponse(TypedDict):
    """Response from a delete operation"""

    success: bool
    id: str


class ExportResponse(TypedDict):
    """Response from an export operation"""

    success: bool
    path: str


class EvalsMixin:
    def prepare_create(
        self,
        name: str,
        json_schema: Dict[str, Any],
        project_id: str | None = None,
        documents: List[EvaluationDocument] = [],
        iterations: List[Iteration] = [],
        default_inference_settings: Optional[InferenceSettings] = None,
    ) -> PreparedRequest:
        eval_data = Evaluation(
            name=name,
            json_schema=json_schema,
            project_id=project_id if project_id else "default_spreadsheets",
            documents=documents,
            iterations=iterations,
            default_inference_settings=default_inference_settings,
        )
        return PreparedRequest(method="POST", url="/v1/evals", data=eval_data.model_dump(exclude_none=True, mode="json"))

    def prepare_get(self, evaluation_id: str) -> PreparedRequest:
        return PreparedRequest(method="GET", url=f"/v1/evals/{evaluation_id}")

    def prepare_update(
        self,
        evaluation_id: str,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        documents: Optional[List[EvaluationDocument]] = None,
        iterations: Optional[List[Iteration]] = None,
        default_inference_settings: Optional[InferenceSettings] = None,
    ) -> PreparedRequest:
        """
        Prepare a request to update an evaluation with partial updates.

        Only the provided fields will be updated. Fields set to None will be excluded from the update.
        """
        # Build a dictionary with only the provided fields

        update_request = UpdateEvaluationRequest(
            name=name,
            project_id=project_id,
            json_schema=json_schema,
            documents=documents,
            iterations=iterations,
            default_inference_settings=default_inference_settings,
        )

        return PreparedRequest(method="PATCH", url=f"/v1/evals/{evaluation_id}", data=update_request.model_dump(exclude_none=True, mode="json"))

    def prepare_list(self, project_id: Optional[str] = None) -> PreparedRequest:
        params = {}
        if project_id:
            params["project_id"] = project_id
        return PreparedRequest(method="GET", url="/v1/evals", params=params)

    def prepare_delete(self, id: str) -> PreparedRequest:
        return PreparedRequest(method="DELETE", url=f"/v1/evals/{id}")


class DocumentsMixin:
    def prepare_get(self, evaluation_id: str, document_id: str) -> PreparedRequest:
        return PreparedRequest(method="GET", url=f"/v1/evals/{evaluation_id}/documents/{document_id}")

    def prepare_create(self, evaluation_id: str, document: MIMEData, annotation: Dict[str, Any]) -> PreparedRequest:
        # Serialize the MIMEData
        document_item = DocumentItem(mime_data=document, annotation=annotation, annotation_metadata=None)
        return PreparedRequest(method="POST", url=f"/v1/evals/{evaluation_id}/documents", data=document_item.model_dump(mode="json"))

    def prepare_list(self, evaluation_id: str, filename: Optional[str] = None) -> PreparedRequest:
        params = {}
        if filename:
            params["filename"] = filename
        return PreparedRequest(method="GET", url=f"/v1/evals/{evaluation_id}/documents", params=params)

    def prepare_update(self, evaluation_id: str, document_id: str, annotation: Dict[str, Any]) -> PreparedRequest:
        update_request = UpdateEvaluationDocumentRequest(annotation=annotation, annotation_metadata=None)
        return PreparedRequest(method="PUT", url=f"/v1/evals/{evaluation_id}/documents/{document_id}", data=update_request.model_dump(mode="json", exclude_none=True))

    def prepare_delete(self, evaluation_id: str, document_id: str) -> PreparedRequest:
        return PreparedRequest(method="DELETE", url=f"/v1/evals/{evaluation_id}/documents/{document_id}")


class IterationsMixin:
    def prepare_get(self, iteration_id: str) -> PreparedRequest:
        return PreparedRequest(method="GET", url=f"/v1/evals/iterations/{iteration_id}")

    def prepare_list(self, evaluation_id: str, model: Optional[str] = None) -> PreparedRequest:
        params = {}
        if model:
            params["model"] = model
        return PreparedRequest(method="GET", url=f"/v1/evals/{evaluation_id}/iterations", params=params)

    def prepare_create(
        self,
        evaluation_id: str,
        model: str,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        modality: Modality = "native",
        reasoning_effort: ChatCompletionReasoningEffort = "medium",
        image_resolution_dpi: int = 96,
        browser_canvas: BrowserCanvas = "A4",
        n_consensus: int = 1,
    ) -> PreparedRequest:
        inference_settings = InferenceSettings(
            model=model,
            temperature=temperature,
            modality=modality,
            reasoning_effort=reasoning_effort,
            image_resolution_dpi=image_resolution_dpi,
            browser_canvas=browser_canvas,
            n_consensus=n_consensus,
        )

        request = CreateIterationRequest(inference_settings=inference_settings, json_schema=json_schema)

        return PreparedRequest(method="POST", url=f"/v1/evals/{evaluation_id}/iterations/create", data=request.model_dump(exclude_none=True, mode="json"))

    def prepare_update(
        self,
        iteration_id: str,
        json_schema: Dict[str, Any],
        model: str,
        temperature: float = 0.0,
        modality: Modality = "native",
        reasoning_effort: ChatCompletionReasoningEffort = "medium",
        image_resolution_dpi: int = 96,
        browser_canvas: BrowserCanvas = "A4",
        n_consensus: int = 1,
    ) -> PreparedRequest:
        inference_settings = InferenceSettings(
            model=model,
            temperature=temperature,
            modality=modality,
            reasoning_effort=reasoning_effort,
            image_resolution_dpi=image_resolution_dpi,
            browser_canvas=browser_canvas,
            n_consensus=n_consensus,
        )

        iteration_data = Iteration(id=iteration_id, json_schema=json_schema, inference_settings=inference_settings, predictions=[])

        return PreparedRequest(method="PUT", url=f"/v1/evals/iterations/{iteration_id}", data=iteration_data.model_dump(exclude_none=True, mode="json"))

    def prepare_delete(self, iteration_id: str) -> PreparedRequest:
        return PreparedRequest(method="DELETE", url=f"/v1/evals/iterations/{iteration_id}")

    def prepare_compute_distances(self, iteration_id: str, document_id: str) -> PreparedRequest:
        return PreparedRequest(method="GET", url=f"/v1/evals/iterations/{iteration_id}/compute_distances/{document_id}")


class Evals(SyncAPIResource, EvalsMixin):
    """Evals API wrapper"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.documents = Documents(self._client)
        self.iterations = Iterations(self._client)

    def create(self, name: str, json_schema: Dict[str, Any], project_id: str | None = None) -> Evaluation:
        """
        Create a new evaluation.

        Args:
            name: The name of the evaluation
            json_schema: The JSON schema for the evaluation
            project_id: The project ID to associate with the evaluation

        Returns:
            Evaluation: The created evaluation
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_create(name, json_schema, project_id)
        response = self._client._prepared_request(request)
        return Evaluation(**response)

    def get(self, evaluation_id: str) -> Evaluation:
        """
        Get an evaluation by ID.

        Args:
            evaluation_id: The ID of the evaluation to retrieve

        Returns:
            Evaluation: The evaluation
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_get(evaluation_id)
        response = self._client._prepared_request(request)
        return Evaluation(**response)

    def update(
        self,
        evaluation_id: str,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        documents: Optional[List[EvaluationDocument]] = None,
        iterations: Optional[List[Iteration]] = None,
        default_inference_settings: Optional[InferenceSettings] = None,
    ) -> Evaluation:
        """
        Update an evaluation with partial updates.

        Args:
            evaluation_id: The ID of the evaluation to update
            name: Optional new name for the evaluation
            project_id: Optional new project ID
            json_schema: Optional new JSON schema
            documents: Optional list of documents to update
            iterations: Optional list of iterations to update
            default_inference_settings: Optional annotation properties

        Returns:
            Evaluation: The updated evaluation
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_update(
            evaluation_id=evaluation_id,
            name=name,
            project_id=project_id,
            json_schema=json_schema,
            documents=documents,
            iterations=iterations,
            default_inference_settings=default_inference_settings,
        )
        response = self._client._prepared_request(request)
        return Evaluation(**response)

    def list(self, project_id: str) -> List[Evaluation]:
        """
        List evaluations for a project.

        Args:
            project_id: The project ID to list evaluations for

        Returns:
            List[Evaluation]: List of evaluations
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_list(project_id)
        response = self._client._prepared_request(request)
        return [Evaluation(**item) for item in response.get("data", [])]

    def delete(self, evaluation_id: str) -> DeleteResponse:
        """
        Delete an evaluation.

        Args:
            evaluation_id: The ID of the evaluation to delete

        Returns:
            DeleteResponse: The response containing success status and ID
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_delete(evaluation_id)
        return self._client._prepared_request(request)


class Documents(SyncAPIResource, DocumentsMixin):
    """Documents API wrapper for evaluations"""

    def create(self, evaluation_id: str, document: Union[Path, str, IOBase, MIMEData, PIL.Image.Image, HttpUrl], annotation: Dict[str, Any]) -> EvaluationDocument:
        """
        Create a document for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            document: The document to process. Can be:
                - A file path (Path or str)
                - A file-like object (IOBase)
                - A MIMEData object
                - A PIL Image object
                - A URL (HttpUrl)
            annotation: The ground truth for the document

        Returns:
            EvaluationDocument: The created document
        Raises:
            HTTPException if the request fails
        """
        # Convert document to MIME data format
        mime_document: MIMEData = prepare_mime_document(document)

        # Let prepare_create handle the serialization
        request = self.prepare_create(evaluation_id, mime_document, annotation)
        response = self._client._prepared_request(request)
        return EvaluationDocument(**response)

    def list(self, evaluation_id: str, filename: Optional[str] = None) -> List[EvaluationDocument]:
        """
        List documents for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            filename: Optional filename to filter by

        Returns:
            List[EvaluationDocument]: List of documents
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_list(evaluation_id, filename)
        response = self._client._prepared_request(request)
        return [EvaluationDocument(**item) for item in response.get("data", [])]

    def get(self, evaluation_id: str, document_id: str) -> EvaluationDocument:
        """
        Get a document by ID.

        Args:
            evaluation_id: The ID of the evaluation
            document_id: The ID of the document

        Returns:
            EvaluationDocument: The document
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_get(evaluation_id, document_id)
        response = self._client._prepared_request(request)
        return EvaluationDocument(**response)

    def update(self, evaluation_id: str, document_id: str, annotation: Dict[str, Any]) -> EvaluationDocument:
        """
        Update a document.

        Args:
            evaluation_id: The ID of the evaluation
            document_id: The ID of the document
            annotation: The ground truth for the document

        Returns:
            EvaluationDocument: The updated document
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_update(evaluation_id, document_id, annotation)
        response = self._client._prepared_request(request)
        return EvaluationDocument(**response)

    def delete(self, evaluation_id: str, document_id: str) -> DeleteResponse:
        """
        Delete a document.

        Args:
            evaluation_id: The ID of the evaluation
            document_id: The ID of the document

        Returns:
            DeleteResponse: The response containing success status and ID
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_delete(evaluation_id, document_id)
        return self._client._prepared_request(request)


class Iterations(SyncAPIResource, IterationsMixin):
    """Iterations API wrapper for evaluations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list(self, evaluation_id: str, model: Optional[str] = None) -> List[Iteration]:
        """
        List iterations for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            model: Optional model to filter by

        Returns:
            List[Iteration]: List of iterations
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_list(evaluation_id, model)
        response = self._client._prepared_request(request)
        return [Iteration(**item) for item in response.get("data", [])]

    def create(
        self,
        evaluation_id: str,
        model: str,
        temperature: float = 0.0,
        modality: Modality = "native",
        json_schema: Optional[Dict[str, Any]] = None,
        reasoning_effort: ChatCompletionReasoningEffort = "medium",
        image_resolution_dpi: int = 96,
        browser_canvas: BrowserCanvas = "A4",
        n_consensus: int = 1,
    ) -> Iteration:
        """
        Create a new iteration for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            json_schema: The JSON schema for the iteration (if not set, we use the one of the eval)
            model: The model to use for the iteration
            temperature: The temperature to use for the model
            modality: The modality to use (text, image, etc.)
            reasoning_effort: The reasoning effort setting for the model (auto, low, medium, high)
            image_resolution_dpi: The DPI of the image. Defaults to 96.
            browser_canvas: The canvas size of the browser. Must be one of:
                - "A3" (11.7in x 16.54in)
                - "A4" (8.27in x 11.7in)
                - "A5" (5.83in x 8.27in)
                Defaults to "A4".
            n_consensus: Number of consensus iterations to perform

        Returns:
            Iteration: The created iteration
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_create(
            evaluation_id=evaluation_id,
            json_schema=json_schema,
            model=model,
            temperature=temperature,
            modality=modality,
            reasoning_effort=reasoning_effort,
            image_resolution_dpi=image_resolution_dpi,
            browser_canvas=browser_canvas,
            n_consensus=n_consensus,
        )
        response = self._client._prepared_request(request)
        return Iteration(**response)

    def delete(self, iteration_id: str) -> DeleteResponse:
        """
        Delete an iteration.

        Args:
            iteration_id: The ID of the iteration

        Returns:
            DeleteResponse: The response containing success status and ID
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_delete(iteration_id)
        return self._client._prepared_request(request)

    def compute_distances(self, iteration_id: str, document_id: str) -> DistancesResult:
        """
        Get distances for a document in an iteration.

        Args:
            iteration_id: The ID of the iteration
            document_id: The ID of the document

        Returns:
            DistancesResult: The distances
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_compute_distances(iteration_id, document_id)
        response = self._client._prepared_request(request)
        return DistancesResult(**response)


class AsyncEvals(AsyncAPIResource, EvalsMixin):
    """Async Evals API wrapper"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.documents = AsyncDocuments(self._client)
        self.iterations = AsyncIterations(self._client)

    async def create(self, name: str, json_schema: Dict[str, Any], project_id: str | None = None) -> Evaluation:
        """
        Create a new evaluation.

        Args:
            name: The name of the evaluation
            json_schema: The JSON schema for the evaluation
            project_id: The project ID to associate with the evaluation

        Returns:
            Evaluation: The created evaluation
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_create(name, json_schema, project_id)
        response = await self._client._prepared_request(request)
        return Evaluation(**response)

    async def get(self, evaluation_id: str) -> Evaluation:
        """
        Get an evaluation by ID.

        Args:
            evaluation_id: The ID of the evaluation to retrieve

        Returns:
            Evaluation: The evaluation
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_get(evaluation_id)
        response = await self._client._prepared_request(request)
        return Evaluation(**response)

    async def update(
        self,
        evaluation_id: str,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        documents: Optional[List[EvaluationDocument]] = None,
        iterations: Optional[List[Iteration]] = None,
        default_inference_settings: Optional[InferenceSettings] = None,
    ) -> Evaluation:
        """
        Update an evaluation with partial updates.

        Args:
            id: The ID of the evaluation to update
            name: Optional new name for the evaluation
            project_id: Optional new project ID
            json_schema: Optional new JSON schema
            documents: Optional list of documents to update
            iterations: Optional list of iterations to update
            default_inference_settings: Optional annotation properties

        Returns:
            Evaluation: The updated evaluation
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_update(
            evaluation_id=evaluation_id,
            name=name,
            project_id=project_id,
            json_schema=json_schema,
            documents=documents,
            iterations=iterations,
            default_inference_settings=default_inference_settings,
        )
        response = await self._client._prepared_request(request)
        return Evaluation(**response)

    async def list(self, project_id: Optional[str] = None) -> List[Evaluation]:
        """
        List evaluations for a project.

        Args:
            project_id: The project ID to list evaluations for

        Returns:
            List[Evaluation]: List of evaluations
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_list(project_id)
        response = await self._client._prepared_request(request)
        return [Evaluation(**item) for item in response.get("data", [])]

    async def delete(self, evaluation_id: str) -> DeleteResponse:
        """
        Delete an evaluation.

        Args:
            evaluation_id: The ID of the evaluation to delete

        Returns:
            DeleteResponse: The response containing success status and ID
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_delete(evaluation_id)
        return await self._client._prepared_request(request)


class AsyncDocuments(AsyncAPIResource, DocumentsMixin):
    """Async Documents API wrapper for evaluations"""

    async def create(self, evaluation_id: str, document: Union[Path, str, IOBase, MIMEData, PIL.Image.Image, HttpUrl], annotation: Dict[str, Any]) -> EvaluationDocument:
        """
        Create a document for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            document: The document to process. Can be:
                - A file path (Path or str)
                - A file-like object (IOBase)
                - A MIMEData object
                - A PIL Image object
                - A URL (HttpUrl)
            annotation: The ground truth for the document

        Returns:
            EvaluationDocument: The created document
        Raises:
            HTTPException if the request fails
        """
        # Convert document to MIME data format
        mime_document: MIMEData = prepare_mime_document(document)

        # Let prepare_create handle the serialization
        request = self.prepare_create(evaluation_id, mime_document, annotation)
        response = await self._client._prepared_request(request)
        return EvaluationDocument(**response)

    async def list(self, evaluation_id: str, filename: Optional[str] = None) -> List[EvaluationDocument]:
        """
        List documents for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            filename: Optional filename to filter by

        Returns:
            List[EvaluationDocument]: List of documents
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_list(evaluation_id, filename)
        response = await self._client._prepared_request(request)
        return [EvaluationDocument(**item) for item in response.get("data", [])]

    async def update(self, evaluation_id: str, document_id: str, annotation: Dict[str, Any]) -> EvaluationDocument:
        """
        Update a document.

        Args:
            evaluation_id: The ID of the evaluation
            document_id: The ID of the document
            annotation: The ground truth for the document

        Returns:
            EvaluationDocument: The updated document
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_update(evaluation_id, document_id, annotation)
        response = await self._client._prepared_request(request)
        return EvaluationDocument(**response)

    async def delete(self, evaluation_id: str, document_id: str) -> DeleteResponse:
        """
        Delete a document.

        Args:
            evaluation_id: The ID of the evaluation
            document_id: The ID of the document

        Returns:
            DeleteResponse: The response containing success status and ID
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_delete(evaluation_id, document_id)
        return await self._client._prepared_request(request)


class AsyncIterations(AsyncAPIResource, IterationsMixin):
    """Async Iterations API wrapper for evaluations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get(self, iteration_id: str) -> Iteration:
        """
        Get an iteration by ID.

        Args:
            iteration_id: The ID of the iteration

        Returns:
            Iteration: The iteration
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_get(iteration_id)
        response = await self._client._prepared_request(request)
        return Iteration(**response)

    async def list(self, evaluation_id: str, model: Optional[str] = None) -> List[Iteration]:
        """
        List iterations for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            model: Optional model to filter by

        Returns:
            List[Iteration]: List of iterations
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_list(evaluation_id, model)
        response = await self._client._prepared_request(request)
        return [Iteration(**item) for item in response.get("data", [])]

    async def create(
        self,
        evaluation_id: str,
        model: str,
        temperature: float = 0.0,
        modality: Modality = "native",
        json_schema: Optional[Dict[str, Any]] = None,
        reasoning_effort: ChatCompletionReasoningEffort = "medium",
        image_resolution_dpi: int = 96,
        browser_canvas: BrowserCanvas = "A4",
        n_consensus: int = 1,
    ) -> Iteration:
        """
        Create a new iteration for an evaluation.

        Args:
            evaluation_id: The ID of the evaluation
            json_schema: The JSON schema for the iteration
            model: The model to use for the iteration
            temperature: The temperature to use for the model
            modality: The modality to use (text, image, etc.)
            reasoning_effort: The reasoning effort setting for the model (auto, low, medium, high)
            image_resolution_dpi: The DPI of the image. Defaults to 96.
            browser_canvas: The canvas size of the browser. Must be one of:
                - "A3" (11.7in x 16.54in)
                - "A4" (8.27in x 11.7in)
                - "A5" (5.83in x 8.27in)
                Defaults to "A4".
            n_consensus: Number of consensus iterations to perform

        Returns:
            Iteration: The created iteration
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_create(
            evaluation_id=evaluation_id,
            json_schema=json_schema,
            model=model,
            temperature=temperature,
            modality=modality,
            reasoning_effort=reasoning_effort,
            image_resolution_dpi=image_resolution_dpi,
            browser_canvas=browser_canvas,
            n_consensus=n_consensus,
        )
        response = await self._client._prepared_request(request)
        return Iteration(**response)

    async def delete(self, iteration_id: str) -> DeleteResponse:
        """
        Delete an iteration.

        Args:
            iteration_id: The ID of the iteration

        Returns:
            DeleteResponse: The response containing success status and ID
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_delete(iteration_id)
        return await self._client._prepared_request(request)

    async def compute_distances(self, iteration_id: str, document_id: str) -> DistancesResult:
        """
        Get distances for a document in an iteration.

        Args:
            iteration_id: The ID of the iteration
            document_id: The ID of the document

        Returns:
            DistancesResult: The distances
        Raises:
            HTTPException if the request fails
        """
        request = self.prepare_compute_distances(iteration_id, document_id)
        response = await self._client._prepared_request(request)
        return DistancesResult(**response)
