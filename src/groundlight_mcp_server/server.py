import logging
from contextlib import asynccontextmanager
from functools import cache
from typing import Annotated, Any, Dict, List, Literal, Optional

from groundlight import (
    ROI,
    Detector,
    ExperimentalApi,
    Groundlight,
    ImageQuery,
    Rule,
    VerbEnum,
)
from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, Field

from groundlight_mcp_server.utils import load_image, render_bounding_boxes, to_mcp_image

logger = logging.getLogger(__name__)


@cache
def get_gl_client() -> Groundlight:
    logger.info("Initializing Groundlight client")
    return Groundlight()


@cache
def get_experimental_client() -> ExperimentalApi:
    logger.info("Initializing Experimental API client")
    return ExperimentalApi()


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle with type-safe context"""
    logger.info("Starting up Groundlight MCP server")
    # Initialize and cache the Groundlight client here so it can be reused across requests
    try:
        get_gl_client()
        get_experimental_client()
    except Exception as e:
        logger.error(f"Failed to initialize Groundlight client: {e}")
        raise e

    logger.info("Groundlight MCP server has started, listening for requests...")
    yield


mcp = FastMCP(
    "groundlight-mcp",
    lifespan=app_lifespan,
    description="Groundlight MCP server",
)


class BinaryDetectorConfig(BaseModel):
    mode: Literal["binary"] = "binary"


class MulticlassDetectorConfig(BaseModel):
    mode: Literal["multiclass"] = "multiclass"
    class_names: list[str] = Field(
        ..., description="List of class names for the multiclass detector"
    )


class CountingDetectorConfig(BaseModel):
    mode: Literal["counting"] = "counting"
    class_name: str = Field(..., description="Class name of the object to count")
    max_count: int = Field(20, description="Maximum count value")


class DetectorConfig(BaseModel):
    name: str = Field(..., description="Name of the detector")
    query: str = Field(..., description="Natural language query for the detector")
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        0.9, description="Confidence threshold for the detector"
    )
    mode: Literal["binary", "multiclass", "counting"]
    config: BinaryDetectorConfig | MulticlassDetectorConfig | CountingDetectorConfig = (
        Field(..., discriminator="mode")
    )

    def to_creation_params(self) -> dict:
        """Returns flattened parameters for detector creation, excluding mode."""
        params = {k: v for k, v in self.dict().items() if k != "mode" and k != "config"}
        if self.config:
            config_dict = self.config.dict()
            config_dict.pop("mode", None)
            params.update(config_dict)
        return params


@mcp.tool(
    name="create_detector",
    description=(
        "Create a detector based on the specified configuration. Supports three modes:\n"
        "1. Binary: Answers 'yes' or 'no' to a natural-language query about images.\n"
        "2. Multiclass: Classifies images into predefined categories based on natural-language queries.\n"
        "3. Counting: Counts occurrences of specified objects in images using natural-language descriptions.\n\n"
        "All detectors analyze images to answer natural-language queries and return confidence scores indicating "
        "result reliability. If confidence falls below the specified threshold, the query is escalated to human "
        "review. Detectors improve over time through continuous learning from feedback and additional examples."
    ),
)
def create_detector(config: DetectorConfig) -> Detector:
    if config.mode == "binary":
        gl = get_gl_client()
        return gl.create_detector(
            **config.to_creation_params(),
        )
    elif config.mode == "multiclass":
        gl_exp = get_experimental_client()
        return gl_exp.create_multiclass_detector(
            **config.to_creation_params(),
        )
    elif config.mode == "counting":
        gl_exp = get_experimental_client()
        return gl_exp.create_counting_detector(
            **config.to_creation_params(),
        )
    else:
        raise AssertionError(
            f"Invalid detector mode: {config.mode}. Supported modes are 'binary', 'multiclass', and 'counting'."
        )


@mcp.tool(
    name="get_detector",
    description="Get a detector by its ID.",
)
def get_detector(detector_id: str) -> Detector:
    gl = get_gl_client()
    return gl.get_detector(detector_id)


@mcp.tool(
    name="list_detectors",
    description="List all detectors associated with the current user.",
)
def list_detectors() -> list[Detector]:
    gl = get_gl_client()
    all_detectors = []
    page = 1
    page_size = 100

    while True:
        response = gl.list_detectors(page=page, page_size=page_size)
        all_detectors.extend(response.results)
        if response.next is None:
            break
        page += 1

    return all_detectors


@mcp.tool(
    name="submit_image_query",
    description=(
        "Submit an image to be answered by the specified detector. The image can be provided as a "
        "file path, URL, or raw bytes. The detector will return a response with a label and confidence score."
    ),
)
def submit_image_query(detector_id: str, image: str | bytes) -> ImageQuery:
    gl = get_gl_client()
    img = load_image(image)
    iq = gl.submit_image_query(detector=detector_id, image=img)
    return iq


@mcp.tool(
    name="get_image_query",
    description="Get an existing image query by its ID.",
)
def get_image_query(image_query_id: str) -> ImageQuery:
    gl = get_gl_client()
    return gl.get_image_query(id=image_query_id)


@mcp.tool(
    name="list_image_queries",
    description="List all image queries associated with the specified detector. Note that this may return a large number of results.",
)
def list_image_queries(detector_id: str) -> list[ImageQuery]:
    gl = get_gl_client()
    all_queries = []
    page = 1
    page_size = 100

    while True:
        response = gl.list_image_queries(
            detector_id=detector_id, page=page, page_size=page_size
        )
        all_queries.extend(response.results)
        if response.next is None:
            break
        page += 1

    return all_queries


@mcp.tool(
    name="get_image",
    description="Get the image associated with an image query by its ID. Optionally annotate with bounding boxes on the image if available.",
)
def get_image(image_query_id: str, annotate: bool = False) -> Image:
    gl_exp = get_experimental_client()
    image_bytes = gl_exp.get_image(iq_id=image_query_id)

    if not annotate:
        return to_mcp_image(image_bytes)

    # Get image query to check for ROIs
    iq = gl_exp.get_image_query(id=image_query_id)
    if iq.rois is None or len(iq.rois) == 0:
        logger.info(
            f"No ROIs found for {image_query_id=}. Returning original image without bounding boxes."
        )
        return to_mcp_image(image_bytes)

    # Load image and draw bounding boxes
    image = load_image(image_bytes)
    annotated_image = render_bounding_boxes(image, iq.rois)
    return to_mcp_image(annotated_image)


class WebhookConfig(BaseModel):
    url: str = Field(..., description="URL to send webhook notifications to")
    include_image: bool = Field(
        False, description="Whether to include the image in the webhook payload"
    )


class EmailConfig(BaseModel):
    email: str = Field(..., description="Email address to send notifications to")
    include_image: bool = Field(
        False, description="Whether to include the image in the email"
    )


class TextConfig(BaseModel):
    phone_number: str = Field(
        ..., description="Phone number to send text notifications to"
    )
    include_image: bool = Field(
        False, description="Whether to include the image in the text message"
    )


class ConditionConfig(BaseModel):
    verb: VerbEnum = Field(
        ...,
        description="Condition verb that defines when a rule should trigger",
    )
    parameters: dict[str, int | str] = Field(
        ...,
        description="Condition parameters that configure the trigger behavior based on the verb. "
        "For ANSWERED_CONSECUTIVELY: {'num_consecutive_labels': N, 'label': 'YES/NO'}, "
        "For CHANGED_TO: {'label': 'YES/NO'}, "
        "For ANSWERED_WITHIN_TIME: {'time_value': N, 'time_unit': 'MINUTES/HOURS/DAYS'}, "
        "These parameters define specific scenarios that will trigger actions like sending notifications.",
    )


class AlertConfig(BaseModel):
    name: str = Field(..., description="Name of the rule")
    detector_id: str = Field(
        ..., description="ID of the detector to attach the rule to"
    )
    condition: ConditionConfig = Field(
        ..., description="Condition that triggers the rule"
    )
    webhook_action: Optional[List[WebhookConfig]] = Field(
        None, description="Webhook actions to perform when the rule is triggered"
    )
    email_action: Optional[List[EmailConfig]] = Field(
        None, description="Email actions to perform when the rule is triggered"
    )
    text_action: Optional[List[TextConfig]] = Field(
        None, description="Text message actions to perform when the rule is triggered"
    )
    enabled: bool = Field(True, description="Whether the rule is enabled")
    human_review_required: bool = Field(
        False, description="Whether human review is required before triggering the rule"
    )


@mcp.tool(
    name="create_alert",
    description="Create an alert for a detector that triggers actions when specific conditions are met.",
)
def create_alert(config: AlertConfig) -> Rule:
    gl_exp = get_experimental_client()

    # Create condition
    condition = gl_exp.make_condition(
        verb=config.condition.verb, parameters=config.condition.parameters
    )

    # Initialize action lists
    actions = []
    webhook_actions = []

    # Add actions by type
    if config.webhook_action:
        webhook_actions = [
            gl_exp.make_webhook_action(w.url, include_image=w.include_image)
            for w in config.webhook_action
        ]

    if config.email_action:
        actions.extend(
            [
                gl_exp.make_action("EMAIL", e.email, include_image=e.include_image)
                for e in config.email_action
            ]
        )

    if config.text_action:
        actions.extend(
            [
                gl_exp.make_action(
                    "TEXT", t.phone_number, include_image=t.include_image
                )
                for t in config.text_action
            ]
        )

    # Create and return the alert
    return gl_exp.create_alert(
        detector=config.detector_id,
        name=config.name,
        condition=condition,
        actions=actions,
        webhook_actions=webhook_actions,
        enabled=config.enabled,
        human_review_required=config.human_review_required,
        snooze_time_enabled=False,
    )


@mcp.tool(
    name="list_alerts",
    description="List all alerts associated with a detector.",
)
def list_alerts(page: int = 1, page_size: int = 100) -> List[Rule]:
    gl_exp = get_experimental_client()
    response = gl_exp.list_rules(page=page, page_size=page_size)
    return response.results


@mcp.tool(
    name="delete_alert",
    description="Delete an alert by its alert ID.",
)
def delete_alert(alert_id: str) -> None:
    gl_exp = get_experimental_client()
    gl_exp.delete_rule(rule_id=alert_id)
    return None


@mcp.tool(
    name="add_label",
    description=(
        "Provide a label (annotation) for an image query. This is used for training detectors "
        "or correcting results. For counting detectors, you can optionally provide regions of interest."
    ),
)
def add_label(
    image_query_id: str, label: int | str, rois: list[ROI] | None = None
) -> None:
    gl = get_gl_client()
    gl.add_label(image_query=image_query_id, label=label, rois=rois)
    return None


@mcp.tool(
    name="get_detector_evaluation_metrics",
    description="Get detailed evaluation metrics for a detector, including confusion matrix and examples.",
)
def get_detector_evaluation(detector_id: str) -> Dict[str, Any]:
    gl_exp = get_experimental_client()
    evaluation = gl_exp.get_detector_evaluation(detector=detector_id)
    return evaluation


@mcp.tool(
    name="update_detector_confidence_threshold",
    description="Update the confidence threshold for a detector.",
)
def update_detector_confidence_threshold(
    detector_id: str, confidence_threshold: float
) -> None:
    gl = get_gl_client()
    gl.update_detector_confidence_threshold(
        detector=detector_id, confidence_threshold=confidence_threshold
    )


@mcp.tool(
    name="update_detector_escalation_type",
    description=(
        "Update the escalation type for a detector. This determines when queries are sent for human review. "
        "Options: 'STANDARD' (escalate based on confidence threshold) or 'NO_HUMAN_LABELING' (never escalate)."
    ),
)
def update_detector_escalation_type(
    detector_id: str, escalation_type: Literal["STANDARD", "NO_HUMAN_LABELING"]
) -> None:
    gl_exp = get_experimental_client()
    gl_exp.update_detector_escalation_type(
        detector=detector_id, escalation_type=escalation_type
    )

# Documentation directory
DOCS_DIR = Path("/opt/groundlight/docs/")

@mcp.resource("docs://list")
async def list_docs() -> list[Resource]:
    """List all available documentation resources"""
    resources = []
    
    # Walk through the docs directory
    for file_path in DOCS_DIR.rglob("*.md"):
        # Create relative path for cleaner URIs
        relative_path = file_path.relative_to(DOCS_DIR)
        uri = f"docs://{relative_path.as_posix()}"
        
        # Extract a friendly name from the filename
        name = file_path.stem.replace("-", " ").replace("_", " ").title()
        
        resources.append(Resource(
            uri=uri,
            name=name,
            description=f"Documentation: {name}",
            mimeType="text/markdown"
        ))
    
    return resources

@mcp.resource("docs://{path:path}")
async def read_doc(path: str) -> str:
    """Read a specific documentation file"""
    file_path = DOCS_DIR / path
    
    # Security check - ensure the path is within our docs directory
    try:
        file_path = file_path.resolve()
        if not file_path.is_relative_to(DOCS_DIR):
            raise ValueError("Invalid path")
    except (ValueError, OSError):
        raise ValueError(f"Documentation not found: {path}")
    
    # Read and return the file content
    if file_path.exists() and file_path.is_file():
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Documentation not found: {path}")

@mcp.resource("docs://index")
async def docs_index() -> str:
    """Get an index of all available documentation"""
    docs = []
    for file_path in sorted(DOCS_DIR.rglob("*.md")):
        relative_path = file_path.relative_to(DOCS_DIR)
        docs.append(f"- [{file_path.stem}](docs://{relative_path.as_posix()})")

    return "# Groundlight Documentation Index\n\n" + "\n".join(docs)
