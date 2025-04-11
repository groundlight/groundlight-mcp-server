import logging
from contextlib import asynccontextmanager
from functools import cache
from typing import Annotated, Literal

from groundlight import Detector, ExperimentalApi, Groundlight, ImageQuery
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
