import logging
from contextlib import asynccontextmanager
from functools import cache
from typing import Annotated

from groundlight import Detector, ExperimentalApi, Groundlight, ImageQuery
from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field

from groundlight_mcp_server.utils import load_image

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
    name="get_image",
    description="Get the image associated with an image query by its ID.",
)
def get_image(image_query_id: str) -> Image:
    gl_exp = get_experimental_client()
    image_bytes = gl_exp.get_image(iq_id=image_query_id)
    if hasattr(image_bytes, "read"):  # Convert BufferedReader to bytes if needed
        image_bytes = image_bytes.read()
    return Image(data=image_bytes, format="jpeg")


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
    name="create_binary_detector",
    description=(
        "Create a binary classification detector that responds with 'yes' or 'no' (or sometimes 'unclear') to a "
        "natural-language query about images. This detector analyzes images to answer questions that have binary "
        "outcomes and returns a calibrated confidence score indicating the likelihood of the answer being correct. "
        "A confidence_threshold indicates the minimum confidence required for the detector to return an answer. If "
        "the confidence score is below this threshold, the detector will escalate the question to a human reviewer. "
        "Notably, Groundlight detectors get stronger and more accurate over time as they learn from human feedback. "
    ),
)
def create_binary_detector(
    detector_name: str,
    detector_query: str,
    detector_confidence_threshold: Annotated[float, Field(ge=0.5, le=1.0)] = 0.9,
) -> Detector:
    gl = get_gl_client()
    return gl.create_detector(
        name=detector_name,
        query=detector_query,
        confidence_threshold=detector_confidence_threshold,
    )
