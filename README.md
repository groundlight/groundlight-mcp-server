# groundlight-mcp-server

## Overview
A Model Context Protocol (MCP) server for interacting with Groundlight. This server provides tools to create and list both Detectors and ImageQueries.

This MCP server is still in early development. The functionality and available tools are subject to change and expansion as we continue to develop and improve the server.

### Tools
The following tools are available in the Groundlight MCP server:

1. **get_detector**
   - Description: Get a detector by its ID.
   - Input: `detector_id` (string)
   - Returns: `Detector` object

2. **list_detectors**
   - Description: List all detectors associated with the current user.
   - Input: None
   - Returns: List of `Detector` objects

3. **submit_image_query**
   - Description: Submit an image to be answered by the specified detector. The image can be provided as a file path, URL, or raw bytes. The detector will return a response with a label and confidence score.
   - Input: `detector_id` (string), `image` (string or bytes)
   - Returns: `ImageQuery` object

4. **get_image_query**
   - Description: Get an existing image query by its ID.
   - Input: `image_query_id` (string)
   - Returns: `ImageQuery` object

5. **get_image**
   - Description: Get the image associated with an image query by its ID.
   - Input: `image_query_id` (string)
   - Returns: `Image` object

6. **list_image_queries**
   - Description: List all image queries associated with the specified detector. Note that this may return a large number of results.
   - Input: `detector_id` (string)
   - Returns: List of `ImageQuery` objects

7. **create_binary_detector**
   - Description: Create a binary classification detector that responds with 'yes' or 'no' (or sometimes 'unclear') to a natural-language query about images. This detector analyzes images to answer questions that have binary outcomes and returns a calibrated confidence score indicating the likelihood of the answer being correct. A confidence_threshold indicates the minimum confidence required for the detector to return an answer. If the confidence score is below this threshold, the detector will escalate the question to a human reviewer. Notably, Groundlight detectors get stronger and more accurate over time as they learn from human feedback.
   - Input: `detector_name` (string), `detector_query` (string), `detector_confidence_threshold` (float, range 0.5-1.0, default 0.9)
   - Returns: `Detector` object


## Configuration

### Usage with Claude Desktop
Add this to your claude_desktop_config.json:

#### Docker
```json
"mcpServers": {
  "groundlight": {
    "command": "docker",
    "args": ["run", "--rm", "-i", "-e", "GROUNDLIGHT_API_TOKEN", "groundlight/groundlight-mcp-server"],
    "env": {
        "GROUNDLIGHT_API_TOKEN": "YOUR_API_TOKEN_HERE"
    }
  }
}
```

## Development

Build the Docker image locally:
```bash
make build-docker
```

Run the Docker image locally:
```bash
make run-docker
```

[Groundlight Internal] Push the Docker image to Docker Hub (requires DockerHub credentials):
```bash
make push-docker
```