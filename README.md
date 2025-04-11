# groundlight-mcp-server

## Overview
A Model Context Protocol (MCP) server for interacting with Groundlight. This server provides tools to create and list both Detectors and ImageQueries.

This MCP server is still in early development. The functionality and available tools are subject to change and expansion as we continue to develop and improve the server.

### Tools
The following tools are available in the Groundlight MCP server:

1. **create_detector**
   - Description: Create a detector based on the specified configuration. Supports three modes:
     1. Binary: Answers 'yes' or 'no' to a natural-language query about images.
     2. Multiclass: Classifies images into predefined categories based on natural-language queries.
     3. Counting: Counts occurrences of specified objects in images using natural-language descriptions.

     All detectors analyze images to answer natural-language queries and return confidence scores indicating result reliability. If confidence falls below the specified threshold, the query is escalated to human review. Detectors improve over time through continuous learning from feedback and additional examples.
   - Input: `config` (DetectorConfig object with name, query, confidence_threshold, mode, and mode-specific configuration)
   - Returns: `Detector` object

2. **get_detector**
   - Description: Get a detector by its ID.
   - Input: `detector_id` (string)
   - Returns: `Detector` object

3. **list_detectors**
   - Description: List all detectors associated with the current user.
   - Input: None
   - Returns: List of `Detector` objects

4. **submit_image_query**
   - Description: Submit an image to be answered by the specified detector. The image can be provided as a file path, URL, or raw bytes. The detector will return a response with a label and confidence score.
   - Input: `detector_id` (string), `image` (string or bytes)
   - Returns: `ImageQuery` object

5. **get_image_query**
   - Description: Get an existing image query by its ID.
   - Input: `image_query_id` (string)
   - Returns: `ImageQuery` object

6. **list_image_queries**
   - Description: List all image queries associated with the specified detector. Note that this may return a large number of results.
   - Input: `detector_id` (string)
   - Returns: List of `ImageQuery` objects

7. **get_image**
   - Description: Get the image associated with an image query by its ID. Optionally annotate with bounding boxes on the image if available.
   - Input: `image_query_id` (string), `annotate` (boolean, default: false)
   - Returns: `Image` object


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
