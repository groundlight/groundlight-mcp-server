<p align="center">
<img src="images/image0_and_claude_zoomed_in.png">
</p>

<p align="center">
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/License-Apache%202%2E0-yellow?style=for-the-badge
    " alt="License: Apache2.0">
  <a href="https://www.groundlight.ai/blog/building-computer-vision-applications-with-the-groundlight-mcp-server">
    <img src="https://img.shields.io/badge/Read%20More-Blog-orange?style=for-the-badge"  alt="Read More">
  </a>
</p>
  </a>
</p>

# groundlight-mcp-server by <img src=resources/images/gl_logo.png height=25>

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

8. **create_alert**
   - Description: Create an alert for a detector that triggers actions when specific conditions are met.
   - Input: `config` (AlertConfig object with name, detector_id, condition, and optional webhook_action, email_action, text_action, enabled, and human_review_required fields)
   - Returns: `Rule` object

9. **list_alerts**
   - Description: List all alerts (rules) in the system. (Note: Not filtered by detector in the current implementation.)
   - Input: `page` (integer, default: 1), `page_size` (integer, default: 100)
   - Returns: List of `Rule` objects

10. **delete_alert**
    - Description: Delete an alert (rule) by its alert ID.
    - Input: `alert_id` (string)
    - Returns: None

11. **add_label**
    - Description: Provide a label (annotation) for an image query. This is used for training detectors or correcting results. For counting detectors, you can optionally provide regions of interest.
    - Input: `image_query_id` (string), `label` (integer or string), `rois` (optional list)
    - Returns: None

12. **get_detector_evaluation_metrics**
    - Description: Get detailed evaluation metrics for a detector, including confusion matrix and examples.
    - Input: `detector_id` (string)
    - Returns: Dictionary of evaluation metrics

13. **update_detector_confidence_threshold**
    - Description: Update the confidence threshold for a detector.
    - Input: `detector_id` (string), `confidence_threshold` (float)
    - Returns: None

14. **update_detector_escalation_type**
    - Description: Update the escalation type for a detector. This determines when queries are sent for human review. Options: 'STANDARD' (escalate based on confidence threshold) or 'NO_HUMAN_LABELING' (never escalate).
    - Input: `detector_id` (string), `escalation_type` (string, either "STANDARD" or "NO_HUMAN_LABELING")
    - Returns: None


## Configuration

### Usage with Claude Desktop
Add this to your claude_desktop_config.json:
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

### Usage with Zed
Add this to your settings.json:
```json
{
  "context_servers": {
    "groundlight": {
      "command": {
        "path": "docker",
        "args": [
          "run",
          "--rm",
          "-i",
          "-e",
          "GROUNDLIGHT_API_TOKEN",
          "groundlight/groundlight-mcp-server"
        ],
        "env": {
          "GROUNDLIGHT_API_TOKEN": "YOUR_API_TOKEN_HERE"
        }
      }
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
s
