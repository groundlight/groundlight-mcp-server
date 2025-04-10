.PHONY: install-server

install-server:  # Install the Groundlight server in Claude Desktop
	uv run mcp install src/groundlight_mcp_server/server.py --name "groundlight" -v GROUNDLIGHT_API_TOKEN=$(GROUNDLIGHT_API_TOKEN)

mcp-inspector:  # Run the MCP inspector
	uv run mcp dev src/groundlight_mcp_server/server.py

build-docker:  # Build the Docker image for the Groundlight MCP server
	docker build -t groundlight-mcp-server .

run-docker:  # Run the Docker container
	docker run -it --rm -e GROUNDLIGHT_API_TOKEN groundlight-mcp-server

push-docker:  # Push the Docker image to the registry
	docker tag groundlight-mcp-server groundlight/groundlight-mcp-server:latest
	docker push groundlight/groundlight-mcp-server
