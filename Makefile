.PHONY: install-server build-docker run-docker push-docker

build-docker:  # Build the Docker image for the Groundlight MCP server
	docker build -t groundlight-mcp-server .

run-docker:  # Run the Docker container
	docker run -it --rm -e GROUNDLIGHT_API_TOKEN groundlight-mcp-server

push-docker:  # Push the Docker image to the registry
	docker tag groundlight-mcp-server groundlight/groundlight-mcp-server:latest
	docker push groundlight/groundlight-mcp-server
