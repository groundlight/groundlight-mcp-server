.PHONY: install-server

install-server:  # Install the Groundlight server in Claude Desktop
	uv run mcp install src/groundlight_mcp_server/server.py --name "groundlight" -v GROUNDLIGHT_API_TOKEN=$(GROUNDLIGHT_API_TOKEN)

mcp-inspector:  # Run the MCP inspector
	uv run mcp dev src/groundlight_mcp_server/server.py