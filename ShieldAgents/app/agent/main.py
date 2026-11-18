"""AgentCore Runtime entrypoint for the Shield Insurance agents.

The agent logic is in agentcore_runtime.py, copied here from claim-intake-agent/
by `python sync_agent_code.py`. Importing it registers the @app.entrypoint.
"""
from agentcore_runtime import app

if __name__ == "__main__":
    app.run()
