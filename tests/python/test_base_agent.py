import pytest
from templates.base_agent import BaseAgent, Thought, Action


class DummyAgent(BaseAgent):
    async def execute(self, task, action: Action):
        return {"echo": str(task), "strategy": action.action_type}


@pytest.mark.asyncio
async def test_base_agent_contract_runs_end_to_end():
    agent = DummyAgent(name="dummy")

    # Think
    thought = await agent.think(task="hello", context={})
    assert isinstance(thought, Thought)
    assert thought.strategy in agent._get_available_strategies()

    # Act/Execute/Observe via pipeline
    result = await agent.process_task(task="hello", context={})
    assert isinstance(result, dict)
    assert "echo" in result

    # Metrics updated
    assert agent.total_tasks == 1
    assert 0.0 <= agent.get_success_rate() <= 1.0

