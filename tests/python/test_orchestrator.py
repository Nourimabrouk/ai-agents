import pytest

from orchestrator import AgentOrchestrator, Task
from templates.base_agent import BaseAgent, Action


class DummyAgent(BaseAgent):
    async def execute(self, task, action: Action):
        # Deterministic result for orchestration tests
        return {"ok": True, "task": str(task)}


@pytest.mark.asyncio
async def test_orchestrator_delegates_to_single_agent():
    orch = AgentOrchestrator(name="test-orch")
    agent = DummyAgent(name="worker-1")
    orch.register_agent(agent)

    task = Task(id="t1", description="Simple task", requirements={})
    result = await orch.delegate_task(task)

    assert result and result.get("ok") is True
    assert task.status == "completed"
    assert orch.total_tasks_completed == 1

