"""Functional tests for basic agent tasks — real Gemini API, real sandbox, no mocks.

Requires GEMINI_API_KEY in backend/.env.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestBasicTasks:

    async def test_create_hello_txt_containing_world(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            result = await agent.run(
                "Create a file called hello.txt containing the word world.",
                working_dir=tmp_path,
            )
        hello_file = tmp_path / "hello.txt"
        assert hello_file.exists(), (
            f"hello.txt was not created in {tmp_path}. Loop result:\n{result}"
        )
        content = hello_file.read_text(encoding="utf-8")
        assert "world" in content.lower(), (
            f"hello.txt content should contain 'world'; got: {content!r}"
        )

    async def test_two_distinct_shells(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            result = await agent.run(
                "In one shell run `echo one`, in a separate (different) shell run "
                "`echo two`, then tell me both outputs.",
                working_dir=tmp_path,
            )
        run_shell_targets = [
            call["args"]["shell_id"]
            for call in agent.tool_calls
            if call["name"] == "run_in_shell"
        ]
        distinct_ids = set(run_shell_targets)
        assert len(distinct_ids) >= 2, (
            "Expected two distinct shell_ids to be used by run_in_shell, "
            f"but saw only: {distinct_ids}."
        )
        result_lower = (result or "").lower()
        assert "one" in result_lower, (
            f"Final output should report 'one'; got:\n{result}"
        )
        assert "two" in result_lower, (
            f"Final output should report 'two'; got:\n{result}"
        )