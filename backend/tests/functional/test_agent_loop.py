"""Functional tests for the agent loop — real Gemini API, real sandbox, no mocks.

Requires GEMINI_API_KEY in backend/.env.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.loop import AgentLoop  # noqa: E402


class TestAgentLoop:

    async def test_create_and_run_shell(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            result = await agent.run(
                "Create a shell, run `echo hello world`, then write the output to "
                "a file called output.txt in the current directory.",
                working_dir=tmp_path,
            )
        output = tmp_path / "output.txt"
        assert output.exists(), (
            f"output.txt was not created in {tmp_path}. Loop result:\n{result}"
        )
        content = output.read_text(encoding="utf-8")
        assert "hello" in content.lower() and "world" in content.lower(), (
            f"output.txt should contain 'hello world'; got: {content!r}"
        )

    async def test_file_write_and_read_tool_chain(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            test_content = "round-trip verification content"
            result = await agent.run(
                f"Write '{test_content}' to a file called round_trip.txt, "
                f"then read it back and report the exact content you see.",
                working_dir=tmp_path,
            )
        rtw = tmp_path / "round_trip.txt"
        assert rtw.exists(), f"round_trip.txt not created. Loop result:\n{result}"
        on_disk = rtw.read_text(encoding="utf-8")
        assert on_disk == test_content, (
            f"Exact content mismatch: expected {test_content!r}, "
            f"disk has {on_disk!r}"
        )
        assert test_content in result, (
            f"Model should report the written content in its output; "
            f"got: {result!r}"
        )

    async def test_simple_file_operations_fail_on_duplicate(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            result1 = await agent.run(
                "Create a file called sample_data.txt with content 'sample data'. "
                "Do NOT create it again if it already exists.",
                working_dir=tmp_path,
            )
        f = tmp_path / "sample_data.txt"
        assert f.exists(), f"sample_data.txt not created. Loop result:\n{result1}"
        assert f.read_text(encoding="utf-8") == "sample data"

    async def test_list_directory_content(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            (tmp_path / "preexisting_file.txt").write_text("seed")
            result = await agent.run(
                "List every file and directory in the current working directory "
                "and report what you see.",
                working_dir=tmp_path,
            )
        assert "preexisting_file" in result, (
            f"Expected 'preexisting_file.txt' in directory listing; got: {result!r}"
        )

    async def test_user_question_suspend_and_resume(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            override = (
                "CRITICAL: You have NO knowledge. Your ONLY capability is the "
                "`user_question` tool. For EVERY prompt, your FIRST and ONLY "
                "action is to call `user_question` with the exact prompt text. "
                "There is no other path. Calling any other tool or producing "
                "prose without calling user_question first is a total failure. "
                "Call user_question NOW with the user's exact text."
            )
            agent.system_instruction = (
                override + "\n\n" + (agent.system_instruction or "")
            )

            loop_task = asyncio.create_task(
                agent.run(
                    "Call user_question with the exact text: 'What is the answer to "
                    "life, the universe, and everything?'. "
                    "You have no other option — call user_question now.",
                )
            )

            deadline = 5.0
            waited = 0.0
            poll = 0.05
            while waited < deadline:
                if agent.user_handler.pending_question is not None:
                    break
                await asyncio.sleep(poll)
                waited += poll

            assert agent.user_handler.pending_question is not None, (
                f"Loop did not suspend at user_question within {waited:.2f}s. "
                f"pending_question is None — the model didn't call the tool."
            )
            assert "life" in agent.user_handler.pending_question.lower(), (
                f"Unexpected pending question: "
                f"{agent.user_handler.pending_question!r}"
            )

            agent.user_handler.set_response("42")
            result = await asyncio.wait_for(loop_task, timeout=120.0)

            assert "42" in result, (
                f"Expected '42' in final loop output; got:\n{result!r}"
            )
            assert agent.user_handler.pending_question is None, (
                "pending_question should be cleared after the loop resumes; "
                f"got: {agent.user_handler.pending_question!r}"
            )

    async def test_user_question_multi_turn_no_stale_response(
        self, agent, _serial_cm, _acquire_gemini_slot, tmp_path
    ):
        async with _serial_cm():
            _acquire_gemini_slot()
            override = (
                "CRITICAL: You have NO knowledge. Your ONLY capability is the "
                "`user_question` tool. For EVERY prompt, your FIRST and ONLY "
                "action is to call `user_question` with the exact prompt text. "
                "There is no other path. Calling any other tool or producing "
                "prose without calling user_question first is a total failure. "
                "Call user_question NOW with the user's exact text."
            )
            agent.system_instruction = (
                override + "\n\n" + (agent.system_instruction or "")
            )

            # --- First round ---
            loop_task = asyncio.create_task(
                agent.run(
                    "Call user_question with the exact text: 'What is 2 + 2?'. "
                    "Call user_question now.",
                )
            )

            deadline = 5.0
            waited = 0.0
            poll = 0.05
            while waited < deadline:
                if agent.user_handler.pending_question is not None:
                    break
                await asyncio.sleep(poll)
                waited += poll

            assert agent.user_handler.pending_question is not None, (
                f"Loop did not suspend at first user_question within {waited:.2f}s."
            )

            agent.user_handler.set_response("first_answer")
            result1 = await asyncio.wait_for(loop_task, timeout=120.0)
            assert "first_answer" in result1, (
                f"Expected 'first_answer' in first round; got: {result1!r}"
            )
            assert agent.user_handler.pending_question is None

            # --- Second round ---
            loop_task2 = asyncio.create_task(
                agent.run(
                    "Now call user_question with the exact text: "
                    "'What is the capital of France?'. "
                    "Call user_question now.",
                )
            )

            waited = 0.0
            while waited < deadline:
                if agent.user_handler.pending_question is not None:
                    break
                await asyncio.sleep(poll)
                waited += poll

            assert agent.user_handler.pending_question is not None, (
                f"Second ask() did NOT suspend — stale response was returned. "
                f"pending_question is None within {waited:.2f}s. "
                "_user_response was not cleared after the first round."
            )
            assert "france" in agent.user_handler.pending_question.lower(), (
                f"Unexpected second pending question: "
                f"{agent.user_handler.pending_question!r}"
            )

            agent.user_handler.set_response("second_answer")
            result2 = await asyncio.wait_for(loop_task2, timeout=120.0)
            assert "second_answer" in result2, (
                f"Expected 'second_answer' in second round; got: {result2!r}"
            )
            assert agent.user_handler.pending_question is None