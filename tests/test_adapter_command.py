from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_agent.adapters.command import run_stage_adapter


class StageAdapterBootstrapTests(unittest.TestCase):
    def test_run_stage_adapter_bootstraps_shell_init_and_conda_env(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            manifest_path = root / "input_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            script_path = root / "adapter.py"
            script_path.write_text(
                """
import json
import os
from pathlib import Path

output_dir = Path(os.environ["KAGGLE_AGENT_OUTPUT_DIR"])
payload = {
    "stage": os.environ["KAGGLE_AGENT_STAGE"],
    "shell_ready": os.environ.get("TEST_SHELL_READY", ""),
    "active_env": os.environ.get("TEST_ACTIVE_ENV", ""),
}
(output_dir / "report.json").write_text(json.dumps(payload) + "\\n", encoding="utf-8")
(output_dir / "report.md").write_text("# ok\\n", encoding="utf-8")
""".strip()
                + "\n",
                encoding="utf-8",
            )

            shell_init = "\n".join(
                [
                    "export TEST_SHELL_READY=1",
                    'conda() { if [ "$1" = "activate" ]; then export TEST_ACTIVE_ENV="$2"; return 0; fi; return 1; }',
                ]
            )
            command = f"{sys.executable} {script_path}"

            result = run_stage_adapter(
                command,
                stage="report",
                workspace_root=root,
                input_manifest_path=manifest_path,
                output_dir=output_dir,
                shell_init=shell_init,
                conda_env="kaggle-agent",
            )

            payload = json.loads(result["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "report")
            self.assertEqual(payload["shell_ready"], "1")
            self.assertEqual(payload["active_env"], "kaggle-agent")


if __name__ == "__main__":
    unittest.main()
