import pathlib
import sys
import tomllib
import unittest

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import slm_post_train


class TestScaffolding(unittest.TestCase):
    def test_python_version_file(self):
        py_version_path = ROOT_DIR / ".python-version"
        self.assertTrue(py_version_path.is_file(), ".python-version must exist")
        content = py_version_path.read_text().strip()
        self.assertEqual(content, "3.11", ".python-version must pin Python 3.11")

    def test_gitignore_file(self):
        gitignore_path = ROOT_DIR / ".gitignore"
        self.assertTrue(gitignore_path.is_file(), ".gitignore must exist")
        content = gitignore_path.read_text()
        required_patterns = [
            "__pycache__/",
            ".venv/",
            "outputs/",
            "checkpoints/",
            "exports/",
            "*.gguf",
            "*Zone.Identifier",
            ".pytest_cache/",
        ]
        for pattern in required_patterns:
            self.assertIn(pattern, content, f"{pattern} should be in .gitignore")

    def test_pyproject_toml_file(self):
        pyproject_path = ROOT_DIR / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file(), "pyproject.toml must exist")
        data = tomllib.loads(pyproject_path.read_text())
        
        # Build system
        self.assertEqual(data["build-system"]["build-backend"], "hatchling.build")
        self.assertIn("hatchling", data["build-system"]["requires"])
        
        # Project metadata
        project = data["project"]
        self.assertEqual(project["name"], "slm-post-train")
        self.assertEqual(project["version"], "0.1.0")
        self.assertEqual(project["requires-python"], ">=3.10,<3.12")
        self.assertIn("torch", project["dependencies"])
        self.assertIn("unsloth", project["dependencies"])
        self.assertEqual(project["scripts"]["slm-post-train"], "slm_post_train.cli:main")

    def test_package_init(self):
        self.assertEqual(slm_post_train.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
