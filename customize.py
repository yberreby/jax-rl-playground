#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
# ]
# ///

import click
import subprocess
from pathlib import Path
import datetime

# Constants
TEMPLATE_NAME = "jax-rl-playground"
TEMPLATE_AUTHOR = "Yohaï-Eliel Berreby"
TEMPLATE_GITHUB = "yberreby"
CURRENT_YEAR = datetime.datetime.now().year


@click.command()
@click.option("--name", prompt="Project name", help="Your project name")
@click.option("--author", prompt="Author name", help="Your full name")
def customize(name, author):
    """Customize the template with your project information."""

    # Find and replace in relevant files
    file_patterns = ["*.md", "*.toml", "*.yml", "*.yaml"]

    # Replace project name
    for pattern in file_patterns:
        subprocess.run(
            [
                "find",
                ".",
                "-type",
                "f",
                "-name",
                pattern,
                "-exec",
                "sed",
                "-i",
                f"s/{TEMPLATE_NAME}/{name}/g",
                "{}",
                "+",
            ]
        )

    # Update author in pyproject.toml
    subprocess.run(["sed", "-i", f"s/{TEMPLATE_AUTHOR}/{author}/g", "pyproject.toml"])

    # Remove GitHub username replacements - user will use git remote set-url

    # Update LICENSE with proper attribution
    license_path = Path("LICENSE")
    if license_path.exists():
        original_license = license_path.read_text()
        new_license = f"MIT License\n\nCopyright (c) {CURRENT_YEAR} {author}\n\nOriginal template Copyright (c) {CURRENT_YEAR} {TEMPLATE_AUTHOR}\n"
        # Keep rest of MIT text after the copyright lines
        license_lines = original_license.split("\n")
        if len(license_lines) > 3:
            new_license += "\n".join(license_lines[3:])
        license_path.write_text(new_license)

    print(f"\n✅ Customized to '{name}' by {author}")
    print("\n📝 Next steps:")
    print("1. Review the changes")
    print("2. Set your remote repository:")
    print("   git remote set-url origin <your-repo-url>")
    print("   # Or if creating new repo:")
    print("   gh repo create <repo-name> --public --source=. --remote=origin")
    print("3. Commit: git add -A && git commit -m 'Customize template'")
    print("4. Push: git push -u origin master")


if __name__ == "__main__":
    customize()
