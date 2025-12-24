"""
PPT Maker Tool Implementation
Creates professional presentations from natural language using Marp
"""
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import config


def log_to_prompts_file(message: str):
    """Write message to prompts.log"""
    try:
        with open(config.PROMPTS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to prompts.log: {e}")


class PPTMakerTool:
    """
    PPT Maker Tool - Create presentations from natural language

    Flow:
    1. User provides natural language instruction
    2. LLM generates Marp markdown
    3. Validate markdown structure
    4. Export to PDF using Marp CLI
    5. Export to PPTX using Marp CLI
    6. Return file paths and metadata
    """

    def __init__(self, session_id: str):
        """
        Initialize PPT Maker tool

        Args:
            session_id: Session ID for workspace isolation
        """
        self.session_id = session_id
        self.workspace = config.PPT_MAKER_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        print(f"[PPT_MAKER] Initialized for session: {session_id}")
        print(f"[PPT_MAKER] Workspace: {self.workspace}")

    def create_presentation(
        self,
        instruction: str,
        theme: Optional[str] = None,
        footer: Optional[str] = None,
        header: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create presentation from natural language instruction

        Args:
            instruction: Natural language description of presentation
            theme: Marp theme ("default", "gaia", "uncover")
            footer: Footer text for all slides
            header: Header text for all slides
            timeout: Execution timeout for Marp CLI

        Returns:
            Dictionary with success, files, and metadata

        Raises:
            ValueError: Invalid input or markdown validation failed
            RuntimeError: Marp CLI execution failed
        """
        start_time = time.time()

        # Validate theme
        selected_theme = theme or config.PPT_MAKER_DEFAULT_THEME
        if selected_theme not in config.PPT_MAKER_THEMES:
            raise ValueError(
                f"Invalid theme '{selected_theme}'. "
                f"Available: {', '.join(config.PPT_MAKER_THEMES)}"
            )

        # Log execution start
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"TOOL EXECUTION: ppt_maker")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Instruction Length: {len(instruction)} chars")
        log_to_prompts_file(f"Theme: {selected_theme}")
        log_to_prompts_file(f"")
        log_to_prompts_file(f"INSTRUCTION:")
        for line in instruction.split('\n'):
            log_to_prompts_file(f"  {line}")

        print("\n" + "=" * 80)
        print("[PPT_MAKER] create_presentation() called")
        print("=" * 80)
        print(f"Instruction: {instruction[:100]}...")
        print(f"Theme: {selected_theme}")
        print(f"Footer: {footer or '(none)'}")
        print(f"Header: {header or '(none)'}")

        try:
            # Step 1: Generate Marp markdown using LLM
            print("\n[PPT_MAKER] Step 1: Generating Marp markdown...")
            log_to_prompts_file(f"")
            log_to_prompts_file(f"STEP 1: GENERATING MARKDOWN")

            markdown = self._generate_markdown(
                instruction=instruction,
                theme=selected_theme,
                footer=footer,
                header=header
            )

            print(f"[PPT_MAKER] [OK] Generated {len(markdown)} chars of markdown")
            log_to_prompts_file(f"  Generated: {len(markdown)} chars")

            # Step 2: Validate markdown structure
            print("\n[PPT_MAKER] Step 2: Validating markdown...")
            log_to_prompts_file(f"")
            log_to_prompts_file(f"STEP 2: VALIDATION")

            num_slides = self._validate_markdown(markdown)

            print(f"[PPT_MAKER] [OK] Validation passed ({num_slides} slides)")
            log_to_prompts_file(f"  Status: PASSED")
            log_to_prompts_file(f"  Slides detected: {num_slides}")

            # Step 3: Save markdown to workspace
            print("\n[PPT_MAKER] Step 3: Saving markdown file...")
            md_file = self._save_markdown(markdown)
            print(f"[PPT_MAKER] [OK] Saved to {md_file.name}")

            # Step 4: Export to PDF
            if config.PPT_MAKER_EXPORT_PDF:
                print("\n[PPT_MAKER] Step 4: Exporting to PDF...")
                log_to_prompts_file(f"")
                log_to_prompts_file(f"STEP 3: EXPORTING PDF")

                pdf_file = self._export_pdf(md_file, timeout)
                pdf_size = pdf_file.stat().st_size

                print(f"[PPT_MAKER] [OK] PDF created: {pdf_file.name} ({pdf_size / 1024:.1f} KB)")
                log_to_prompts_file(f"  File: {pdf_file.name}")
                log_to_prompts_file(f"  Size: {pdf_size / 1024:.1f} KB")
            else:
                pdf_file = None
                print("[PPT_MAKER] PDF export disabled")

            # Step 5: Export to PPTX
            if config.PPT_MAKER_EXPORT_PPTX:
                print("\n[PPT_MAKER] Step 5: Exporting to PPTX...")
                log_to_prompts_file(f"")
                log_to_prompts_file(f"STEP 4: EXPORTING PPTX")

                pptx_file = self._export_pptx(md_file, timeout)
                pptx_size = pptx_file.stat().st_size

                print(f"[PPT_MAKER] [OK] PPTX created: {pptx_file.name} ({pptx_size / 1024:.1f} KB)")
                log_to_prompts_file(f"  File: {pptx_file.name}")
                log_to_prompts_file(f"  Size: {pptx_size / 1024:.1f} KB")
            else:
                pptx_file = None
                print("[PPT_MAKER] PPTX export disabled")

            # Step 6: Collect results
            execution_time = time.time() - start_time

            result = {
                "success": True,
                "markdown": markdown,
                "markdown_file": str(md_file),
                "pdf_file": str(pdf_file) if pdf_file else None,
                "pptx_file": str(pptx_file) if pptx_file else None,
                "num_slides": num_slides,
                "workspace": str(self.workspace),
                "files": self._get_workspace_files(),
                "execution_time": execution_time
            }

            # Log success
            log_to_prompts_file(f"")
            log_to_prompts_file(f"OUTPUT:")
            log_to_prompts_file(f"  Status: SUCCESS")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"  Slides: {num_slides}")
            log_to_prompts_file(f"  Files: {md_file.name}, {pdf_file.name if pdf_file else 'N/A'}, {pptx_file.name if pptx_file else 'N/A'}")
            log_to_prompts_file("=" * 80)

            print(f"\n[PPT_MAKER] Execution completed in {execution_time:.2f}s")
            print(f"[PPT_MAKER] Success: {num_slides} slides generated")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            print(f"\n[PPT_MAKER] ERROR: {error_msg}")

            log_to_prompts_file(f"")
            log_to_prompts_file(f"ERROR: {type(e).__name__}")
            log_to_prompts_file(f"  {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file("=" * 80)

            raise

    def _generate_markdown(
        self,
        instruction: str,
        theme: str,
        footer: Optional[str],
        header: Optional[str]
    ) -> str:
        """
        Generate Marp markdown using LLM

        Args:
            instruction: User's natural language instruction
            theme: Marp theme to use
            footer: Footer text
            header: Header text

        Returns:
            Generated Marp markdown

        Raises:
            Exception: LLM call failed
        """
        from backend.core.llm_backend import llm_backend

        # Load prompt template
        prompt = self._load_prompt(
            "ppt_maker_generate.txt",
            instruction=instruction,
            theme=theme,
            paginate="true" if config.PPT_MAKER_PAGINATE else "false",
            footer_directive=f"footer: '{footer}'" if footer else "",
            header_directive=f"header: '{header}'" if header else "",
            max_slides=config.PPT_MAKER_MAX_SLIDES
        )

        log_to_prompts_file(f"  Model: {config.PPT_MAKER_MODEL}")
        log_to_prompts_file(f"  Temperature: {config.PPT_MAKER_TEMPERATURE}")
        log_to_prompts_file(f"  Prompt length: {len(prompt)} chars")

        print(f"[PPT_MAKER] Calling LLM ({config.PPT_MAKER_MODEL})...")

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        markdown = llm_backend.chat(
            messages,
            config.PPT_MAKER_MODEL,
            config.PPT_MAKER_TEMPERATURE
        )

        # Clean up markdown (remove potential code fences)
        markdown = markdown.strip()
        if markdown.startswith("```markdown"):
            markdown = markdown[len("```markdown"):].strip()
        if markdown.startswith("```"):
            markdown = markdown[3:].strip()
        if markdown.endswith("```"):
            markdown = markdown[:-3].strip()

        # Log generated markdown
        log_to_prompts_file(f"")
        log_to_prompts_file(f"GENERATED MARKDOWN:")
        for line in markdown.split('\n')[:50]:  # First 50 lines
            log_to_prompts_file(f"  {line}")
        if markdown.count('\n') > 50:
            log_to_prompts_file(f"  ... ({markdown.count('\n') - 50} more lines)")

        return markdown

    def _validate_markdown(self, markdown: str) -> int:
        """
        Validate Marp markdown structure (strict, no fallback)

        Args:
            markdown: Generated markdown content

        Returns:
            Number of slides detected

        Raises:
            ValueError: Validation failed
        """
        # Check front matter exists
        if not markdown.startswith("---"):
            raise ValueError(
                "Invalid Marp markdown: missing front matter. "
                "Markdown must start with '---'"
            )

        # Check marp: true directive
        if "marp: true" not in markdown:
            raise ValueError(
                "Invalid Marp markdown: missing 'marp: true' directive in front matter"
            )

        # Count slide separators
        # Front matter ends with ---, then slides are separated by ---
        parts = markdown.split('\n---\n')

        # First part is front matter, rest are slides
        if len(parts) < 2:
            raise ValueError(
                "Invalid Marp markdown: no slide separators found. "
                "Use '---' on a new line to separate slides"
            )

        # Number of slides = parts - 1 (excluding front matter)
        num_slides = len(parts) - 1

        # Check slide count limits
        if num_slides > config.PPT_MAKER_MAX_SLIDES:
            raise ValueError(
                f"Too many slides: {num_slides} (max: {config.PPT_MAKER_MAX_SLIDES})"
            )

        if num_slides < 1:
            raise ValueError("No slides found in presentation")

        return num_slides

    def _save_markdown(self, markdown: str) -> Path:
        """
        Save markdown to workspace

        Args:
            markdown: Markdown content

        Returns:
            Path to saved markdown file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = self.workspace / f"presentation_{timestamp}.md"

        # Write to file
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        return md_file

    def _export_pdf(self, md_file: Path, timeout: Optional[int]) -> Path:
        """
        Export markdown to PDF using Marp CLI (strict, no fallback)

        Args:
            md_file: Path to markdown file
            timeout: Execution timeout

        Returns:
            Path to generated PDF file

        Raises:
            RuntimeError: Marp CLI execution failed
        """
        pdf_file = md_file.with_suffix('.pdf')

        # Build command
        cmd_parts = config.PPT_MAKER_MARP_CLI.split()
        cmd = [
            *cmd_parts,
            str(md_file),
            "--pdf",
            "-o", str(pdf_file)
        ]

        if config.PPT_MAKER_ALLOW_LOCAL_FILES:
            cmd.insert(-3, "--allow-local-files")

        print(f"[PPT_MAKER] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or config.PPT_MAKER_TIMEOUT,
                cwd=str(self.workspace)
            )

            if result.returncode != 0:
                error_detail = result.stderr.strip() if result.stderr else result.stdout.strip()
                raise RuntimeError(
                    f"Marp PDF export failed (exit code {result.returncode}): {error_detail}"
                )

            if not pdf_file.exists():
                raise RuntimeError(
                    "PDF file was not created. Marp CLI ran but produced no output."
                )

            return pdf_file

        except FileNotFoundError:
            raise RuntimeError(
                "Marp CLI not found. Install with: npm install -g @marp-team/marp-cli\n"
                "Or ensure Node.js is installed and npx is available."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Marp PDF export timeout ({timeout or config.PPT_MAKER_TIMEOUT}s). "
                "Presentation may be too complex or system is slow."
            )

    def _export_pptx(self, md_file: Path, timeout: Optional[int]) -> Path:
        """
        Export markdown to PPTX using Marp CLI (strict, no fallback)

        Args:
            md_file: Path to markdown file
            timeout: Execution timeout

        Returns:
            Path to generated PPTX file

        Raises:
            RuntimeError: Marp CLI execution failed
        """
        pptx_file = md_file.with_suffix('.pptx')

        # Build command
        cmd_parts = config.PPT_MAKER_MARP_CLI.split()
        cmd = [
            *cmd_parts,
            str(md_file),
            "--pptx",
            "-o", str(pptx_file)
        ]

        if config.PPT_MAKER_ALLOW_LOCAL_FILES:
            cmd.insert(-3, "--allow-local-files")

        print(f"[PPT_MAKER] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or config.PPT_MAKER_TIMEOUT,
                cwd=str(self.workspace)
            )

            if result.returncode != 0:
                error_detail = result.stderr.strip() if result.stderr else result.stdout.strip()
                raise RuntimeError(
                    f"Marp PPTX export failed (exit code {result.returncode}): {error_detail}"
                )

            if not pptx_file.exists():
                raise RuntimeError(
                    "PPTX file was not created. Marp CLI ran but produced no output."
                )

            return pptx_file

        except FileNotFoundError:
            raise RuntimeError(
                "Marp CLI not found. Install with: npm install -g @marp-team/marp-cli\n"
                "Or ensure Node.js is installed and npx is available."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Marp PPTX export timeout ({timeout or config.PPT_MAKER_TIMEOUT}s). "
                "Presentation may be too complex or system is slow."
            )

    def _load_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load and format prompt template

        Args:
            prompt_name: Prompt filename
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        prompt_path = config.PROMPTS_DIR / "tools" / prompt_name

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        return template.format(**kwargs)

    def _get_workspace_files(self) -> Dict[str, Any]:
        """
        Get list of files in workspace

        Returns:
            Dictionary of files with metadata
        """
        files = {}

        try:
            for file_path in self.workspace.iterdir():
                if file_path.is_file():
                    files[file_path.name] = {
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "path": str(file_path)
                    }
        except Exception as e:
            print(f"[PPT_MAKER] Warning: Failed to list files: {e}")

        return files

    def _count_slides(self, markdown: str) -> int:
        """
        Count number of slides in markdown

        Args:
            markdown: Marp markdown

        Returns:
            Number of slides
        """
        parts = markdown.split('\n---\n')
        return len(parts) - 1  # Exclude front matter

    def list_files(self) -> list:
        """
        List all files in workspace

        Returns:
            List of file names
        """
        try:
            return [f.name for f in self.workspace.iterdir() if f.is_file()]
        except Exception as e:
            print(f"[PPT_MAKER] Warning: Failed to list files: {e}")
            return []

    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from workspace

        Args:
            filename: File name

        Returns:
            File contents or None if not found
        """
        file_path = self.workspace / filename

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[PPT_MAKER] Warning: Failed to read '{filename}': {e}")
            return None
