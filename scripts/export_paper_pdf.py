#!/usr/bin/env python3
"""Export the paper manuscript to an MNRAS-style PDF without changing scientific content."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="paper/manuscript.md")
    parser.add_argument("--bibliography", default="paper/references.bib")
    parser.add_argument("--output", default="paper/manuscript.pdf")
    return parser.parse_args()


def require_pandoc() -> str:
    pandoc_path = shutil.which("pandoc")
    if pandoc_path:
        return pandoc_path
    homebrew_pandoc = Path("/opt/homebrew/bin/pandoc")
    if homebrew_pandoc.exists():
        return str(homebrew_pandoc)
    raise SystemExit(
        "Missing dependency: pandoc. "
        "Install it with `brew install pandoc`."
    )


def require_latexmk() -> str:
    latexmk_path = shutil.which("latexmk")
    if latexmk_path:
        return latexmk_path
    homebrew_latexmk = Path("/opt/homebrew/bin/latexmk")
    if homebrew_latexmk.exists():
        return str(homebrew_latexmk)
    raise SystemExit(
        "Missing dependency: latexmk. "
        "Install it with `brew install --cask mactex-no-gui` or a TeX Live package that provides latexmk."
    )


def require_mnras_class() -> None:
    result = subprocess.run(
        ["kpsewhich", "mnras.cls"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(
            "Missing dependency: mnras.cls. Install the MNRAS LaTeX class through TeX Live/MacTeX."
        )


def parse_frontmatter(markdown_text: str) -> dict:
    if not markdown_text.startswith("---\n"):
        return {}
    match = re.match(r"^---\n(.*?)\n---\n", markdown_text, re.DOTALL)
    if not match:
        raise ValueError("Frontmatter starts with '---' but does not terminate correctly.")
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Expected manuscript frontmatter to parse as a mapping.")
    return metadata


def extract_title(markdown_text: str) -> tuple[str, str]:
    lines = markdown_text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            body = "\n".join(lines[index + 1 :]).lstrip()
            return title, body
    raise ValueError("Expected a top-level manuscript title starting with '# '.")


def split_level2_sections(markdown_body: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"(?m)^## (.+?)\n")
    matches = list(pattern.finditer(markdown_body))
    if not matches:
        raise ValueError("Expected level-2 sections in the manuscript body.")

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_body)
        content = markdown_body[start:end].strip()
        sections.append((title, content))
    return sections


def downgrade_headings(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        hashes = match.group(1)
        label = match.group(2)
        return f"{'#' * (len(hashes) - 1)} {label}"

    return re.sub(r"(?m)^(#{2,6})\s+(.*)$", repl, text)


def remove_duplicate_figure_caption_paragraphs(text: str) -> str:
    cleaned = re.sub(r"(?m)^\*Figure\s+\d+.*\*\s*$", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def normalize_image_captions(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        caption = match.group(1)
        path = match.group(2)
        caption = re.sub(r"^\s*Figure\s+\d+\.\s*", "", caption)
        return f"![{caption}]({path})"

    return re.sub(r"!\[([^\]]+)\]\(([^)]+)\)", repl, text)


def absolutize_image_paths(text: str, paper_root: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        caption = match.group(1)
        path_text = match.group(2).strip()
        if re.match(r"^[a-zA-Z]+://", path_text):
            return match.group(0)
        absolute_path = (paper_root / path_text).resolve() if path_text.startswith(".") else Path(path_text).resolve()
        return f"![{caption}]({absolute_path.as_posix()})"

    return re.sub(r"!\[([^\]]+)\]\(([^)]+)\)", repl, text)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_running_title(title: str) -> str:
    if ":" in title:
        return title.split(":", 1)[0].strip()
    if len(title) <= 60:
        return title
    return title[:57].rstrip() + "..."


def build_running_author(authors: list[dict]) -> str:
    first_name = str(authors[0].get("name", "")).strip()
    surname = first_name.split()[-1] if first_name else "Author"
    return surname if len(authors) == 1 else f"{surname} et al."


def build_author_block(manuscript_metadata: dict) -> tuple[str, str]:
    authors = manuscript_metadata.get("authors") or []
    corresponding = manuscript_metadata.get("corresponding_author") or {}

    if not authors:
        raise ValueError("Expected manuscript frontmatter to include at least one author.")
    if not corresponding.get("email"):
        raise ValueError("Expected manuscript frontmatter to include corresponding author email.")

    affiliation_index: dict[str, int] = {}
    ordered_affiliations: list[str] = []
    for author in authors:
        affiliation = str(author.get("affiliation", "")).strip()
        if not affiliation:
            raise ValueError("Each author in manuscript frontmatter must have name and affiliation.")
        if affiliation not in affiliation_index:
            affiliation_index[affiliation] = len(ordered_affiliations) + 1
            ordered_affiliations.append(affiliation)

    corresponding_email = str(corresponding.get("email", "")).strip()
    corresponding_name = str(corresponding.get("name", "")).strip()
    author_chunks: list[str] = []
    for index, author in enumerate(authors):
        name = str(author.get("name", "")).strip()
        affiliation = str(author.get("affiliation", "")).strip()
        if not name or not affiliation:
            raise ValueError("Each author in manuscript frontmatter must have name and affiliation.")
        aff_index = affiliation_index[affiliation]
        footnote = ""
        if corresponding_name and name == corresponding_name and corresponding_email:
            footnote = rf"\thanks{{E-mail: {latex_escape(corresponding_email)}}}"
        author_chunks.append(f"{latex_escape(name)}$^{{{aff_index}}}$" + footnote)

    affiliation_lines = [
        rf"$^{{{idx}}}$\parbox[t]{{0.84\textwidth}}{{{latex_escape(aff)}}}"
        for idx, aff in enumerate(ordered_affiliations, start=1)
    ]
    author_block = ", ".join(author_chunks) + r"\\" + "\n" + r"\\ ".join(affiliation_lines)
    return author_block, corresponding_email


def build_keywords_block(manuscript_metadata: dict) -> str:
    keywords = manuscript_metadata.get("keywords") or []
    keywords_line = " -- ".join(latex_escape(str(keyword).strip()) for keyword in keywords if str(keyword).strip())
    if not keywords_line:
        raise ValueError("Expected manuscript frontmatter to include at least one keyword.")
    return keywords_line


def pandoc_markdown_to_latex(
    *,
    pandoc_path: str,
    markdown_text: str,
    bibliography_path: Path,
    working_dir: Path,
) -> str:
    input_path = working_dir / "pandoc_input.md"
    input_path.write_text(markdown_text, encoding="utf-8")
    result = subprocess.run(
        [
            pandoc_path,
            str(input_path),
            "-t",
            "latex",
            "--natbib",
            "--from",
            "markdown+tex_math_dollars+tex_math_single_backslash+implicit_figures+citations",
            "--bibliography",
            str(bibliography_path),
            "--resource-path",
            f"{ROOT / 'paper'}:{ROOT}",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return result.stdout


def _find_matching_brace(text: str, start_index: int) -> int:
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Could not find a matching brace while post-processing LaTeX.")


def convert_longtables_for_mnras(latex_text: str) -> str:
    marker = "{\\def\\LTcaptype{none} % do not increment counter"
    output_parts: list[str] = []
    cursor = 0

    while True:
        start = latex_text.find(marker, cursor)
        if start == -1:
            output_parts.append(latex_text[cursor:])
            break

        output_parts.append(latex_text[cursor:start])
        begin_marker = "\\begin{longtable}[]{"
        begin = latex_text.find(begin_marker, start)
        if begin == -1:
            raise ValueError("Malformed Pandoc longtable block: missing longtable begin marker.")
        colspec_start = begin + len(begin_marker) - 1
        colspec_end = _find_matching_brace(latex_text, colspec_start)
        colspec = latex_text[colspec_start + 1 : colspec_end]

        head_end_marker = "\\endhead"
        head_end = latex_text.find(head_end_marker, colspec_end)
        if head_end == -1:
            raise ValueError("Malformed Pandoc longtable block: missing \\endhead.")
        header_block = latex_text[colspec_end + 1 : head_end].strip()

        lastfoot_marker = "\\endlastfoot"
        lastfoot = latex_text.find(lastfoot_marker, head_end)
        if lastfoot == -1:
            raise ValueError("Malformed Pandoc longtable block: missing \\endlastfoot.")
        body_start = lastfoot + len(lastfoot_marker)
        table_end_marker = "\\end{longtable}"
        table_end = latex_text.find(table_end_marker, body_start)
        if table_end == -1:
            raise ValueError("Malformed Pandoc longtable block: missing \\end{longtable}.")
        body_block = latex_text[body_start:table_end].strip()

        close_group = latex_text.find("}", table_end + len(table_end_marker))
        if close_group == -1:
            raise ValueError("Malformed Pandoc longtable block: missing closing group brace.")

        header_block = header_block.replace("\\endhead", "").strip()
        body_block = body_block.replace("\\bottomrule\\noalign{}", "").strip()
        replacement = dedent(
            f"""
            \\begin{{table*}}
            \\centering
            \\small
            \\setlength{{\\tabcolsep}}{{3pt}}
            \\begin{{tabular}}{{{colspec}}}
            {header_block}
            {body_block}
            \\bottomrule
            \\end{{tabular}}
            \\end{{table*}}
            """
        ).strip()
        output_parts.append(replacement)
        cursor = close_group + 1

    return "".join(output_parts)


def protect_headings_for_layout(latex_text: str) -> str:
    latex_text = re.sub(
        r"(?m)^(\\section\*?\{)",
        r"\\FloatBarrier\n\\Needspace{8\\baselineskip}\n\1",
        latex_text,
    )
    latex_text = re.sub(
        r"(?m)^(\\subsection\*?\{)",
        r"\\Needspace{5\\baselineskip}\n\1",
        latex_text,
    )
    return latex_text


def promote_selected_figures_for_layout(latex_text: str) -> str:
    figure_specs = [
        ("fig01_emc_schematic.png", "0.96\\textwidth", "t"),
    ]

    for image_name, width, placement in figure_specs:
        pattern = re.compile(
            rf"\\begin\{{figure\}}\s*\\centering\s*"
            rf"\\pandocbounded\{{\\includegraphics\[keepaspectratio,alt=\{{.*?\}}\]\{{[^}}]*{re.escape(image_name)}\}}\}}\s*"
            rf"\\caption\{{(.*?)\}}\s*\\end\{{figure\}}",
            re.S,
        )

        def repl(match: re.Match[str]) -> str:
            caption = match.group(1).strip()
            image_path_match = re.search(rf"\{{([^}}]*{re.escape(image_name)})\}}", match.group(0))
            if not image_path_match:
                return match.group(0)
            image_path = image_path_match.group(1)
            return dedent(
                f"""
                \\begin{{figure*}}[{placement}]
                \\centering
                \\includegraphics[width={width},keepaspectratio]{{{image_path}}}
                \\caption{{{caption}}}
                \\end{{figure*}}
                """
            ).strip()

        latex_text = pattern.sub(repl, latex_text, count=1)

    return latex_text


def build_mnras_wrapper(
    *,
    title: str,
    abstract_tex: str,
    body_tex: str,
    bibliography_stem: str,
    manuscript_metadata: dict,
) -> str:
    author_block, _ = build_author_block(manuscript_metadata)
    running_title = latex_escape(build_running_title(title))
    running_author = latex_escape(build_running_author(manuscript_metadata.get("authors") or []))
    keywords_block = build_keywords_block(manuscript_metadata)
    long_title = latex_escape(title)
    pubyear = "2026"
    return dedent(
        f"""
        \\documentclass[fleqn,usenatbib]{{mnras}}
        \\usepackage[T1]{{fontenc}}
        \\usepackage{{graphicx}}
        \\usepackage{{amsmath,amssymb}}
        \\usepackage{{booktabs}}
        \\usepackage{{longtable}}
        \\usepackage{{array}}
        \\usepackage{{calc}}
        \\usepackage{{float}}
        \\usepackage[section]{{placeins}}
        \\usepackage{{needspace}}
        \\usepackage{{microtype}}
        \\usepackage{{hyperref}}
        \\hypersetup{{hidelinks}}
        \\providecommand{{\\tightlist}}{{%
          \\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{0pt}}}}
        \\makeatletter
        \\def\\maxwidth{{\\ifdim\\Gin@nat@width>\\linewidth\\linewidth\\else\\Gin@nat@width\\fi}}
        \\def\\maxheight{{\\ifdim\\Gin@nat@height>\\textheight\\textheight\\else\\Gin@nat@height\\fi}}
        \\makeatother
        \\setkeys{{Gin}}{{width=\\maxwidth,height=\\maxheight,keepaspectratio}}
        \\widowpenalty=10000
        \\clubpenalty=10000
        \\displaywidowpenalty=10000
        \\newcommand{{\\pandocbounded}}[1]{{#1}}
        \\title[{running_title}]{{{long_title}}}
        \\author[{running_author}]{{{author_block}}}
        \\date{{Accepted XXX. Received YYY; in original form ZZZ}}
        \\pubyear{{{pubyear}}}
        \\begin{{document}}
        \\label{{firstpage}}
        \\pagerange{{\\pageref{{firstpage}}--\\pageref{{lastpage}}}}
        \\maketitle
        \\begin{{abstract}}
        {abstract_tex.strip()}
        \\end{{abstract}}
        \\begin{{keywords}}
        {keywords_block}
        \\end{{keywords}}

        {body_tex.strip()}

        \\bibliographystyle{{mnras}}
        \\bibliography{{{latex_escape(bibliography_stem)}}}
        \\label{{lastpage}}
        \\end{{document}}
        """
    ).strip() + "\n"


def main() -> int:
    args = parse_args()
    pandoc_path = require_pandoc()
    latexmk_path = require_latexmk()
    require_mnras_class()

    input_path = (ROOT / args.input).resolve()
    bibliography_path = (ROOT / args.bibliography).resolve()
    output_path = (ROOT / args.output).resolve()
    paper_root = (ROOT / "paper").resolve()

    manuscript_text = input_path.read_text(encoding="utf-8")
    manuscript_metadata = parse_frontmatter(manuscript_text)
    title, body = extract_title(manuscript_text)
    sections = split_level2_sections(body)
    if not sections or sections[0][0] != "Abstract":
        raise ValueError("Expected the first level-2 section to be 'Abstract'.")

    abstract_text = sections[0][1]
    remaining_sections = sections[1:]

    rebuilt_sections: list[str] = []
    for section_title, section_body in remaining_sections:
        if section_title in {"Acknowledgements", "Data availability", "Data Availability"}:
            rebuilt_sections.append(f"# {section_title} {{-}}\n\n{section_body}")
        else:
            rebuilt_sections.append(f"# {section_title}\n\n{section_body}")
    export_body = "\n\n".join(rebuilt_sections)
    export_body = downgrade_headings(export_body)
    export_body = remove_duplicate_figure_caption_paragraphs(export_body)
    export_body = normalize_image_captions(export_body)
    export_body = absolutize_image_paths(export_body, paper_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dynadiff-paper-export-") as temp_dir:
        temp_root = Path(temp_dir)
        temp_bibliography = temp_root / bibliography_path.name
        abstract_tex = pandoc_markdown_to_latex(
            pandoc_path=pandoc_path,
            markdown_text=abstract_text,
            bibliography_path=bibliography_path,
            working_dir=temp_root,
        )
        body_tex = pandoc_markdown_to_latex(
            pandoc_path=pandoc_path,
            markdown_text=export_body,
            bibliography_path=bibliography_path,
            working_dir=temp_root,
        )
        body_tex = convert_longtables_for_mnras(body_tex)
        body_tex = protect_headings_for_layout(body_tex)
        body_tex = promote_selected_figures_for_layout(body_tex)
        temp_bibliography.write_text(bibliography_path.read_text(encoding="utf-8"), encoding="utf-8")
        temp_tex = temp_root / "manuscript_mnras.tex"
        temp_tex.write_text(
            build_mnras_wrapper(
                title=title,
                abstract_tex=abstract_tex,
                body_tex=body_tex,
                bibliography_stem=bibliography_path.stem,
                manuscript_metadata=manuscript_metadata,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                latexmk_path,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                str(temp_tex.name),
            ],
            cwd=str(temp_root),
            check=True,
        )
        shutil.copyfile(temp_root / "manuscript_mnras.pdf", output_path)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
