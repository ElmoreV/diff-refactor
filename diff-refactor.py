import sys
import difflib
import re
from dataclasses import dataclass
from enum import Enum

# --- Configuration ---

# Minimal block length to trigger block header/footer.
GROUP_THRESHOLD = 6
MIN_BLOCK_SIZE = 4

##################
### Parsing #####
#################


class FileDiffStatus(Enum):
    UNKNOWN = "unknown"
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"


# type alias
Entry = tuple[int, str]


@dataclass
class ParsedHunkDiff:
    hunk_header: str
    lines: list[str]
    added_entries: list[Entry]  #
    removed_entries: list[Entry]


@dataclass
class ParsedFileDiff:
    header: list[str]
    file_path: str
    status: FileDiffStatus
    hunks: list[ParsedHunkDiff]


def norm_line(line):
    return line[1:].lstrip() if line and line[0] in "+- " else line.lstrip()


def parse_diff(diff_text: str) -> list[ParsedFileDiff]:
    files: list[ParsedFileDiff] = []
    current_file: str | None = None
    current_header: list[str] = []
    current_hunks: list[ParsedHunkDiff] = []
    current_hunk: ParsedHunkDiff | None = None
    status: FileDiffStatus = FileDiffStatus.MODIFIED

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # diff --git a/src/App.js b/src/App.js
            if current_file is not None:
                if current_hunk is not None:
                    current_hunks.append(current_hunk)
                    current_hunk = None
                files.append(
                    ParsedFileDiff(
                        header=current_header,
                        file_path=current_file,
                        status=status,
                        hunks=current_hunks,
                    )
                )
            current_header = [line]
            parts = line.split()
            current_file = parts[3][2:] if len(parts) >= 4 else "unknown"
            current_hunks = []
            current_hunk = None
            status = FileDiffStatus.MODIFIED
        elif line.startswith("new file mode"):
            # new file mode 100644
            current_header.append(line)
            status = FileDiffStatus.ADDED
        elif line.startswith("deleted file mode"):
            # deleted file mode 100644
            current_header.append(line)
            status = FileDiffStatus.DELETED
        elif line.startswith("@@"):
            # @@ -1,256 +0,0 @@
            if current_hunk is not None:
                current_hunks.append(current_hunk)
            current_hunk = ParsedHunkDiff(
                hunk_header=line,
                lines=[],
                added_entries=[],
                removed_entries=[],
            )
        else:
            if current_hunk is not None:
                idx = len(current_hunk.lines)
                current_hunk.lines.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    current_hunk.added_entries.append((idx, norm_line(line)))
                elif line.startswith("-") and not line.startswith("---"):
                    current_hunk.removed_entries.append((idx, norm_line(line)))
            else:
                current_header.append(line)
    if current_file is not None:
        if current_hunk is not None:
            current_hunks.append(current_hunk)
        files.append(
            ParsedFileDiff(
                header=current_header,
                file_path=current_file,
                status=status,
                hunks=current_hunks,
            )
        )
    return files


####################
### Mapping ########
####################
class Marker(Enum):
    MOVED = "moved"
    DUPLICATED = "duplicated"
    COMBINED = "combined"
    MODIFIED = "modified"


# type alias
MappingDict = dict[tuple[str, int, int], list[tuple[str, int, int]]]


def build_match_mappings(
    files: list[ParsedFileDiff],
) -> tuple[
    MappingDict,
    MappingDict,
]:
    """Returns:
    added_mapping: dict mapping (dst_file, hunk_idx, added_idx) -> list of (src_file, hunk_idx, removed_idx)
    removed_mapping: dict mapping (src_file, hunk_idx, removed_idx) -> list of (dst_file, hunk_idx, added_idx)
    """
    added_mapping: MappingDict = {}
    removed_mapping: MappingDict = {}
    n = len(files)
    for i in range(n):
        src = files[i]
        if src.status not in (FileDiffStatus.DELETED, FileDiffStatus.MODIFIED):
            # Ignore files that are added, they cannot provide a source line
            continue
        for j in range(n):
            dst = files[j]
            if dst.status not in (FileDiffStatus.ADDED, FileDiffStatus.MODIFIED):
                # Ignore files that are deleted, they cannot provide a destination line
                continue
            for hi, src_hunk in enumerate(src.hunks):
                # Find all the normed lines that are removed in this hunk
                src_lines = [entry[1] for entry in src_hunk.removed_entries]
                if len(src_lines) == 0:
                    continue
                for hj, dst_hunk in enumerate(dst.hunks):
                    # Find all the normed lines that are added in this hunk
                    dst_lines = [entry[1] for entry in dst_hunk.added_entries]
                    if len(dst_lines) == 0:
                        continue
                    matcher = difflib.SequenceMatcher(None, src_lines, dst_lines)
                    for block in matcher.get_matching_blocks():
                        if block.size >= MIN_BLOCK_SIZE:
                            for k in range(block.size):
                                src_idx = block.a + k
                                dst_idx = block.b + k
                                src_key = (src.file_path, hi, src_idx)
                                dst_key = (dst.file_path, hj, dst_idx)
                                added_mapping.setdefault(dst_key, []).append(src_key)
                                removed_mapping.setdefault(src_key, []).append(dst_key)
    return added_mapping, removed_mapping


# type alias
MarkerDict = dict[tuple[str, int, int], Marker | None]


def compute_markers_individual(
    added_mapping: MappingDict,
    removed_mapping: MappingDict,
) -> tuple[MarkerDict, MarkerDict]:
    """Assign a marker per mapped line."""
    added_markers = {}
    for key, src_list in added_mapping.items():
        if len(src_list) == 1:
            src_key = src_list[0]
            marker = "M+" if len(removed_mapping.get(src_key, [])) == 1 else "D+"
        elif len(src_list) > 1:
            marker = "S+"
        else:
            marker = None
        added_markers[key] = marker
    removed_markers = {}
    for key, dst_list in removed_mapping.items():
        if len(dst_list) == 1:
            dst_key = dst_list[0]
            marker = "M-" if len(added_mapping.get(dst_key, [])) == 1 else "S-"
        elif len(dst_list) > 1:
            marker = "S-"
        else:
            marker = None
        removed_markers[key] = marker
    return added_markers, removed_markers


####################
### Outputting #####
####################
# Colors for markers (dull for moved, bright for unmapped).
MARKER_COLORS = {
    "M+": "\033[2;32m",  # dim green
    "M-": "\033[2;31m",  # dim red
    "D+": "\033[94m",  # bright blue
    "S+": "\033[95m",  # bright magenta
    "S-": "\033[96m",  # bright cyan
}
DEFAULT_ADDED_COLOR = "\033[1;32m"  # bright green for unmapped +
DEFAULT_REMOVED_COLOR = "\033[1;31m"  # bright red for unmapped -

RESET = "\033[0m"


def marker_description(marker: str) -> Marker:
    if marker in ("M+", "M-"):
        return Marker.MOVED
    elif marker == "D+":
        return Marker.DUPLICATED
    elif marker in ("S+", "S-"):
        return Marker.COMBINED
    return Marker.MODIFIED


def parse_hunk_header(header: str) -> tuple[int, int, int, int]:
    """Extract old/new start and count from a hunk header.
    E.g., for header "@@ -12,5 +12,6 @@" return (12, 5, 12, 6)."""
    m = re.search(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", header)
    if m:
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) != "" else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) != "" else 1
        return (old_start, old_count, new_start, new_count)
    return (None, None, None, None)


def output_annotated_diff(
    files: list[ParsedFileDiff],
    added_markers: MarkerDict,
    removed_markers: MarkerDict,
    file_dict: dict[str, ParsedFileDiff],
    added_mapping: MappingDict,
    removed_mapping: MappingDict,
):
    out_lines = []
    for f in files:
        for header_line in f.header:
            out_lines.append(header_line)
        for hi, hunk in enumerate(f.hunks):
            out_lines.append(hunk.hunk_header)
            # Parse destination hunk header for new start line.
            dst_info = parse_hunk_header(hunk.hunk_header)
            dst_new_start = dst_info[2] if dst_info[2] is not None else 0

            i = 0
            added_counter = 0
            removed_counter = 0
            while i < len(hunk.lines):
                line = hunk.lines[i]
                # Process added lines
                if line.startswith("+") and not line.startswith("+++"):
                    key = (f.file_path, hi, added_counter)
                    if key in added_markers:
                        group_lines = []
                        group_keys = []
                        # Group contiguous mapped added lines.
                        while (
                            i < len(hunk.lines)
                            and hunk.lines[i].startswith("+")
                            and not hunk.lines[i].startswith("+++")
                            and ((f.file_path, hi, added_counter) in added_markers)
                        ):
                            group_lines.append(hunk.lines[i])
                            group_keys.append((f.file_path, hi, added_counter))
                            i += 1
                            added_counter += 1
                        if len(group_lines) >= GROUP_THRESHOLD:
                            marker = added_markers[group_keys[0]]
                            desc = marker_description(marker)
                            # Get source info from the mapping if available:
                            src_keys = added_mapping.get(group_keys[0], [])
                            if src_keys:
                                first_src = src_keys[0]
                                src_file, src_hunk_idx, src_line_idx = first_src
                                src_hunk_header = (
                                    file_dict[src_file].hunks[src_hunk_idx].hunk_header
                                )
                                src_info = parse_hunk_header(src_hunk_header)
                                src_line = (
                                    (src_info[0] + src_line_idx)
                                    if src_info[0] is not None
                                    else src_line_idx
                                )
                            else:
                                src_file, src_line = "unknown", "?"
                            dst_line = dst_new_start + group_keys[0][2]
                            header_blk = f"----- {desc} block from {src_file}:{src_line} to {f.file_path}:{dst_line} -----"
                            footer_blk = f"----- end {desc} block -----"
                            out_lines.append(header_blk)
                            for k, gline in zip(group_keys, group_lines):
                                color = MARKER_COLORS.get(added_markers[k], "")
                                out_lines.append(
                                    f"{color}{added_markers[k]}{gline[1:]}{RESET}"
                                )
                            out_lines.append(footer_blk)
                        else:
                            for k, gline in zip(group_keys, group_lines):
                                color = MARKER_COLORS.get(added_markers[k], "")
                                out_lines.append(
                                    f"{color}{added_markers[k]}{gline[1:]}{RESET}"
                                )
                    else:
                        out_lines.append(f"{DEFAULT_ADDED_COLOR}+{line[1:]}{RESET}")
                        i += 1
                        added_counter += 1

                # Process removed lines
                elif line.startswith("-") and not line.startswith("---"):
                    key = (f.file_path, hi, removed_counter)
                    if key in removed_markers:
                        group_lines = []
                        group_keys = []
                        while (
                            i < len(hunk.lines)
                            and hunk.lines[i].startswith("-")
                            and not hunk.lines[i].startswith("---")
                            and ((f.file_path, hi, removed_counter) in removed_markers)
                        ):
                            group_lines.append(hunk.lines[i])
                            group_keys.append((f.file_path, hi, removed_counter))
                            i += 1
                            removed_counter += 1
                        if len(group_lines) >= GROUP_THRESHOLD:
                            marker = removed_markers[group_keys[0]]
                            desc = marker_description(marker)
                            dst_keys = removed_mapping.get(group_keys[0], [])
                            if dst_keys:
                                first_dst = dst_keys[0]
                                dst_file, dst_hunk_idx, dst_line_idx = first_dst
                                dst_hunk_header = (
                                    file_dict[dst_file].hunks[dst_hunk_idx].hunk_header
                                )
                                dst_info = parse_hunk_header(dst_hunk_header)
                                dst_line = (
                                    (dst_info[2] + dst_line_idx)
                                    if dst_info[2] is not None
                                    else dst_line_idx
                                )
                            else:
                                dst_file, dst_line = "unknown", "?"
                            src_info = parse_hunk_header(hunk.hunk_header)
                            src_line = (
                                (src_info[0] + group_keys[0][2])
                                if src_info[0] is not None
                                else group_keys[0][2]
                            )
                            header_blk = f"----- {desc} block from {f.file_path}:{src_line} to {dst_file}:{dst_line} -----"
                            footer_blk = f"----- end {desc} block -----"
                            out_lines.append(header_blk)
                            for k, gline in zip(group_keys, group_lines):
                                color = MARKER_COLORS.get(removed_markers[k], "")
                                out_lines.append(
                                    f"{color}{removed_markers[k]}{gline[1:]}{RESET}"
                                )
                            out_lines.append(footer_blk)
                        else:
                            for k, gline in zip(group_keys, group_lines):
                                color = MARKER_COLORS.get(removed_markers[k], "")
                                out_lines.append(
                                    f"{color}{removed_markers[k]}{gline[1:]}{RESET}"
                                )
                    else:
                        out_lines.append(f"{DEFAULT_REMOVED_COLOR}-{line[1:]}{RESET}")
                        i += 1
                        removed_counter += 1
                else:
                    out_lines.append(line)
                    i += 1
            out_lines.append("")
    return "\n".join(out_lines)


####################
### Main ###########
####################


def main():
    diff_text = sys.stdin.read()
    files = parse_diff(diff_text)
    for f in files:
        print(f"File: {f.file_path}")
        print(f"Status: {f.status}")
        for hunk in f.hunks:
            print(f"Hunk: {hunk.added_entries}")
    file_dict = {f.file_path: f for f in files}
    added_mapping, removed_mapping = build_match_mappings(files)
    added_markers, removed_markers = compute_markers_individual(
        added_mapping, removed_mapping
    )
    annotated = output_annotated_diff(
        files, added_markers, removed_markers, file_dict, added_mapping, removed_mapping
    )
    print(annotated)


if __name__ == "__main__":
    main()
