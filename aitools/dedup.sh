#!/bin/bash
# dedup.sh — Remove duplicate top-level function definitions in a Python file,
# keeping only the *last* definition of each function name.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <python-file>"
    exit 1
fi

file="$1"

if [[ ! -f "$file" ]]; then
    echo "Error: File '$file' not found"
    exit 1
fi

# Make a safety backup
cp -p "$file" "$file.bak"

# Create a temporary file
tmpfile=$(mktemp)

# Extract all top-level function names and their line numbers (no leading whitespace)
declare -A last_func_line
while IFS=':' read -r line_num rest; do
    func_name=$(echo "$rest" | sed 's/^[[:space:]]*def[[:space:]]\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/')
    last_func_line["$func_name"]=$line_num
done < <(grep -nE '^[[:space:]]*def[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file")

# Build a comma-separated string of line numbers to keep
keep_lines=""
for func in "${!last_func_line[@]}"; do
    keep_lines="${keep_lines}${last_func_line[$func]},"
done
keep_lines="${keep_lines%,}"

# Use awk to process the file
awk -v keep="$keep_lines" '
BEGIN {
    split(keep, keep_arr, ",")
    for (i in keep_arr) {
        keep_map[keep_arr[i]] = 1
    }
    in_skip_function = 0
}
{
    line_num = NR

    # Check if this is a top-level function definition
    if ($0 ~ /^[[:space:]]*def[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
        if (line_num in keep_map) {
            # This is the last occurrence - keep it
            in_skip_function = 0
            print
        } else {
            # This is a duplicate - skip this function
            in_skip_function = 1
        }
    }
    # Check if we should stop skipping (encountered a non-indented, non-empty, non-comment line)
    else if (in_skip_function && $0 ~ /^[^[:space:]]/ && $0 !~ /^$/ && $0 !~ /^#/) {
        # Hit a non-indented, non-empty, non-comment line - stop skipping
        in_skip_function = 0
        print
    }
    # Normal line handling
    else if (!in_skip_function) {
        print
    }
    # else: we are in a skip function and this is an indented line - skip it
}
' "$file" > "$tmpfile"

# Overwrite the original file while preserving permissions
mv -f -- "$tmpfile" "$file"

# Optional summary
echo "Deduplicated: kept ${#last_func_line[@]} unique functions in $file"
