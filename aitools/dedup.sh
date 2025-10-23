#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <python-file>"
    exit 1
fi

file="$1"

if [[ ! -f "$file" ]]; then
    echo "Error: File '$file' not found"
    exit 1
fi

# Create a temporary file
tmpfile=$(mktemp)

# Extract all function names and their line numbers
functions=$(grep -n "^def " "$file" | awk -F: '{print $1, $2}' | sed 's/def \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/')

# Build an awk script that will keep only the last occurrence of each function
awk_script='
BEGIN {
    # First pass: read all function definitions and find the last line number for each
}
'

# Read the file and track the last occurrence of each function
declare -A last_func_line
while IFS=' ' read -r line_num func_name; do
    last_func_line["$func_name"]=$line_num
done < <(grep -n "^def " "$file" | awk -F: '{print $1, $2}' | sed 's/def \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/')

# Build a string of functions to keep (only the last occurrence)
keep_lines=""
for func in "${!last_func_line[@]}"; do
    keep_lines="${keep_lines}${last_func_line[$func]},"
done
keep_lines="${keep_lines%,}"  # Remove trailing comma

# Use awk to print lines up to each function definition, then skip earlier definitions
awk -v keep="$keep_lines" '
BEGIN {
    split(keep, keep_arr, ",")
    for (i in keep_arr) {
        keep_map[keep_arr[i]] = 1
    }
    current_line = 0
    skip = 0
}
{
    current_line++
    if ($0 ~ /^def /) {
        if (!(current_line in keep_map)) {
            skip = 1
        } else {
            skip = 0
        }
    }
    if (!skip) {
        print
    }
}
' "$file" > "$tmpfile"

# Overwrite the original file
mv "$tmpfile" "$file"
