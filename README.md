# diff-refactor
When you need to diff a large refactor based on moving, merging, splitting or abstracting code


# Usage

Run a git diff and save it into a file, e.g.

`git diff > changes.diff`

Then call the python code with the file by

`python diff-refactor.py < changes.diff`

