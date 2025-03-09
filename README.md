# diff-refactor
When you need to diff a large refactor based on moving, merging, splitting or abstracting code

# Requirements

- Python 3.9+
- Python 3.7+ with some manual imports.

# Usage

Run a git diff and save it into a file, e.g.

`git diff > changes.diff`

Then call the python code with the file by

`python diff-refactor.py < changes.diff`


# Concepts

- An element: this is the thing to be compared. It can be a character, a line, a file, etc. In this tool, the default (and the only implemented) is a line.
- A block: a sequence of contiguous matched elements. In this case, a block of lines.
- A match: two elements are matched if they are 'equal', however that is defined. In this case, two lines are equal if all the characters are equal, except for any whitespace at the beginning or end of the line.



The diff-refactor tool extends the basic diff concepts for added and deleted elements to include element merges, splits and moves.

## Layer 1: Adds and deletes

These are the basic concepts that are implemented by the basic diff concepts, and is pretty much used by the original diff tool, and also by git diff in its default mode.

![Level 1 diff concepts](Diff_concepts_level_1.drawio.svg)
An open circle means that on this position, this block of code wasn't there. A filled green circle means that the block of code is there afterwards. A filled red circle means that the block of code is not there afterwards.

## Layer 2: Moves

If we look at two elements being compared, we can see that - for all elements that are matched - there are three distinct possibilities:

![Level 2 diff concepts](Diff_concepts_level_2A.drawio.svg)

In this image, the circles represent a block of code. All the filled circles are for the code content. So all the filled circles match.

Note: we disregard the order of the elements. We also remove all possibilities where one of elements on a position stays the same.

However, I felt that this was not the most human representation of the concept. So I instead look at it like this:

![Level 2 diff concepts - revised](Diff_concepts_level_2B.drawio.svg)

Notice how I changed the +added and -removed into a "moved" operation.

## Layer 3: Splits and merges
If we have three position in the codebase that have multiple copied of the same block of code, we can consider the following possibilities:\


If we look at two elements being compared, we can see that - for all elements that are matched - there are three distinct possibilities:

![Level 3 diff concepts](Diff_concepts_level_3A.drawio.svg)

Here again, we have a block of code that is either created, but the middle two, I have an easier time visualizing it as this:

![Level 3 diff concepts revised](Diff_concepts_level_3B.drawio.svg)

Most saliently: we can see that instead of 1 add and 2 deletes, we have a "merge" of the block of code. This is the one I would have really liked in my git diffs up until now, and is basically the reason why I created this tool.

Additionally, we can see that instead of 2 adds and one delete, we have a "split" of the block of code. I don't know why this could be useful. Please let me know if you encounter a good use case.


## Other possibilities

 There are other possibilities, but they are not included or implemented, as I'm not sure on an elegant way to handle them or what they actually represent. 

The different possibilities I excluded are: 
- Blocks of codes that were unchanged, but still match the blocks of code that are added or removed.
- The order of the blocks of code changes.
