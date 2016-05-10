import sys
import re

from_file = sys.argv[1]
to_file = sys.argv[2]
to_file_out = open(to_file, "w")

regex = "<.*>"

tag_match = re.compile(regex)
matched_lines = []

with open(from_file) as from_file:
    content = from_file.readlines()
    for line in content:
        if (tag_match.match(line)):
            pass
        else:
            matched_lines.append(line)

matched_lines = "".join(matched_lines)
to_file_out.write(matched_lines)
to_file_out.close()

