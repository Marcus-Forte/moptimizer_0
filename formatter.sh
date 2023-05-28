#! /bin/bash

echo "Formatting code..."
find . -iname *.h -o -iname *.cpp -o -iname *.md | xargs clang-format -i
echo "Done!"