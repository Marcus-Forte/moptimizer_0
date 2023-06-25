#! /bin/bash

echo "Formatting code..."
find . \( -iname *.h -o -iname *.cpp -o -iname *.md \) -and -not -iname *duna_exports.h | xargs clang-format-16 -i
