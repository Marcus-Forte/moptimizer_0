#! /bin/bash

echo "Formatting code..."
find src tst include \( -iname *.h -o -iname *.cpp -o -iname *.md \) -and -not -iname *duna_exports.h \
| xargs clang-format -i --verbose
