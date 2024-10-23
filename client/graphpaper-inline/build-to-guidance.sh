#!/bin/bash
set -x

npm run build
cp dist/index.html ../../guidance/resources/graphpaper-inline.html