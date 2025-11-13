#!/bin/bash
# adapted from https://github.com/jlumbroso/free-disk-space/blob/main/action.yml

set -x

echo "Remove Android library"
sudo rm -rf /usr/local/lib/android || true
echo "Remove .NET runtime"
sudo rm -rf /usr/share/dotnet || true
