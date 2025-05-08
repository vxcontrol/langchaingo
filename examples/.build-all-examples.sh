#!/bin/bash
# .update-all-to-latest.sh is a small helper to update all examples to point to the latest langchaingo release
#
export GOPROXY=direct
export GOWORK=off

mkdir -p ../.build

for gm in $(find . -name go.mod); do
  (
    cd $(dirname $gm)
    go build -o ../../.build/$(basename $(dirname $gm)) ./...
) &
done
wait
