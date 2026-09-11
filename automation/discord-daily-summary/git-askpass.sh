#!/bin/sh
case "$1" in
    *Username*) printf '%s\n' 'x-access-token' ;;
    *Password*) /usr/bin/cat "$GITHUB_TOKEN_FILE" ;;
    *) exit 1 ;;
esac
