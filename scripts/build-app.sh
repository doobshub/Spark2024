#!/bin/bash

source venv/bin/activate

set -e

# Define paths for tools
BLACK_PATH="build/tools/black"
PYLINT_PATH="build/tools/pylint"
PYTEST_PATH="build/tools/pytest"

# Ensure the necessary tool directories exist
mkdir -p "$BLACK_PATH" "$PYLINT_PATH" "$PYTEST_PATH"

echo " "
echo "--------- BLACK ---------------------------"
echo " "
if ! black app test --config pyproject.toml --verbose 2>&1 | tee "$BLACK_PATH/black_output.txt"; then
    echo "Black found format issues or encountered an error."
    exit 1
fi

echo " "
echo "--------- PYLINT ---------------------------"
echo " "
if ! pylint --rcfile pyproject.toml app test --verbose 2>&1 | tee "$PYLINT_PATH/pylint_output.txt"; then
    echo "Pylint encountered issues or errors."
    exit 1
fi

echo " "
echo "--------- PYTEST /W COVERAGE ---------------------------"
echo " "
if ! pytest 2>&1 | tee "$PYTEST_PATH/pytest_output.txt"; then
    echo "Pytest encountered errors."
    exit 1
fi

deactivate
