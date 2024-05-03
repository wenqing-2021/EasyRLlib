#!/bin/bash
echo "Format python files..."
package_name="black"

if pip show "$package_name" >/dev/null 2>&1; then
    echo "$package_name is installed."
else
    echo "$package_name is not installed."
    echo "Installing $package_name..."
    pip3 install "$package_name"
fi

# format python file
python3 -m black .