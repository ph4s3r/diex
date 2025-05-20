#!/bin/bash

#####
#   about: Script to clone a git repo sparsely: filter for markdowns only
#  author: Peter Karacsonyi <peterkaracsonyi85@gmail.com>
#    date: 24 Nov 2024
# license: GNU General Public License, version 2
#####

# Set up repository URL and target directory
REPO_URL="https://github.com/MicrosoftDocs/azure-docs.git"
TARGET_DIR="azure-docs-md"

# Check if the directory already exists
if [ -d "$TARGET_DIR" ]; then
  echo "Directory $TARGET_DIR already exists. Pulling latest changes..."
  cd "$TARGET_DIR"

  # Update the shallow clone with sparse-checkout for only .md files
  git pull --depth=1 origin main
else
  echo "Cloning repository with sparse-checkout for only markdown files..."

  # Clone repository with sparse-checkout for only markdown files
  git init "$TARGET_DIR"
  cd "$TARGET_DIR"
  git remote add origin "$REPO_URL"
  git config core.sparseCheckout true

  # Specify only markdown files in sparse-checkout
  echo "**/*.md" > .git/info/sparse-checkout

  # Perform a shallow clone (latest commit only)
  git pull --depth=1 origin main
fi

echo "Shallow clone complete with only markdown files retained in structure."

# Display the actual size of the azure-docs directory
echo "Calculating directory size for $TARGET_DIR..."
cd ..
du -sh "$TARGET_DIR"