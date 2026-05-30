#!/bin/bash
# deploy_to_hf.sh — Run this once on your local machine to push to Hugging Face Spaces
# Usage: bash deploy_to_hf.sh YOUR_HF_USERNAME

set -e

USERNAME=${1:-"your-username"}
SPACE_NAME="cancer-mutation-detector"
REPO="$USERNAME/$SPACE_NAME"

echo "========================================"
echo " Deploying to Hugging Face Space"
echo " Repo: $REPO"
echo "========================================"

# 1. Login to HF
echo ""
echo "Step 1: Login to Hugging Face"
huggingface-cli login

# 2. Create the Space (skip if already exists)
echo ""
echo "Step 2: Creating Space '$REPO' ..."
huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk gradio || echo "Space may already exist, continuing..."

# 3. Clone the Space repo
echo ""
echo "Step 3: Cloning Space repo ..."
git clone "https://huggingface.co/spaces/$REPO" hf_space_deploy
cd hf_space_deploy

# 4. Copy all files
echo ""
echo "Step 4: Copying project files ..."
cp ../app.py .
cp ../requirements.txt .
cp ../README.md .
cp ../sample_mutations.maf .

# 5. Push
echo ""
echo "Step 5: Pushing to Hugging Face ..."
git add .
git commit -m "Deploy cancer mutation detector"
git push

cd ..
rm -rf hf_space_deploy

echo ""
echo "========================================"
echo " Done! Visit your Space at:"
echo " https://huggingface.co/spaces/$REPO"
echo "========================================"
