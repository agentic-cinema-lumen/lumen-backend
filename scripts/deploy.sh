#!/usr/bin/env bash
# ==============================================================================
# Lumen Prediction API — One-Click Google Cloud Run Deployment
# ==============================================================================
set -euo pipefail

SERVICE_NAME="lumen-api"
REGION="europe-north1"
MEMORY="2Gi"
CPU="2"
TIMEOUT="300s"

echo "======================================================================"
echo "🎬 Deploying ${SERVICE_NAME} to Google Cloud Run (${REGION})..."
echo "======================================================================"

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed or not in PATH."
    echo "   Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    echo "   Or submit via Cloud Build in the Google Cloud Console."
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
if [[ -z "${PROJECT_ID}" ]]; then
    echo "❌ Error: No default Google Cloud project set."
    echo "   Run: gcloud config set project <YOUR_PROJECT_ID>"
    exit 1
fi

echo "📦 Project: ${PROJECT_ID}"
echo "📍 Region:  ${REGION}"
echo "🚀 Submitting build to Google Cloud Build & deploying to Cloud Run..."

# Deploy directly from source directory using Cloud Build
gcloud run deploy "${SERVICE_NAME}" \
    --source . \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory "${MEMORY}" \
    --cpu "${CPU}" \
    --timeout "${TIMEOUT}" \
    --set-env-vars="LUMEN_AGENT_MODEL=gemini-2.0-flash"

echo ""
echo "✅ Deployment complete! Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)"
