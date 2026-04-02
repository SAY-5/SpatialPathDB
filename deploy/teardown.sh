#!/usr/bin/env bash
# Tear down AWS benchmark infrastructure
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
KEY_NAME="spdb-benchmark"
SG_NAME="spdb-benchmark-sg"

echo "=== SpatialPathDB AWS Teardown ==="

# Load connection info if available
if [[ -f deploy/connection.env ]]; then
    source deploy/connection.env
fi

# Terminate instance
if [[ -n "${INSTANCE_ID:-}" ]]; then
    echo "Terminating instance ${INSTANCE_ID}..."
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
    echo "  Waiting for termination..."
    aws ec2 wait instance-terminated --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
    echo "  Instance terminated."
else
    echo "  No instance ID found. Searching by name..."
    INSTANCE_ID=$(aws ec2 describe-instances --region "${REGION}" \
        --filters "Name=tag:Name,Values=spdb-benchmark" "Name=instance-state-name,Values=running,stopped" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")
    if [[ "${INSTANCE_ID}" != "None" ]] && [[ -n "${INSTANCE_ID}" ]]; then
        echo "  Found ${INSTANCE_ID}. Terminating..."
        aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
        aws ec2 wait instance-terminated --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
    else
        echo "  No running instances found."
    fi
fi

# Delete security group (may need to wait for instance to fully terminate)
sleep 5
SG_ID=$(aws ec2 describe-security-groups --region "${REGION}" \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [[ "${SG_ID}" != "None" ]] && [[ -n "${SG_ID}" ]]; then
    echo "Deleting security group ${SG_ID}..."
    aws ec2 delete-security-group --group-id "${SG_ID}" --region "${REGION}" || true
fi

# Delete key pair
echo "Deleting key pair ${KEY_NAME}..."
aws ec2 delete-key-pair --key-name "${KEY_NAME}" --region "${REGION}" || true
rm -f "${KEY_NAME}.pem"

# Clean up connection info
rm -f deploy/connection.env

echo ""
echo "=== Teardown complete ==="
