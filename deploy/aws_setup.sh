#!/usr/bin/env bash
# SpatialPathDB AWS Benchmark Server Setup
# Provisions an r5.4xlarge EC2 instance with PostgreSQL 17 + PostGIS 3.6
# configured for database benchmarking.
#
# Usage:
#   ./deploy/aws_setup.sh              # Full setup
#   ./deploy/aws_setup.sh --key-only   # Just create key pair
#
# Prerequisites: aws cli configured with appropriate credentials

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INSTANCE_TYPE="r5.4xlarge"      # 16 vCPUs, 128GB RAM
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS us-east-1 (update for region)
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
KEY_NAME="spdb-benchmark"
SG_NAME="spdb-benchmark-sg"
INSTANCE_NAME="spdb-benchmark"
VOLUME_SIZE=500                 # GB, gp3 NVMe
VOLUME_IOPS=10000
VOLUME_THROUGHPUT=500           # MB/s

PG_VERSION=17

echo "=== SpatialPathDB AWS Benchmark Setup ==="
echo "  Instance: ${INSTANCE_TYPE}"
echo "  Region:   ${REGION}"
echo "  Volume:   ${VOLUME_SIZE}GB gp3 (${VOLUME_IOPS} IOPS, ${VOLUME_THROUGHPUT} MB/s)"
echo ""

# ---------------------------------------------------------------------------
# Key pair
# ---------------------------------------------------------------------------
if ! aws ec2 describe-key-pairs --key-names "${KEY_NAME}" --region "${REGION}" &>/dev/null; then
    echo "Creating key pair ${KEY_NAME}..."
    aws ec2 create-key-pair \
        --key-name "${KEY_NAME}" \
        --region "${REGION}" \
        --query 'KeyMaterial' \
        --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "  Key saved to ${KEY_NAME}.pem"
else
    echo "  Key pair ${KEY_NAME} already exists"
fi

[[ "${1:-}" == "--key-only" ]] && exit 0

# ---------------------------------------------------------------------------
# Security group
# ---------------------------------------------------------------------------
VPC_ID=$(aws ec2 describe-vpcs --region "${REGION}" \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)

SG_ID=$(aws ec2 describe-security-groups --region "${REGION}" \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [[ "${SG_ID}" == "None" ]] || [[ -z "${SG_ID}" ]]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "${SG_NAME}" \
        --description "SpatialPathDB benchmark server" \
        --vpc-id "${VPC_ID}" \
        --region "${REGION}" \
        --query 'GroupId' --output text)

    # SSH access
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp --port 22 \
        --cidr "0.0.0.0/0" \
        --region "${REGION}"
    echo "  Security group: ${SG_ID}"
else
    echo "  Security group already exists: ${SG_ID}"
fi

# ---------------------------------------------------------------------------
# Launch instance
# ---------------------------------------------------------------------------
echo "Launching ${INSTANCE_TYPE} instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --region "${REGION}" \
    --block-device-mappings "[{
        \"DeviceName\":\"/dev/sda1\",
        \"Ebs\":{
            \"VolumeSize\":${VOLUME_SIZE},
            \"VolumeType\":\"gp3\",
            \"Iops\":${VOLUME_IOPS},
            \"Throughput\":${VOLUME_THROUGHPUT},
            \"DeleteOnTermination\":true
        }
    }]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "  Instance ID: ${INSTANCE_ID}"
echo "  Waiting for running state..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${REGION}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "  Public IP: ${PUBLIC_IP}"
echo ""

# Save connection info
cat > deploy/connection.env <<EOF
INSTANCE_ID=${INSTANCE_ID}
PUBLIC_IP=${PUBLIC_IP}
KEY_FILE=${KEY_NAME}.pem
SSH_CMD="ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
EOF

echo "  Connection info saved to deploy/connection.env"
echo "  SSH: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""

# ---------------------------------------------------------------------------
# Wait for SSH and run remote setup
# ---------------------------------------------------------------------------
echo "Waiting for SSH to be ready..."
for i in $(seq 1 30); do
    if ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
       ubuntu@"${PUBLIC_IP}" "echo ready" &>/dev/null; then
        break
    fi
    sleep 10
done

echo "Running remote setup..."
ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no ubuntu@"${PUBLIC_IP}" bash <<'REMOTE_SETUP'
set -euo pipefail

echo "=== Remote Server Setup ==="

# System updates
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

# PostgreSQL 17 from PGDG
sudo apt-get install -y -qq curl ca-certificates gnupg lsb-release
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo gpg --dearmor -o /usr/share/keyrings/pgdg.gpg
echo "deb [signed-by=/usr/share/keyrings/pgdg.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    | sudo tee /etc/apt/sources.list.d/pgdg.list
sudo apt-get update -qq
sudo apt-get install -y -qq postgresql-17 postgresql-17-postgis-3

# Python
sudo apt-get install -y -qq python3-pip python3-venv python3-dev libpq-dev

# PostgreSQL configuration for benchmarking
sudo tee /etc/postgresql/17/main/conf.d/benchmark.conf > /dev/null <<PGCONF
# SpatialPathDB benchmark configuration
shared_buffers = 32GB
effective_cache_size = 96GB
work_mem = 1GB
maintenance_work_mem = 4GB
max_connections = 200
random_page_cost = 1.1
effective_io_concurrency = 200
max_parallel_workers_per_gather = 4
max_worker_processes = 16
max_parallel_workers = 16
wal_buffers = 64MB
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB
huge_pages = try
shared_preload_libraries = ''
track_io_timing = on
PGCONF

sudo systemctl restart postgresql

# Create database and user
sudo -u postgres createuser --superuser ubuntu 2>/dev/null || true
sudo -u postgres createdb spdb 2>/dev/null || true
sudo -u postgres psql -d spdb -c "CREATE EXTENSION IF NOT EXISTS postgis;" 2>/dev/null || true

# Verify
echo ""
echo "=== Verification ==="
psql -d spdb -c "SELECT version();"
psql -d spdb -c "SELECT PostGIS_Version();"
python3 --version

# Create cache flush helper
sudo tee /usr/local/bin/flush-caches > /dev/null <<'FLUSH'
#!/bin/bash
sync
echo 3 > /proc/sys/vm/drop_caches
echo "OS caches flushed"
FLUSH
sudo chmod +x /usr/local/bin/flush-caches

# Allow ubuntu user to flush caches without password
echo "ubuntu ALL=(root) NOPASSWD: /usr/local/bin/flush-caches" | sudo tee /etc/sudoers.d/flush-caches
echo "ubuntu ALL=(root) NOPASSWD: /bin/systemctl restart postgresql" | sudo tee -a /etc/sudoers.d/flush-caches

echo ""
echo "=== Remote setup complete ==="
REMOTE_SETUP

echo ""
echo "=== AWS Setup Complete ==="
echo "  Instance: ${INSTANCE_ID}"
echo "  IP: ${PUBLIC_IP}"
echo "  SSH: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  1. make deploy    # rsync code to server"
echo "  2. make benchmark # run full benchmark suite"
echo "  3. make results   # download results"
