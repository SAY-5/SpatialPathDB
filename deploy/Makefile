# SpatialPathDB AWS Benchmark Deployment
# Usage:
#   make setup      - Provision AWS EC2 instance
#   make deploy     - rsync code to server
#   make benchmark  - Run full benchmark suite
#   make results    - Download results from server
#   make teardown   - Clean up AWS resources

include deploy/connection.env 2>/dev/null

SSH = ssh -i $(KEY_FILE) -o StrictHostKeyChecking=no ubuntu@$(PUBLIC_IP)
SCP = scp -i $(KEY_FILE) -o StrictHostKeyChecking=no
RSYNC = rsync -avz --exclude='.git' --exclude='data_cache' --exclude='__pycache__' \
        --exclude='*.pyc' --exclude='results/raw' \
        -e "ssh -i $(KEY_FILE) -o StrictHostKeyChecking=no"

.PHONY: setup deploy benchmark results teardown status

setup:
	bash deploy/aws_setup.sh

deploy:
	@echo "Syncing code to server..."
	$(RSYNC) . ubuntu@$(PUBLIC_IP):~/SpatialPathDB_CLEAN/
	$(SSH) "cd ~/SpatialPathDB_CLEAN && pip3 install -r requirements.txt"
	@echo "Deploy complete."

ingest:
	@echo "Running data ingestion..."
	$(SSH) "cd ~/SpatialPathDB_CLEAN && python3 run_ingest.py"

benchmark:
	@echo "Running isolated benchmarks..."
	$(SSH) "cd ~/SpatialPathDB_CLEAN && python3 deploy/benchmark_linux.py --trials 500"

benchmark-extended:
	@echo "Running extended benchmarks..."
	$(SSH) "cd ~/SpatialPathDB_CLEAN && python3 run_benchmarks.py extended"

benchmark-all:
	@echo "Running complete benchmark suite..."
	$(SSH) "cd ~/SpatialPathDB_CLEAN && python3 run_benchmarks.py all"

results:
	@echo "Downloading results..."
	$(RSYNC) ubuntu@$(PUBLIC_IP):~/SpatialPathDB_CLEAN/results/ ./results/
	@echo "Results downloaded to ./results/"

status:
	@echo "Server status:"
	$(SSH) "uptime && echo '' && df -h / && echo '' && psql -d spdb -c 'SELECT COUNT(*) FROM objects_mono;'"

teardown:
	@echo "WARNING: This will terminate the EC2 instance and delete all data."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	bash deploy/teardown.sh
