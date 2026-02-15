#!/bin/bash
# deploy_production.sh
# Production Deployment Script for Grok AdMaster

set -e

echo "=== Grok AdMaster Production Deployment ==="

# 1. Create dedicated user
sudo useradd -r -s /bin/bash -d /opt/grok-admaster admaster || true

# 2. Setup directory structure
sudo mkdir -p /opt/grok-admaster/{logs,data,venv,server}
sudo chown -R admaster:admaster /opt/grok-admaster

# 3. Copy application files (Run this from project root)
sudo cp -r ./grok-admaster/server/* /opt/grok-admaster/server/
sudo chown -R admaster:admaster /opt/grok-admaster/server

# 4. Create virtual environment
sudo -u admaster python3 -m venv /opt/grok-admaster/venv
sudo -u admaster /opt/grok-admaster/venv/bin/pip install --upgrade pip
sudo -u admaster /opt/grok-admaster/venv/bin/pip install -r /opt/grok-admaster/server/requirements.txt

# 5. Install systemd service
sudo cp ./grok-admaster/deployment/grok-admaster.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable grok-admaster.service

# 6. Start service
sudo systemctl start grok-admaster.service

echo "=== Deployment Complete ==="
echo "Check status: sudo systemctl status grok-admaster"
echo "View logs: sudo journalctl -u grok-admaster -f"
