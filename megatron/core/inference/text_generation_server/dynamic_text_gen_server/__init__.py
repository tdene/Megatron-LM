# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from .quart_server import run_server, run_server_on_client

# Backward-compatible aliases
run_flask_server = run_server
run_flask_server_on_client = run_server_on_client
