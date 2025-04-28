#!/bin/bash

# Entrypoint script for the FL server

FL_SERVER_TIMEOUT=${FL_SERVER_TIMEOUT:-900}

timeout $FL_SERVER_TIMEOUT python3 server.py

sleep INF