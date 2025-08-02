#!/bin/sh

echo "PORT is: $PORT"
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT
