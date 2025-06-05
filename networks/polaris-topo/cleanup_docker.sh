#!/bin/bash

echo "🛑 Stopping all running containers..."
if [ "$(docker ps -q)" ]; then
    docker stop $(docker ps -q)
else
    echo "No running containers found."
fi

echo "🗑️  Removing all containers..."
if [ "$(docker ps -aq)" ]; then
    docker rm $(docker ps -aq)
else
    echo "No containers to remove."
fi

echo "🧹 Removing dangling images..."
docker image prune -f

echo "🌐 Removing unused networks..."
docker network prune -f

echo "✅ Cleanup complete!"
