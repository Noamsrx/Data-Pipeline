#!/bin/sh

HOST="$1"
PORT="$2"

shift 2

echo "⏳ Attente de l'hôte $HOST:$PORT..."
while ! nc -z "$HOST" "$PORT"; do
  sleep 1
done

echo "✅ $HOST:$PORT est disponible. Exécution de la commande : $@"
exec "$@"
