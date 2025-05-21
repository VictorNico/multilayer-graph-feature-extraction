#!/bin/bash

# Récupère la liste des sessions screen et les tue une par une
screen -ls | grep Detached | awk '{print $1}' | while read session; do
  echo "Terminaison de la session screen: $session"
  screen -X -S "$session" quit
done

echo "Toutes les sessions screen ont été terminées."
