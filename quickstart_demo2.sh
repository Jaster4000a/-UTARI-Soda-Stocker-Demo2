#!/bin/bash
python bottle_stocker_v4.py &
python bottle_subscriber_v3.py &
python blinking2.py &
trap 'kill $(jobs -p)' SIGINT
wait

