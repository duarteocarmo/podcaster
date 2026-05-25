#!/bin/bash
set -e

cron
tail -f /var/log/cron.log
