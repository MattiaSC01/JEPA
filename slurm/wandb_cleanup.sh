#!/bin/bash

wandb artifact cache cleanup 1GB  # clean artifact cache
wandb sync --clean # clean synced runs older than 24 hours