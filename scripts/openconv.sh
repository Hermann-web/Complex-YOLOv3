#!/bin/bash

# Name your session
SESSION="fileconv-session"

# Create a new tmux session, detached (-d), named $SESSION. If it exists, don't create a new one.
tmux has-session -t $SESSION 2>/dev/null || tmux new-session -d -s $SESSION

# env setup
CMD_SETUP_PRIM="uv pip install pandas pillow pypdf2 opencv-python python-magic --upgrade"
CMD_SETUP_FILE="pip install -i https://test.pypi.org/simple/ openconv-python --upgrade"

# Command to run in the tmux session, using the provided folder path
CMD="openconv $@ > ~.fileconv.out.log"

# Send the commands to the tmux session. This activates the conda environment and runs your script.
tmux send-keys -t $SESSION "conda activate fileconv" C-m
# tmux send-keys -t $SESSION "$CMD_SETUP_PRIM" C-m
# tmux send-keys -t $SESSION "$CMD_SETUP_FILE" C-m
tmux send-keys -t $SESSION "$CMD" C-m

# Optional: Send a command to exit after the previous commands complete. Remove the '#' to enable.
# tmux send-keys -t $SESSION "exit" C-m

# Detach from the session
tmux detach -s $SESSION

# Uncomment the following line if you want to kill the session after the command executes
# tmux kill-session -t $SESSION
