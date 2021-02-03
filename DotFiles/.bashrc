# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# For using vi key binidngs 
set -o vi
# Allows for clear screen in vi-insert mode 
bind -m vi-insert "\C-l":clear-screen
# Extends tab-autocompletion to commands and directories without commands 
complete -cf sudo

# Show current git branch in bash prompt
parse_git_branch() {
	     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
     }

#export PS1="\u@\h \[\033[32m\]\w\[\033[33m\]\$(parse_git_branch)\[\033[00m\] $ "
export PS1="\e[01;34m\e[01;34m[\u@\h \[\033[32m\]\w\[\033[33m\]\$(parse_git_branch)\[\033[00m\] $ "


# To start the agent automatically and make sure that only one ssh-agent process runs at a time
if ! pgrep -u "$USER" ssh-agent > /dev/null; then
	    ssh-agent -t 1h > "$XDG_RUNTIME_DIR/ssh-agent.env"
    fi
    if [[ ! "$SSH_AUTH_SOCK" ]]; then
	        source "$XDG_RUNTIME_DIR/ssh-agent.env" >/dev/null
	fi

module load CUDA/11.0
module load GCC/9.3.0
module load cmake


export CUDA_PATH=/opt/apps/rhel7/cuda-11.0
export PATH=$PATH:$HOME/Code/genn/bin


# Adding second 'bin' to path 
export PATH=~/Code/bin:$PATH


# User specific aliases and functions
export PATH=~/.local/bin:$PATH
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=~/.local/lib:$LIBRARY_PATH
export CPATH=~/.local/include:$CPATH
# Pointing towrads local python file 
#export PATH=/hpc/home/kbw29/.local/bin/Python-3.9.1/:$PATH
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/apps/rhel7/anaconda3/5.1.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/rhel7/anaconda3/5.1.0/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/rhel7/anaconda3/5.1.0/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/rhel7/anaconda3/5.1.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
