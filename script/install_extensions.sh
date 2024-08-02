#! /bin/bash
function has_pkg(){
    local find_pkg=0
    for pkg_name in $1
    do
        result=$(echo $pkg_name | grep "$2")
        if [[ $result != "" ]]; then
            find_pkg=1
        fi
    done
    return $find_pkg
}

function export_text(){
    # 检查 ~/.zshrc 文件中是否已经存在
    if ! grep -q "$1" ~/.zshrc; then
        # 如果不存在，追加到文件末尾
        echo "$1" >> ~/.zshrc
        echo "Text '$1' has been appended to ~/.zshrc"
    else
        echo "Text '$1' already exists in ~/.zshrc"
    fi
}

# install zsh
all_shells=$(cat /etc/shells)
has_pkg "${all_shells[@]}" '/bin/zsh'
find_zsh=$?

# install zsh
if [[ $find_zsh == 0 ]]; then
    apt-get update
    apt-get install -y zsh
    git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh
    cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
    chsh -s /bin/zsh
    echo "successfully installed zsh"
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
    # replace ZSH theme with powerlevel10k
    ## 文件路径
    file_path="/root/.zshrc"  # 替换为要修改的文件路径
    ## 要替换的旧文本模式和新文本
    old_pattern='ZSH_THEME=".*"'  # 匹配 ZSH_THEME="任何内容"
    new_text='ZSH_THEME="powerlevel10k/powerlevel10k"'

    ## 使用 sed 替换文件中的文本
    if grep -q 'ZSH_THEME=' "$file_path"; then
        sed -i "s|$old_pattern|$new_text|" "$file_path"
        echo "已经把 ZSH_THEME 替换成 powerlevel10k/powerlevel10k"
    else
        echo "无法找到要替换的 ZSH_THEME"
    fi
else
    echo "has installed zsh"
fi

# install black
pip_list=$(pip3 list)
has_pkg "$pip_list" 'black'
has_black=$?
if [[ $has_black == 0 ]]; then
    echo "start installing black"
    pip3 install black==23.12.1
else
    echo "has installed black"
fi

# install gymnasium
pip_list=$(pip3 list)
has_pkg "$pip_list" 'gymnasium'
has_gymnasium=$?
if [[ $has_gymnasium == 0 ]]; then
    echo "start installing gymnasium"
    pip3 install gymnasium
else
    echo "has installed gymnasium"
fi

# install tmux
has_tmux=$(which tmux)
if [[ $has_tmux == "" ]]; then
    apt-get install -y tmux
    echo "successfully installed tmux"
else
    echo "has installed tmux"
fi

# install MPI
pip_list=$(pip3 list)
has_pkg "$pip_list" 'mpi4py'
has_mpi4py=$?
if [[ $has_mpi4py == 0 ]]; then
    echo "start installing mpi4py"
    apt-get install -y libopenmpi-dev libopenblas-base
    pip3 install mpi4py
    echo "has installed mpi4py"
else
    echo "has installed mpi4py"
fi
export_text "export OMPI_ALLOW_RUN_AS_ROOT=1"
export_text "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
source ~/.zshrc

# install vscode extensions
plugins=(
    GitHub.copilot
    eamodio.gitlens
    mhutchie.git-graph
    ms-python.python
    doi.fileheadercomment
    alefragnani.Bookmarks
    ms-python.black-formatter
)

## 遍历插件列表并安装每个插件
for plugin in "${plugins[@]}"
do
    echo "Installing plugin: $plugin"
    code --install-extension $plugin
    if [ $? -eq 0 ]; then
        echo "Successfully installed $plugin"
    else
        echo "Failed to install $plugin"
    fi
done