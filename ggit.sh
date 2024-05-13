#!/bin/bash

# 获取.gitignore中的忽略规则
ignore_patterns=$(git ls-files --others -i --exclude-standard --directory)

# 查找当前目录及其子目录下大于50MB的文件，忽略.gitignore中的目录
large_files=$(find . -type f -size +50M \( -path "${ignore_patterns}" -o -wholename '*/.git/*' \) -prune -o -print)

# 检查是否有大文件
if [ -n "$large_files" ]; then
    # 如果有大文件，则输出黄色信息并退出
    echo "$large_files"
    echo -e "\033[33mFound files larger than 50MB. Here are the paths:\033[0m"

else
    # 如果没有找到大于50MB的文件，则继续执行
    current_time=$(date +%Y-%m-%d-%H:%M:%S)
    git add .
    git commit -m "new commit by ggit.sh. TIME:$current_time"
    git push origin main:main
fi
