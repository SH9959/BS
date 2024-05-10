import os
import sys
print("asdasd")
# os.execv("/usr/bin/python", ["python", "-u", "test.py"])
os.execv(sys.executable, ['python'] + sys.argv) 
#print(sys.executable, ['python -u'] + sys.argv)
# 当前进程被替换，以下代码不会执行
print("restart ")
