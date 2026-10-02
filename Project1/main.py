import subprocess
import os

filepath = os.path.dirname(os.path.abspath(__file__))
pypath = lambda path: os.path.join(filepath, "src/" + path + ".py")

for a in ["A", "B","C", "D", "E", "F", "G", "H", "I"]:
    proc = subprocess.Popen(['python3', pypath(a)])
    proc.wait()