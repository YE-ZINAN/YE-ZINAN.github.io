#run do in python

stata_path = '' #your path stata.exe (the exe of 64-bit! IMPORTANT)

#exmaple do file code:
stata_cmds = [
    'set maxvar 10000',
    'use '+oloca+", clear",
    'gen year = 2019',
    "keep hhid year "+question_code,
    'save '+interloca+", replace",
    "exit" #better to keep the "exit" at the end of the do file
]

do_file_path = r"path\to\stata_commands.do" #temp do-file
with open(do_file_path, "w", encoding='utf-8') as f:
    f.write("\n".join(stata_cmds))
cmd_file_path = r"path\to\stata_run.cmd" #temp cmd file
with open(cmd_file_path, "w", encoding='utf-8') as f:
    f.write(f'"{stata_path}" -b do "{do_file_path}"\n')
return_code = subprocess.call(["cmd.exe", "/c", cmd_file_path])

os.remove(do_file_path) #delete temp files
os.remove(cmd_file_path)
